/*
 * gpu_pipeline_pinned.cu
 *
 * GPU Spectrum Sensing Pipeline using Pinned Memory + Async Stream (MEX CUDA)
 *
 * Processing pipeline:
 *   [H2D] Pinned + Async host-to-device transfer
 *   [1]   Hamming Windowing
 *   [2]   FFT (cuFFT batch)
 *   [3]   Magnitude
 *   [4]   Moving Average (CFAR, window=16)
 *   [5]   Threshold Detection & Count
 *   [D2H] Async device-to-host result retrieval
 *
 * Compile (MATLAB):
 *   mexcuda -R2018a gpu_pipeline_pinned.cu -lcufft -output gpu_pipeline_pinned
 *
 * Usage:
 *   [count, time_s] = gpu_pipeline_pinned(data, threshold)
 *   data      : complex single matrix (fft_size x M)
 *   threshold : detection threshold (scalar)
 *   count     : number of detections (scalar)
 *   time_s    : total transfer + compute time (seconds)
 */

#include "mex.h"
#include "cuda_runtime.h"
#include "cufft.h"
#include <math.h>
#include <string.h>

#define CFAR_HALF_WIN  8    /* movmean window=16 -> half=8 */
#define BLOCK_DIM_1D   256

/* ------------------------------------------------------------------ */
/*  Error check macros                                                  */
/* ------------------------------------------------------------------ */
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t _e = (call);                                           \
        if (_e != cudaSuccess) {                                           \
            char _msg[256];                                                \
            snprintf(_msg, sizeof(_msg), "CUDA error at line %d: %s",     \
                     __LINE__, cudaGetErrorString(_e));                    \
            mexErrMsgIdAndTxt("gpu_pipeline:cuda", _msg);                 \
        }                                                                  \
    } while (0)

#define CUFFT_CHECK(call)                                                  \
    do {                                                                   \
        cufftResult _r = (call);                                           \
        if (_r != CUFFT_SUCCESS) {                                         \
            char _msg[64];                                                 \
            snprintf(_msg, sizeof(_msg),                                   \
                     "cuFFT error at line %d (code %d)", __LINE__, _r);   \
            mexErrMsgIdAndTxt("gpu_pipeline:cufft", _msg);                \
        }                                                                  \
    } while (0)

/* ------------------------------------------------------------------ */
/*  Hamming window coefficient generation (host)                        */
/* ------------------------------------------------------------------ */
static void make_hamming(float* win, int N)
{
    for (int i = 0; i < N; i++)
        win[i] = 0.54f - 0.46f * cosf(2.0f * 3.14159265358979f * i / (N - 1));
}

/* ================================================================== */
/*  CUDA Kernels                                                        */
/* ================================================================== */

/*
 * kernel_window
 * Applies Hamming window to complex input data (element-wise)
 * Data layout: column-major (fft_size x M) -> col * fft_size + row
 */
__global__ void kernel_window(cuComplex* __restrict__ d,
                               const float* __restrict__ win,
                               int fft_size, int M)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= fft_size || col >= M) return;

    float w   = win[row];
    int   idx = col * fft_size + row;
    d[idx].x *= w;
    d[idx].y *= w;
}

/*
 * kernel_magnitude
 * Computes |complex| = sqrt(re^2 + im^2) -> float array
 */
__global__ void kernel_magnitude(const cuComplex* __restrict__ in,
                                  float*          __restrict__ out,
                                  int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float re = in[i].x, im = in[i].y;
    out[i] = sqrtf(re * re + im * im);
}

/*
 * kernel_movmean
 * Moving average along frequency axis (row direction) of magnitude array
 * Equivalent to MATLAB movmean(X, 16, 1)
 */
__global__ void kernel_movmean(const float* __restrict__ in,
                                float*       __restrict__ out,
                                int fft_size, int M, int half_win)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= fft_size || col >= M) return;

    int r0 = (row - half_win < 0)          ? 0          : row - half_win;
    int r1 = (row + half_win >= fft_size)  ? fft_size-1 : row + half_win;

    const float* col_ptr = in + col * fft_size;
    float sum = 0.0f;
    for (int r = r0; r <= r1; r++) sum += col_ptr[r];

    out[col * fft_size + row] = sum / (float)(r1 - r0 + 1);
}

/*
 * kernel_detect
 * Counts elements where cfar[i] > threshold using atomicAdd
 */
__global__ void kernel_detect(const float* __restrict__ cfar,
                               int*         d_count,
                               float        threshold,
                               int          n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (cfar[i] > threshold) atomicAdd(d_count, 1);
}

/* ================================================================== */
/*  MEX Entry Point                                                     */
/* ================================================================== */
void mexFunction(int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[])
{
    /* ---- Input validation ---- */
    if (nrhs < 2)
        mexErrMsgIdAndTxt("gpu_pipeline:input",
            "Usage: [count, time_s] = gpu_pipeline_pinned(data, threshold)");
    if (!mxIsSingle(prhs[0]) || !mxIsComplex(prhs[0]))
        mexErrMsgIdAndTxt("gpu_pipeline:input",
            "data must be a complex single matrix.");

    /* ---- Parse dimensions ---- */
    int    fft_size = (int)mxGetM(prhs[0]);
    int    M        = (int)mxGetN(prhs[0]);
    int    N_total  = fft_size * M;
    size_t bytes    = (size_t)N_total * sizeof(cuComplex);
    float  threshold = (float)mxGetScalar(prhs[1]);

    /* ---- MATLAB complex pointer (-R2018a interleaved) ---- */
    /* mxComplexSingle = {float real; float imag;} matches cuComplex layout */
    mxComplexSingle* h_src = mxGetComplexSingles(prhs[0]);

    /* ================================================================
     * Allocate pinned host memory
     * ================================================================ */
    cuComplex* h_pinned = NULL;
    int*       h_count  = NULL;
    float*     h_win    = (float*)malloc(fft_size * sizeof(float));
    if (!h_win) mexErrMsgIdAndTxt("gpu_pipeline:malloc", "malloc failed");

    CUDA_CHECK(cudaMallocHost((void**)&h_pinned, bytes));
    CUDA_CHECK(cudaMallocHost((void**)&h_count,  sizeof(int)));
    *h_count = 0;

    /* Copy MATLAB interleaved complex to pinned buffer
     * (same layout: real, imag, real, imag, ... = cuComplex x, y, x, y ...) */
    memcpy(h_pinned, h_src, bytes);

    /* Generate Hamming window coefficients */
    make_hamming(h_win, fft_size);

    /* ================================================================
     * Allocate GPU memory
     * ================================================================ */
    cuComplex* d_data  = NULL;
    float*     d_win   = NULL;
    float*     d_mag   = NULL;
    float*     d_cfar  = NULL;
    int*       d_count = NULL;

    CUDA_CHECK(cudaMalloc((void**)&d_data,  bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_win,   fft_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_mag,   N_total  * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_cfar,  N_total  * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_count, sizeof(int)));

    /* ================================================================
     * cuFFT plan (batch: M C2C FFTs of size fft_size)
     * column-major array: stride=1, dist=fft_size -> default cuFFT layout
     * ================================================================ */
    cufftHandle plan;
    CUFFT_CHECK(cufftPlan1d(&plan, fft_size, CUFFT_C2C, M));

    /* ================================================================
     * Create CUDA stream
     * ================================================================ */
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUFFT_CHECK(cufftSetStream(plan, stream));

    /* ================================================================
     * Timing events
     * ================================================================ */
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    /* Transfer window coefficients and reset count before timing */
    CUDA_CHECK(cudaMemcpyAsync(d_win, h_win, fft_size * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemsetAsync(d_count, 0, sizeof(int), stream));

    /* ================================================================
     * Start timing — includes H2D data transfer
     * ================================================================ */
    CUDA_CHECK(cudaEventRecord(ev_start, stream));

    /* [H2D] Pinned + Async transfer */
    CUDA_CHECK(cudaMemcpyAsync(d_data, h_pinned, bytes,
                               cudaMemcpyHostToDevice, stream));

    /* [1] Hamming Windowing */
    dim3 blk2d(32, 8);
    dim3 grd2d((fft_size + 31) / 32, (M + 7) / 8);
    kernel_window<<<grd2d, blk2d, 0, stream>>>(d_data, d_win, fft_size, M);

    /* [2] FFT (in-place) */
    CUFFT_CHECK(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));

    /* [3] Magnitude */
    int grd1d = (N_total + BLOCK_DIM_1D - 1) / BLOCK_DIM_1D;
    kernel_magnitude<<<grd1d, BLOCK_DIM_1D, 0, stream>>>(d_data, d_mag, N_total);

    /* [4] Moving Average (CFAR, window=16) */
    kernel_movmean<<<grd2d, blk2d, 0, stream>>>(
        d_mag, d_cfar, fft_size, M, CFAR_HALF_WIN);

    /* [5] Detection & Count */
    kernel_detect<<<grd1d, BLOCK_DIM_1D, 0, stream>>>(
        d_cfar, d_count, threshold, N_total);

    /* [D2H] Async result retrieval */
    CUDA_CHECK(cudaMemcpyAsync(h_count, d_count, sizeof(int),
                               cudaMemcpyDeviceToHost, stream));

    /* ================================================================
     * Stop timing — single CPU sync point
     * ================================================================ */
    CUDA_CHECK(cudaEventRecord(ev_stop, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));

    /* ---- MATLAB output ---- */
    if (nlhs >= 1) plhs[0] = mxCreateDoubleScalar((double)(*h_count));
    if (nlhs >= 2) plhs[1] = mxCreateDoubleScalar((double)(ms * 1e-3f));

    /* ---- Cleanup ---- */
    free(h_win);
    CUDA_CHECK(cudaFreeHost(h_pinned));
    CUDA_CHECK(cudaFreeHost(h_count));
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_win));
    CUDA_CHECK(cudaFree(d_mag));
    CUDA_CHECK(cudaFree(d_cfar));
    CUDA_CHECK(cudaFree(d_count));
    cufftDestroy(plan);
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    CUDA_CHECK(cudaStreamDestroy(stream));
}
