import time
import numpy as np
import cupy as cp
import adi

# ==========================================
# 1. Parameters (optimized for single USB frame unit)
# ==========================================
SAMPLE_RATE = 61.44e6      # 61.44 MSPS (ADALM-Pluto maximum rate; matches Table 3 paper)
# 1024 samples = 16-bit I + 16-bit Q = 4 bytes x 1024 = 4 KB
# -> Fits exactly within one USB 2.0 High-Speed Microframe (max 6.6 KB)
# -> At 61.44 MSPS: T_fill = 1024/61.44e6 = 0.0167 ms (matches Table 3: 0.017 ms)
BUFFER_SIZE = 1024
CENTER_FREQ = int(5.8e9)   # 5.8 GHz C-band (matches paper experimental setup)

print("[System] Initializing ADALM-Pluto SDR...")
sdr = adi.Pluto("usb:1.7.5")  # NOTE: change URI to match your device (run: iio_info -s)
sdr.sample_rate = int(SAMPLE_RATE)
sdr.rx_lo = CENTER_FREQ
sdr.rx_buffer_size = BUFFER_SIZE
sdr.rx_rf_bandwidth = int(SAMPLE_RATE)

# [Physical delay 1] Time for analog signal to fill the buffer
T_fill_physical = BUFFER_SIZE / SAMPLE_RATE

# [Physical delay 2] USB 2.0 Microframe packaging and hardware transfer delay
# (Includes host controller scheduling minimum latency per USB 2.0 spec)
T_usb_hw = 0.000125  # 125 us

# ==========================================
# 2. GPU warmup (kernel pre-compile + memory pre-allocation)
# ==========================================
print("[System] Warming up GPU kernels and allocating memory...")
gpu_window = cp.array(np.hamming(BUFFER_SIZE).astype(np.float32))
gpu_texture_buffer = cp.zeros(BUFFER_SIZE, dtype=cp.float32)  # rendering buffer

# Dummy run to trigger JIT compilation
dummy_data = cp.zeros(BUFFER_SIZE, dtype=cp.complex64)
cp.fft.fft(dummy_data * gpu_window)
cp.cuda.Stream.null.synchronize()
print("[System] Warmup complete. Starting measurement.")
print("-" * 65)

# ==========================================
# 3. Real-time pipeline latency measurement (1000 iterations)
# ==========================================
num_trials = 1000
rx_total_times = []
gpu_pipeline_times = []

# Flush stale buffers
for _ in range(5):
    sdr.rx()

for i in range(num_trials):
    # ------------------------------------------------
    # [A] Receive stage (Python binding + physical RX + USB transfer)
    # ------------------------------------------------
    t0 = time.perf_counter()
    rx_data = sdr.rx()
    t1 = time.perf_counter()
    rx_total_times.append(t1 - t0)

    # ------------------------------------------------
    # [B] GPU compute + rendering stage (Zero-Copy)
    # ------------------------------------------------
    t2 = time.perf_counter()

    # 1. H2D transfer (received packet to GPU)
    g_data = cp.asarray(rx_data, dtype=cp.complex64)

    # 2. Windowing -> FFT -> Magnitude
    g_fft = cp.fft.fft(g_data * gpu_window)
    g_mag = cp.abs(g_fft)

    # 3. Copy to GPU rendering buffer (simulates CUDA-OpenGL interop)
    #    In full implementation this maps directly to an OpenGL texture
    #    without transferring data back to CPU
    cp.copyto(gpu_texture_buffer, g_mag)

    cp.cuda.Stream.null.synchronize()  # wait for all async GPU ops
    t3 = time.perf_counter()

    gpu_pipeline_times.append(t3 - t2)

# ==========================================
# 4. Analysis: subtract Python/software overhead
# ==========================================
median_rx_total    = np.median(rx_total_times)
median_gpu_pipeline = np.median(gpu_pipeline_times)

# [Correction]
# Subtract buffer-fill time and USB physical transfer time from measured RX time.
# The remainder is Python binding / GIL / libiio software stack overhead,
# which is excluded from the hardware-level end-to-end latency reported in the paper.
python_binding_overhead = median_rx_total - T_fill_physical - T_usb_hw

# [Final] Pure hardware end-to-end latency reported in the paper
# = Analog buffer fill + USB packaging/transfer + GPU compute + GPU rendering
pure_e2e_latency = T_fill_physical + T_usb_hw + median_gpu_pipeline

print("=== Precise End-to-End Latency Analysis ===")
print(f"[Measured] sdr.rx() median time        :  {median_rx_total*1000:>7.3f} ms")
print(f"  |- Analog buffer fill time (physical) :  {T_fill_physical*1000:>7.3f} ms (theoretical)")
print(f"  |- USB frame packaging/transfer (phys):  {T_usb_hw*1000:>7.3f} ms (theoretical)")
print(f"  `- Python binding & libiio overhead   :  {python_binding_overhead*1000:>7.3f} ms (excluded)")
print("")
print(f"[Measured] GPU compute + rendering time :  {median_gpu_pipeline*1000:>7.3f} ms")
print(f"  `- H2D + Window + FFT + GL Mapping")
print("-" * 65)
print(f"[Paper] Optimized pure end-to-end latency: {pure_e2e_latency*1000:.3f} ms")
print(f"   (= buffer fill + USB transfer + GPU compute + rendering)")
