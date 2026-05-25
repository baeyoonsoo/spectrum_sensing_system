#ifndef CA_CFAR_H_
#define CA_CFAR_H_

#include <ap_int.h>
#include <hls_stream.h>

// number of training (reference) cells on each side
#define TRAIN_CELLS 32
// number of guard cells on each side
#define GUARD_CELLS 12
// total window size (training + guard + cell under test)
#define WINDOW_SIZE (2*TRAIN_CELLS + 2*GUARD_CELLS + 1)

// scaling constant for threshold: k = 9.9063
// derived from CA-CFAR false alarm formula (1 + k/N)^(-N) = Pfa, N=64, Pfa=1e-4
// k = 64 * (1e-4^(-1/64) - 1) = 9.906  =>  K_NUM = round(9.906 * 256) = 2536
// we compute threshold = (average * K_NUM) >> K_SHIFT
#define K_NUM    2536  // round(9.906 * 256), Q8 fixed-point
#define K_SHIFT  8

// sample type and detection flag
typedef ap_int<16> sample_t;     // 16-bit signed from ADC
typedef ap_uint<1>  detect_t;     // detection output

// top-level function for HLS
// the IP consumes I/Q streams from the decimator and produces
// filtered I/Q samples plus a detection flag.  only samples with
// det==1 are written to the output streams - this reduces the
// data sent to the host.
void ca_cfar(
    hls::stream<sample_t> &in_i,
    hls::stream<sample_t> &in_q,
    hls::stream<sample_t> &out_i,
    hls::stream<sample_t> &out_q,
    hls::stream<detect_t> &out_flag);

#endif // CA_CFAR_H_
