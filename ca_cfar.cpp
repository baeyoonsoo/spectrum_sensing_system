#include "ca_cfar.h"

// CA‑CFAR implementation for Vitis HLS with I/Q processing
//  - sliding window stored in registers/LUTs (no BRAM)
//  - parameters (guard cells, training cells, k factor) defined in header
//  - input: two AXI‑stream channels (I and Q from rx_fir_decimator)
//  - output: filtered I/Q samples plus a detection flag stream
//  - only detected samples are forwarded to reduce host bandwidth
//  - pipeline II=1

void ca_cfar(
    hls::stream<sample_t> &in_i,
    hls::stream<sample_t> &in_q,
    hls::stream<sample_t> &out_i,
    hls::stream<sample_t> &out_q,
    hls::stream<detect_t> &out_flag) {
#pragma HLS INTERFACE axis port=in_i
#pragma HLS INTERFACE axis port=in_q
#pragma HLS INTERFACE axis port=out_i
#pragma HLS INTERFACE axis port=out_q
#pragma HLS INTERFACE axis port=out_flag
#pragma HLS PIPELINE II=1

    // sliding window buffer for power values (I² + Q²)
    static ap_uint<32> window[WINDOW_SIZE];
#pragma HLS ARRAY_PARTITION variable=window complete
#pragma HLS RESET variable=window

    // read new complex sample
    sample_t i_sample = in_i.read();
    sample_t q_sample = in_q.read();

    // compute power metric: I² + Q²  (4 DSP blocks as confirmed by synthesis)
    ap_uint<32> mag;
    {
        ap_int<32> isq = (ap_int<32>)i_sample * (ap_int<32>)i_sample;
        ap_int<32> qsq = (ap_int<32>)q_sample * (ap_int<32>)q_sample;
        mag = (ap_uint<32>)(isq + qsq);
    }

    // shift register: move every element one step toward index 0
    for (int idx = 0; idx < WINDOW_SIZE-1; idx++) {
#pragma HLS UNROLL
        window[idx] = window[idx+1];
    }
    window[WINDOW_SIZE-1] = mag;

    // accumulate training cells on both sides (sum of 64 × 32-bit power values)
    ap_uint<64> sum = 0;
    for (int idx = 0; idx < TRAIN_CELLS; idx++) {
#pragma HLS UNROLL
        sum += window[idx];
        sum += window[WINDOW_SIZE - 1 - idx];
    }

    // average over total training cells (2*TRAIN_CELLS=64) by shifting 6 bits
    ap_uint<64> avg = sum >> 6;
    // compute threshold = avg * k  (Q8 fixed-point)
    ap_uint<64> thresh = (avg * (ap_uint<64>)K_NUM) >> K_SHIFT;

    // center cell after guards
    ap_uint<32> center = window[TRAIN_CELLS + GUARD_CELLS];
    detect_t det = ((ap_uint<64>)center > thresh) ? (detect_t)1 : (detect_t)0;

    // always emit a sample to preserve timing; pad noise positions with zero
    sample_t oi = det ? i_sample : (sample_t)0;
    sample_t oq = det ? q_sample : (sample_t)0;
    out_i.write(oi);
    out_q.write(oq);
    out_flag.write(det);
}
