#include "ca_cfar.h"
#include <hls_stream.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <cmath>

// ── Testbench parameters ──────────────────────────────────────────────────────
// Noise model (must match CACFAR_ROC_Curve.m)
static const double SIGMA_ADC       = 5.0;   // noise std per I or Q channel [ADC counts]
static const double NOISE_FLOOR_dBm = -96.0; // measured noise floor (ADALM-Pluto, 5.8 GHz C-band)

// Signal model: Swerling 0 (non-fluctuating target)
//   I_CUT = amplitude + N(0, SIGMA^2)
//   Q_CUT = N(0, SIGMA^2)
// This is identical to the model used in CACFAR_ROC_Curve.m.

// dBm calibration: P_dBm = NOISE_FLOOR + SNR_dB
//   SNR = amplitude^2 / (2 * SIGMA^2)   [power-based, I-channel only]
static double amplitude_to_dbm(double A)
{
    double snr_lin = (A * A) / (2.0 * SIGMA_ADC * SIGMA_ADC);
    return NOISE_FLOOR_dBm + 10.0 * log10(snr_lin + 1e-15);
}

// ── Experiment parameters ─────────────────────────────────────────────────────
// Amplitude sweep: 5 to 50 counts
//   A =  5 -> SNR ≈  3 dB -> P_signal ≈ -93 dBm  (Pd ≈ 0%)
//   A = 20 -> SNR ≈  9 dB -> P_signal ≈ -87 dBm  (transition begins)
//   A = 32 -> SNR ≈ 13 dB -> P_signal ≈ -83 dBm  (Pd > 95%, paper target)
//   A = 50 -> SNR ≈ 17 dB -> P_signal ≈ -79 dBm  (Pd ≈ 100%)
static const int AMP_MIN      = 5;
static const int AMP_MAX      = 50;

// Pulse train geometry (10% duty cycle, matching hardware experiment)
static const int TOTAL_SAMPLES = 100000; // samples per amplitude level
static const int BLOCK_SIZE    = 100;    // samples per PRI block
static const int HIGH_LEN      = 10;     // pulse-on samples per block (10% duty cycle)
static const int TOTAL_PULSES  = (TOTAL_SAMPLES / BLOCK_SIZE) * HIGH_LEN; // 10 000 per level

int main()
{
    hls::stream<sample_t> in_i, in_q, out_i, out_q;
    hls::stream<detect_t> out_flag;

    std::mt19937 gen(42); // fixed seed — reproducible
    std::normal_distribution<double> noise_dist(0.0, SIGMA_ADC);

    // ── Helper: flush sliding window with noise to remove inter-level contamination
    auto flush_window = [&]()
    {
        for (int i = 0; i < WINDOW_SIZE + 10; ++i) {
            in_i.write((sample_t)(int)std::round(noise_dist(gen)));
            in_q.write((sample_t)(int)std::round(noise_dist(gen)));
            ca_cfar(in_i, in_q, out_i, out_q, out_flag);
            if (!out_flag.empty()) {
                out_flag.read(); out_i.read(); out_q.read();
            }
        }
    };

    // ─────────────────────────────────────────────────────────────────────────
    // Experiment 1: Noise-only false alarm probability
    // ─────────────────────────────────────────────────────────────────────────
    std::cout << "=== Experiment 1: Noise-only P_fa measurement ===\n";
    const int NOISE_SAMPLES = 100000;

    flush_window();

    int fa_count = 0;
    for (int i = 0; i < NOISE_SAMPLES; ++i) {
        in_i.write((sample_t)(int)std::round(noise_dist(gen)));
        in_q.write((sample_t)(int)std::round(noise_dist(gen)));
        ca_cfar(in_i, in_q, out_i, out_q, out_flag);
        if (!out_flag.empty()) {
            detect_t d = out_flag.read();
            if (d) ++fa_count;
            out_i.read(); out_q.read();
        }
    }
    double p_fa = (double)fa_count / NOISE_SAMPLES;
    std::cout << "False alarms: " << fa_count << " / " << NOISE_SAMPLES
              << "  ->  P_fa = " << p_fa << "\n";
    std::cout << "Note: P_fa should be close to 10^-4 for k=9.906 (K_NUM=2536)\n\n";

    // ─────────────────────────────────────────────────────────────────────────
    // Experiment 2: Pd vs. Input Power Level  (Figure 11)
    // ─────────────────────────────────────────────────────────────────────────
    std::cout << "=== Experiment 2: Pd vs. Input Power Level (Figure 11) ===\n";
    std::cout << "Signal model : Swerling 0  (I = A + N(0,sigma^2),  Q = N(0,sigma^2))\n";
    std::cout << "Calibration  : sigma=" << SIGMA_ADC
              << " counts,  noise floor=" << NOISE_FLOOR_dBm << " dBm\n";
    std::cout << "Trials/level : " << TOTAL_PULSES << " pulses  (" << TOTAL_SAMPLES
              << " samples at 10%% duty cycle)\n\n";

    std::cout << "Level  Power(dBm)  Detected   Total     Pd\n";
    std::cout << std::string(55, '-') << "\n";
    std::cout << std::fixed;
    std::cout.precision(4);

    struct Result { int amp; double dbm; int detected; double pd; };
    std::vector<Result> results;

    for (int amp = AMP_MIN; amp <= AMP_MAX; ++amp) {
        double dbm_val = amplitude_to_dbm((double)amp);

        // Flush window to eliminate contamination from the previous level
        flush_window();

        // Build sample arrays: noise baseline, then overwrite pulse windows
        // with Swerling 0 signal  (I = amp + noise,  Q = noise)
        std::vector<sample_t> i_samps(TOTAL_SAMPLES), q_samps(TOTAL_SAMPLES);
        for (int i = 0; i < TOTAL_SAMPLES; ++i) {
            i_samps[i] = (sample_t)(int)std::round(noise_dist(gen));
            q_samps[i] = (sample_t)(int)std::round(noise_dist(gen));
        }
        for (int i = 0; i < TOTAL_SAMPLES; i += BLOCK_SIZE) {
            for (int j = 0; j < HIGH_LEN && (i + j) < TOTAL_SAMPLES; ++j) {
                i_samps[i + j] = (sample_t)(int)std::round(amp + noise_dist(gen));
                q_samps[i + j] = (sample_t)(int)std::round(noise_dist(gen));
            }
        }

        // Run CA-CFAR and count detections
        // Note: det at output t reflects the CUT from (TRAIN_CELLS+GUARD_CELLS)=44 cycles ago.
        // Because the pulse samples in each block are exactly the first HIGH_LEN slots,
        // their detection results appear at output indices block_start+44 to block_start+53.
        // The loop counts every det=1 event.  False-alarm contribution is negligible
        // (~P_fa * (TOTAL_SAMPLES - TOTAL_PULSES) ≈ 1e-4 * 90000 ≈ 9 events).
        int detected = 0;
        for (int i = 0; i < TOTAL_SAMPLES; ++i) {
            in_i.write(i_samps[i]);
            in_q.write(q_samps[i]);
            ca_cfar(in_i, in_q, out_i, out_q, out_flag);
            if (!out_flag.empty()) {
                detect_t d = out_flag.read();
                if (d) ++detected;
                out_i.read(); out_q.read();
            }
        }

        // Use TOTAL_PULSES as denominator (detection window only).
        // Subtract expected false alarms (p_fa * noise_samples) for clean Pd estimate.
        int noise_samples_count = TOTAL_SAMPLES - TOTAL_PULSES;
        double fa_in_noise = p_fa * noise_samples_count;
        double pd = ((double)detected - fa_in_noise) / TOTAL_PULSES;
        if (pd < 0.0) pd = 0.0;
        if (pd > 1.0) pd = 1.0;

        results.push_back({amp, dbm_val, detected, pd});
        std::cout << "  " << amp << "      "
                  << dbm_val << "    "
                  << detected << " / " << TOTAL_PULSES << "  "
                  << pd << "\n";
    }

    // ─────────────────────────────────────────────────────────────────────────
    // CSV output for MATLAB plotting
    // ─────────────────────────────────────────────────────────────────────────
    std::cout << "\n=== Figure 11 Data (CSV) ===\n";
    std::cout << "amplitude_counts,power_dBm,pd\n";
    for (const auto& r : results)
        std::cout << r.amp << "," << r.dbm << "," << r.pd << "\n";

    // Also write CSV file
    std::ofstream csv("pd_vs_power_fig11.csv");
    csv << "amplitude_counts,power_dBm,pd\n";
    csv << std::fixed;
    csv.precision(4);
    for (const auto& r : results)
        csv << r.amp << "," << r.dbm << "," << r.pd << "\n";
    csv.close();
    std::cout << "\n(CSV also saved to: pd_vs_power_fig11.csv)\n";

    return 0;
}
