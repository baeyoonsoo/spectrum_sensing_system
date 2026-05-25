# Data Availability — Edge CA-CFAR Wideband Spectrum Sensing

This repository contains the source code supporting the results reported in:

> **"Edge CA-CFAR Data Reduction for Bandwidth-Efficient Real-Time Wideband Spectrum Sensing on Low-Cost SDRs"**
> Yunsu Bae, Hajung Lee, Hyojun Park, Won-ho Jang, Byung-Jun Jang
> *Sensors*, MDPI, 2026.

---

## File Index

| File | Reproduces | Description |
|------|-----------|-------------|
| `ca_cfar.h` | — | Vitis HLS header: FPGA CA-CFAR IP parameters and interface |
| `ca_cfar.cpp` | — | Vitis HLS implementation: CA-CFAR IP core (I²+Q² power metric, pipeline II=1) |
| `ca_cfar_tb.cpp` | Fig. 11 | Vitis HLS C-simulation testbench: Pfa and Pd vs. input power sweep |
| `pd_vs_power_fig11.csv` | Fig. 11 | C-simulation output: Pd vs. input power (amplitude 5–50 counts, 10,000 pulses/level) |
| `plot_fig11_pd_vs_power.m` | Fig. 11 | MATLAB: plots Pd vs. input power from `pd_vs_power_fig11.csv` |
| `CACFAR_ROC_Curve.m` | Fig. 12 | MATLAB: fixed-point Monte Carlo ROC simulation (2×10⁶ trials, seed=43) |
| `data_throughput_comp.m` | Fig. 10 | MATLAB: bar chart comparing data rate (conventional vs. proposed) against USB 2.0 limits |
| `gpu_pipeline_pinned.cu` | Fig. 13 | CUDA MEX: GPU processing pipeline (pinned memory + async stream: Window→FFT→Magnitude→CFAR→Detection) |
| `CPU_vs_GPU_computing_time.m` | Fig. 13 | MATLAB: CPU vs. GPU latency benchmark across PDU counts (random input, no fixed seed; GPU speedup consistently 4–6× at ≥50,000 PDUs — see note below) |
| `duration_check.py` | Table 3 | Python: end-to-end latency measurement — **requires physical ADALM-Pluto SDR** (see note below). Measures GPU signal processing and rendering as a combined stage; Table 3 individual breakdown obtained via CUDA timing events in the full display-connected system. |

---

## Software Requirements

### FPGA CA-CFAR IP (`ca_cfar.h`, `ca_cfar.cpp`, `ca_cfar_tb.cpp`)
- **Vitis HLS** 2022.1 or later (AMD/Xilinx)
- `ap_int.h`, `hls_stream.h` from Vitis HLS include path

### MATLAB scripts (`*.m`)
- **MATLAB** R2021b or later
- **Parallel Computing Toolbox** (required for `mexcuda` and GPU functions)
- **CUDA Toolkit** 11.x or later (matching your GPU driver)
- GPU with CUDA compute capability 6.0+ (tested on NVIDIA GeForce RTX 3050)

### Python script (`duration_check.py`)
- Python 3.9+
- `numpy`, `cupy` (CUDA-compatible build)
- `pyadi-iio` (`pip install pyadi-iio`) for ADALM-Pluto control
- `libiio` installed on the host system

---

## How to Run

### FPGA CA-CFAR C-Simulation (Fig. 11 data)
1. Open Vitis HLS and create a new project.
2. Add `ca_cfar.h` and `ca_cfar.cpp` as design sources, `ca_cfar_tb.cpp` as testbench.
3. Set top-level function to `ca_cfar`.
4. Run **C Simulation**. The testbench prints Pfa and Pd vs. input power to the console and writes `pd_vs_power_fig11.csv`.

### Pd vs. Input Power Plot (Fig. 11)
```matlab
run('plot_fig11_pd_vs_power.m')
```
Reads `pd_vs_power_fig11.csv` directly. No Vitis HLS run required if the CSV is already present.

### ROC Curves (Fig. 12)
```matlab
run('CACFAR_ROC_Curve.m')
```
Fully self-contained Monte Carlo simulation with fixed seed (`rng(43)`). Runtime: ~3–5 minutes on a modern CPU.

### Data Rate Comparison (Fig. 10)
```matlab
run('data_throughput_comp.m')
```

### CPU vs. GPU Latency Benchmark (Fig. 13)
```matlab
run('CPU_vs_GPU_computing_time.m')
```
On first run, the script compiles `gpu_pipeline_pinned.cu` via `mexcuda`. Subsequent runs skip compilation if the MEX file is already present.

> **Note:** The benchmark generates random input data on every run (no fixed seed). Absolute timing values therefore vary across runs due to data randomness and OS scheduling. The GPU speedup over CPU nonetheless consistently falls in the **4–6× range** at large PDU counts (≥50,000 PDUs), as reported in Figure 13.

### End-to-End Latency Measurement (Table 3) — Hardware Required
```bash
python duration_check.py
```
> **Note:** This script requires a physical ADALM-Pluto SDR connected via USB at address `usb:1.7.5`. It cannot be executed without hardware. The measured values reported in Table 3 of the paper are:
> - Analog buffer filling: 0.017 ms
> - USB packaging & transfer: 0.125 ms
> - GPU signal processing: 0.112 ms
> - Rendering: 0.116 ms
> - **Total end-to-end latency: 0.370 ms**
>
> **Note on measurement scope:** `duration_check.py` measures GPU signal processing and rendering as a **combined** stage and approximates the rendering step with a GPU buffer copy (`cp.copyto`) for hardware-independent execution. The individual Table 3 values (GPU signal processing: 0.112 ms, rendering: 0.116 ms) were profiled separately using CUDA timing events in the full display-connected system and are not directly reproducible from this script alone.

---

## Hardware Parameters (CA-CFAR IP)

| Parameter | Value |
|-----------|-------|
| Training cells (per side) | 32 |
| Guard cells (per side) | 12 |
| Total reference cells (N) | 64 |
| Power metric | I² + Q² (4 DSP blocks) |
| Average | sum >> 6 (exact division by 64) |
| Threshold factor K | 9.9063 (K_NUM = 2536, Q8) |
| Target Pfa | 10⁻⁴ |
| Sample width | 16-bit signed I/Q |
| Pipeline | II = 1 |
| Clock | 10 ns (100 MHz) |
| Latency | 10 cycles |
| FF / LUT / BRAM / DSP | 4068 / 2861 / 0 / 4 |

The threshold factor K is derived from the CA-CFAR false alarm formula:

**Pfa = (1 + K/N)^(−N)** → K = N × (Pfa^(−1/N) − 1) = 64 × (10000^(1/64) − 1) ≈ 9.906

---

## Notes

- Raw RF measurement data (IQ captures from 5 ADALM-Pluto SDRs) are not included as the hardware noise environment cannot be exactly reproduced. These data are available from the corresponding author upon reasonable request.

---

## License

MIT License. See `LICENSE.txt` for full license text.

## Contact

Byung-Jun Jang — bjjang@kookmin.ac.kr
