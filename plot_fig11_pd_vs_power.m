% plot_fig11_pd_vs_power.m
% Plots Pd vs. input power from pd_vs_power_fig11.csv (Vitis HLS C-sim output).
% Place the CSV in the same directory before running.
clear; clc; close all;

%% ── Load CSV ─────────────────────────────────────────────────────────────
T = readtable('pd_vs_power_fig11.csv');
power_dBm = T.power_dBm;
Pd        = T.pd;

%% ── Reference values ─────────────────────────────────────────────────────
P_TARGET = 0.9626;
P_REF    = -83.1625;   % dBm  (amplitude = 31 counts, first point Pd > 95%)

%% ── Plot ─────────────────────────────────────────────────────────────────
figure('Units', 'inches', 'Position', [1 2 7 5], 'Color', 'w');
ax = gca; hold on; grid on; box on;

plot(ax, power_dBm, Pd, 'b-o', ...
    'LineWidth', 2.0, 'MarkerSize', 4, 'MarkerFaceColor', 'b', ...
    'DisplayName', 'CA-CFAR (HLS C-simulation)');

% Pd = 0.959 horizontal reference
yline(ax, P_TARGET, ':', 'Color', [0.8 0 0], 'LineWidth', 1.4, ...
    'HandleVisibility', 'off');
text(ax, min(power_dBm) + 0.4, P_TARGET + 0.040, ...
    sprintf('P_d = %.4f', P_TARGET), ...
    'FontSize', 11, 'Color', [0.8 0 0], 'FontName', 'Times New Roman');

% -83.25 dBm vertical reference
xline(ax, P_REF, '--k', 'LineWidth', 1.4, 'HandleVisibility', 'off');
text(ax, P_REF + 0.25, 0.08, ...
    sprintf('%.4f dBm\n(SNR \\approx 13 dB)', P_REF), ...
    'FontSize', 11, 'FontName', 'Times New Roman', 'VerticalAlignment', 'bottom');

% Operating-point marker
[~, idx_op] = min(abs(power_dBm - P_REF));
plot(ax, power_dBm(idx_op), Pd(idx_op), '^', ...
    'Color', [0 0.6 0], 'MarkerSize', 10, 'MarkerFaceColor', [0 0.6 0], ...
    'HandleVisibility', 'off');

%% ── Axes formatting ──────────────────────────────────────────────────────
xlim(ax, [min(power_dBm) - 0.5,  max(power_dBm) + 0.5]);
ylim(ax, [0,  1.05]);
set(ax, 'FontSize', 13, 'FontName', 'Times New Roman');
xlabel(ax, 'Input Power Level (dBm)', ...
    'FontSize', 14, 'FontName', 'Times New Roman');
ylabel(ax, 'Probability of Detection, P_d', ...
    'FontSize', 14, 'FontName', 'Times New Roman');

%% ── Summary ──────────────────────────────────────────────────────────────
[~, i_target] = min(abs(Pd - P_TARGET));
fprintf('=== Summary ===\n');
fprintf('  Pd = %.3f at %.2f dBm  (SNR = %.2f dB)\n', ...
        P_TARGET, power_dBm(i_target), power_dBm(i_target) - (-96));
fprintf('  Operating point (%.2f dBm): Pd = %.4f\n', ...
        power_dBm(idx_op), Pd(idx_op));
