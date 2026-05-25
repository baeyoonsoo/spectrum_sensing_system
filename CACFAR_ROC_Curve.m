% CACFAR_ROC_Curve.m
% Fixed-point CA-CFAR ROC simulation matching ca_cfar.h/ca_cfar.cpp.
% N=64 training cells, K=9.9063 (K_NUM=2536, Q8), metric: I^2+Q^2.
clear; clc; close all;

%% ── 1. Hardware Parameters ───────────────────────────────────────────────
TRAIN_CELLS = 32;                         % reference cells per side
GUARD_CELLS = 12;                         % guard cells per side
N_TOTAL     = 2 * TRAIN_CELLS;           % = 64 total reference cells
DIV_SHIFT   = int32(log2(N_TOTAL));      % = 6  (exact: 64 = 2^6)
K_NUM       = int64(2536);               % Q8 numerator: (1+K/64)^(-64)=1e-4 => K=9.906
K_SHIFT     = 8;                          % Q8 denominator shift
K_fp        = double(K_NUM) / 2^K_SHIFT; % = 9.9063

%% ── 2. Noise Parameters ──────────────────────────────────────────────────
SIGMA = 5;                               % noise std per I or Q [ADC counts]

cnr_dB2A = @(snr_dB) SIGMA * sqrt(2 * 10.^(snr_dB/10));

fprintf('N_total=%d, K=%.4f (K_NUM=%d/256), SIGMA=%d counts\n\n', ...
    N_TOTAL, K_fp, K_NUM, SIGMA);

%% ── 3. Monte Carlo Settings ──────────────────────────────────────────────
M          = 2e6;                         % trials
rng(43);                                  % fixed seed

SNR_dB_roc = [7, 9, 11, 13];            % SNR values for ROC family
SNR_dB_det = 0 : 0.2 : 25;              % fine sweep for Pd-vs-SNR

% K sweep to trace full ROC (empirical Pfa axis)
K_vec      = logspace(-0.2, 2.5, 500);
K_NUM_vec  = int64(round(K_vec * 2^K_SHIFT));  % quantised to Q8

%% ── 4. Reference Cells (noise only, shared across all SNR) ──────────────
nR_ref  = int16(round(randn(M, N_TOTAL) * SIGMA));
nI_ref  = int16(round(randn(M, N_TOTAL) * SIGMA));
mag_ref = int64(nR_ref).^2 + int64(nI_ref).^2;  % power metric

S_ref   = int64(sum(mag_ref, 2));        % sum of 64 power values
avg_int = bitshift(S_ref, -DIV_SHIFT);  % >>6 = divide by 64

%% ── 5. CUT under H0 (noise only) ────────────────────────────────────────
nR_H0  = int16(round(randn(M,1) * SIGMA));
nI_H0  = int16(round(randn(M,1) * SIGMA));
mag_H0 = int64(nR_H0).^2 + int64(nI_H0).^2;

%% ── 6. Empirical Pfa Sweep ───────────────────────────────────────────────
Pfa_emp = zeros(1, numel(K_vec));
for ki = 1:numel(K_vec)
    T = bitshift(K_NUM_vec(ki) .* avg_int, -K_SHIFT);
    Pfa_emp(ki) = mean(mag_H0 > T);
end

% Pfa at the design threshold K_fp
[~, ki_des]  = min(abs(K_vec - K_fp));
Pfa_at_Kfp   = Pfa_emp(ki_des);

% Empirically find K giving exactly Pfa = 1e-4 (validation check)
[~, ki_emp]  = min(abs(Pfa_emp - 1e-4));
K_emp        = K_vec(ki_emp);
K_NUM_emp    = K_NUM_vec(ki_emp);
Pfa_emp_val  = Pfa_emp(ki_emp);

fprintf('Pfa check (N=%d):\n', N_TOTAL);
fprintf('  K_fp  = %.4f (K_NUM=%d) -> Pfa = %.2e  [hardware]\n', K_fp, double(K_NUM), Pfa_at_Kfp);
fprintf('  theory (1+K/64)^-64    -> Pfa = %.2e\n', (1 + K_fp/N_TOTAL)^(-N_TOTAL));
fprintf('  K_emp = %.4f (K_NUM=%d) -> Pfa = %.2e  [MC 1e-4 target]\n\n', ...
        K_emp, double(K_NUM_emp), Pfa_emp_val);

%% ── 7. ROC: Pd vs Pfa for each SNR ──────────────────────────────────────
Pd_roc = zeros(numel(SNR_dB_roc), numel(K_vec));

fprintf('ROC sweep:\n');
for si = 1:numel(SNR_dB_roc)
    A_adc  = cnr_dB2A(SNR_dB_roc(si));

    % Swerling 0: non-fluctuating amplitude on I channel, noise on both
    nR_H1  = int16(round(A_adc + randn(M,1)*SIGMA));
    nI_H1  = int16(round(randn(M,1)*SIGMA));
    mag_H1 = int64(nR_H1).^2 + int64(nI_H1).^2;

    for ki = 1:numel(K_vec)
        T = bitshift(K_NUM_vec(ki) .* avg_int, -K_SHIFT);
        Pd_roc(si,ki) = mean(mag_H1 > T);
    end

    fprintf('  SNR = %2d dB | A = %5.1f counts | Pd @ K_fp = %.4f\n', ...
        SNR_dB_roc(si), A_adc, Pd_roc(si, ki_des));
end

%% ── 8. Pd vs SNR at design threshold ────────────────────────────────────
fprintf('\nPd vs SNR sweep:\n');
Pd_det     = zeros(1, numel(SNR_dB_det));
Pd_det_emp = zeros(1, numel(SNR_dB_det));

T_des = bitshift(K_NUM     .* avg_int, -K_SHIFT);  % design threshold (K_fp)
T_emp = bitshift(K_NUM_emp .* avg_int, -K_SHIFT);  % empirical threshold (K_emp)

for si = 1:numel(SNR_dB_det)
    A_adc  = cnr_dB2A(SNR_dB_det(si));
    nR_H1  = int16(round(A_adc + randn(M,1)*SIGMA));
    nI_H1  = int16(round(randn(M,1)*SIGMA));
    mag_H1 = int64(nR_H1).^2 + int64(nI_H1).^2;
    Pd_det(si)     = mean(mag_H1 > T_des);
    Pd_det_emp(si) = mean(mag_H1 > T_emp);
end

[~, i50]    = min(abs(Pd_det     - 0.50));
[~, i95]    = min(abs(Pd_det     - 0.95));
[~, i50_emp] = min(abs(Pd_det_emp - 0.50));
[~, i95_emp] = min(abs(Pd_det_emp - 0.95));
fprintf('  [K_fp =%.4f, Pfa=%.2e]  Pd=50%% @ SNR=%.1f dB\n', K_fp,  Pfa_at_Kfp,  SNR_dB_det(i50));
fprintf('  [K_fp =%.4f, Pfa=%.2e]  Pd=95%% @ SNR=%.1f dB\n', K_fp,  Pfa_at_Kfp,  SNR_dB_det(i95));
fprintf('  [K_emp=%.4f, Pfa=%.2e]  Pd=50%% @ SNR=%.1f dB\n', K_emp, Pfa_emp_val, SNR_dB_det(i50_emp));
fprintf('  [K_emp=%.4f, Pfa=%.2e]  Pd=95%% @ SNR=%.1f dB\n', K_emp, Pfa_emp_val, SNR_dB_det(i95_emp));
fprintf('  (Paper: Pd>95%% at SNR=13 dB)\n\n');

%% ── 9. Figure: ROC Curves ────────────────────────────────────────────────
[Pfa_plt, sort_idx] = sort(Pfa_emp, 'ascend');

colors = lines(numel(SNR_dB_roc));
figure('Units','inches','Position',[1 2 7 5]);
ax1 = gca; hold on; grid on; box on;

for si = 1:numel(SNR_dB_roc)
    plot(ax1, Pfa_plt, Pd_roc(si, sort_idx), '-', ...
        'Color', colors(si,:), 'LineWidth', 1.8, ...
        'DisplayName', sprintf('SNR = %d dB', SNR_dB_roc(si)));
end

% Reference lines
xline(ax1, Pfa_at_Kfp, ':', 'Color',[0.3 0.3 0.3], 'LineWidth',1.4, ...
    'HandleVisibility','off');                          % K_fp design point
xline(ax1, Pfa_emp_val, '--', 'Color',[0 0.55 0], 'LineWidth',1.4, ...
    'HandleVisibility','off');                          % K_emp exact 1e-4
yline(ax1, 0.95, ':r', 'LineWidth',1.2, 'HandleVisibility','off');

% Operating-point marker at SNR = 13 dB (K_emp threshold)
si13 = find(SNR_dB_roc == 13);
plot(ax1, Pfa_emp_val, Pd_roc(si13, ki_emp), 'g^', ...
    'MarkerSize',10, 'MarkerFaceColor','g', 'HandleVisibility','off');

% Annotations
text(ax1, Pfa_at_Kfp*1.8, 0.37, ...
    sprintf('K_{fp}=%.2f', K_fp), ...
    'FontSize',10, 'Rotation',90, 'VerticalAlignment','bottom','Color',[0.3 0.3 0.3], 'FontName', 'Times New Roman');
text(ax1, Pfa_emp_val*2.3, 0.03, ...
    sprintf('K_{emp}=%.2f(P_{fa}=10^{-4})', K_emp), ...
    'FontSize',10, 'Rotation',90, 'VerticalAlignment','bottom', 'Color',[0 0.5 0], 'FontName', 'Times New Roman');
text(ax1, Pfa_emp_val*1.3, Pd_roc(si13, ki_emp)-0.06, ...
    sprintf('SNR \\approx 13 dB\nP_d=0.9626'), ...
    'FontSize',10, 'Color',[0 0.5 0], 'FontName', 'Times New Roman');
text(ax1, 1.2e-6, 0.975, 'P_d = 0.95', ...
    'FontSize',10, 'Color',[0.8 0 0], 'FontName', 'Times New Roman');

set(ax1,'XScale','log','FontSize',12,'FontName', 'Times New Roman');
xlim(ax1,[1e-6 1]); ylim(ax1,[0 1]);
xlabel(ax1,'Probability of False Alarm,  P_{fa}', ...
    'FontSize',13, 'FontName', 'Times New Roman');
ylabel(ax1,'Probability of Detection,  P_d', ...
    'FontSize',13, 'FontName', 'Times New Roman');
legend(ax1,'Location','southeast','FontSize',10,'NumColumns',1, 'FontName', 'Times New Roman');

%% ── 10. Summary ──────────────────────────────────────────────────────────
fprintf('K_fp=%.4f (Pfa=%.2e): Pd50=%.1fdB, Pd95=%.1fdB\n', ...
        K_fp, Pfa_at_Kfp, SNR_dB_det(i50), SNR_dB_det(i95));
fprintf('K_emp=%.4f (Pfa=%.2e): Pd50=%.1fdB, Pd95=%.1fdB  (paper: 13dB)\n\n', ...
        K_emp, Pfa_emp_val, SNR_dB_det(i50_emp), SNR_dB_det(i95_emp));

fprintf('%-8s %-8s %-10s %-10s\n', 'SNR(dB)', 'A(cts)', 'Pd@K_fp', 'Pd@K_emp');
for si = 1:numel(SNR_dB_roc)
    fprintf('  %2d dB   %5.1f   %.4f     %.4f\n', ...
        SNR_dB_roc(si), cnr_dB2A(SNR_dB_roc(si)), ...
        Pd_roc(si,ki_des), Pd_roc(si,ki_emp));
end
