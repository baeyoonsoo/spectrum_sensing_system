% Data rate comparison for paper (required bandwidth per SDR vs. USB 2.0 limits)
clear; clc; close all;

%% 1. Parameters
categories = {'Conventional', 'Proposed'};
data_rate_per_SDR = [80, 80 * (1 - 0.88)]; % 80 MB/s vs 9.6 MB/s (88% reduction at 10% duty cycle)

%% 2. Create figure
figure('Color', 'w', 'Position', [100, 100, 600, 500]);
b = bar(1:2, data_rate_per_SDR, 0.5);

% Individual bar colors
b.FaceColor = 'flat';
b.CData(1,:) = [0.2 0.4 0.7]; % blue (Conventional)
b.CData(2,:) = [0.8 0.3 0.3]; % red  (Proposed)

%% 3. Axes and labels
set(gca, 'XTick', 1:2, 'XTickLabel', categories);
ylabel('Required Data Rate per SDR (MB/s)', 'FontName', 'Times New Roman');

% Y-axis range
ylim([0 100]);
grid on;

%% 4. USB 2.0 bottleneck limit lines
limit_val = 40;
yline(limit_val, 'k-.', 'USB 2.0 Practical Limit (~40 MB/s)', ...
    'LineWidth', 1, 'LabelHorizontalAlignment', 'left', ...
    'LabelVerticalAlignment', 'bottom', 'FontSize', 12,  'FontName', 'Times New Roman');

limit_theo = 60;
yline(limit_theo, 'k:', 'USB 2.0 Theoretical Limit (60 MB/s)', ...
    'LineWidth', 1, 'LabelHorizontalAlignment', 'left', ...
    'LabelVerticalAlignment', 'bottom', 'FontSize', 12, 'FontName', 'Times New Roman');

%% 5. Text annotations
% Conventional: overflow warning above bar
text(1, data_rate_per_SDR(1) + 5, 'Overflow (Data Loss)', ...
    'HorizontalAlignment', 'center', 'FontSize', 11, 'Color', 'r', 'FontName', 'Times New Roman');

% Proposed: numerical value above bar
text(2, data_rate_per_SDR(2) + 3, sprintf('%.1f MB/s', data_rate_per_SDR(2)), ...
    'HorizontalAlignment', 'center', 'FontSize', 11, 'Color', 'k', 'FontName', 'Times New Roman');

%% 6. Final formatting
set(gca, 'XTick', 1:2, 'XTickLabel', categories, 'FontSize', 12, 'FontName', 'Times New Roman');
xlabel('Processing Method', 'FontSize', 12, 'FontName', 'Times New Roman');
box on;
hold off;
