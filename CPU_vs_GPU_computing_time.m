clear; clc;

%% 0. Compile gpu_pipeline_pinned MEX (CUDA)
script_dir = fileparts(mfilename('fullpath'));
cu_file    = fullfile(script_dir, 'gpu_pipeline_pinned.cu');
mex_out    = fullfile(script_dir, 'gpu_pipeline_pinned');

if ~exist(mex_out, 'file') && ~exist([mex_out '.mexw64'], 'file')
    fprintf('Compiling gpu_pipeline_pinned MEX...\n');
    mexcuda('-R2018a', cu_file, '-lcufft', '-output', mex_out);
end

addpath(script_dir);

%% 1. Experiment configuration
fft_size  = 1024;
M_blocks  = [1000, 5000, 10000, 50000, 100000];
num_cases = length(M_blocks);
num_trials = 5;   % 5 repeated measurements -> use median to remove outliers
threshold  = single(3.0);

cpu_latency = zeros(num_cases, 1);
gpu_latency = zeros(num_cases, 1);

fprintf('CPU vs. GPU (Pinned+Async) latency benchmark\n\n');

%% 2. GPU warmup (cuFFT kernel compile + memory pre-allocation)
disp('Warming up GPU...');
dummy = complex(randn(fft_size, 1000, 'single'), randn(fft_size, 1000, 'single'));
gpu_pipeline_pinned(dummy, threshold);   % initialize cuFFT plan inside MEX
gpu_pipeline_pinned(dummy, threshold);   % second call to fully warm up
disp('Warmup done. Starting benchmark.');
fprintf('%s\n', repmat('-', 1, 50));

%% 3. Benchmark (repeated measurements)
for i = 1:num_cases
    M = M_blocks(i);

    % Generate complex single input data (fft_size x M)
    data       = complex(randn(fft_size, M, 'single'), ...
                         randn(fft_size, M, 'single'));
    win_matrix = repmat(window(@hamming, fft_size), 1, M);

    cpu_times = zeros(num_trials, 1);
    gpu_times = zeros(num_trials, 1);

    for t = 1:num_trials

        % -------------------------------------------------
        % [CPU] Windowing -> FFT -> movmean -> Detection
        % -------------------------------------------------
        tic;
        cpu_windowed    = data .* win_matrix;
        cpu_fft         = fft(cpu_windowed);
        cpu_cfar        = movmean(abs(cpu_fft), 16, 1);
        cpu_detections  = cpu_cfar > threshold;
        cpu_target_count = sum(cpu_detections(:));   %#ok<NASGU>
        cpu_times(t) = toc;

        % -------------------------------------------------
        % [GPU] Pinned + Async (MEX CUDA)
        %   H2D transfer -> Windowing -> cuFFT -> Magnitude
        %   -> Moving Average -> Detection -> D2H
        %   Single StreamSynchronize for all ops
        % -------------------------------------------------
        [gpu_count, gpu_times(t)] = gpu_pipeline_pinned(data, threshold); %#ok<ASGLU>

    end

    % Use median to suppress OS scheduling outliers
    cpu_latency(i) = median(cpu_times);
    gpu_latency(i) = median(gpu_times);

    fprintf('[M = %-6d PDUs]  CPU: %8.4f s  |  GPU (Pinned+Async): %8.4f s  |  Speedup: %.1fx\n', ...
        M, cpu_latency(i), gpu_latency(i), cpu_latency(i)/gpu_latency(i));
end

%% 4. Results summary table
fprintf('\n%-10s %10s %10s %8s\n', 'M (PDUs)', 'CPU (s)', 'GPU (s)', 'Speedup');
for i = 1:num_cases
    fprintf('%-10d %10.4f %10.4f %7.1fx\n', ...
        M_blocks(i), cpu_latency(i), gpu_latency(i), ...
        cpu_latency(i)/gpu_latency(i));
end

%% 5. Plot results (Figure 13)
ax1 = gca; hold on; grid on; box on;
% --- Processing latency comparison ---
figure(1);
set(ax1,'FontSize',12,'FontName', 'Times New Roman');
plot(M_blocks, cpu_latency, '-ob', 'LineWidth', 2, ...
     'MarkerSize', 8, 'MarkerFaceColor', 'b');
hold on; grid on;
plot(M_blocks, gpu_latency, '-sr', 'LineWidth', 2, ...
     'MarkerSize', 8, 'MarkerFaceColor', 'r');
xlabel('Data Size (Number of 1024-point PDUs)', 'FontSize', 11, 'FontName', 'Times New Roman');
ylabel('Processing Latency [sec]', 'FontSize', 11, 'FontName', 'Times New Roman');
legend('CPU', 'GPU (Pinned+Async)', 'Location', 'northwest', 'FontSize', 11, 'FontName', 'Times New Roman');
hold off;

% --- Speedup ratio ---
figure(2);
speedup = cpu_latency ./ gpu_latency;
bar(M_blocks / 1000, speedup, 'FaceColor', [0.2 0.6 0.3]);
xlabel('Data Size (x 10^3 PDUs)', 'FontSize', 11, 'FontName', 'Times New Roman');
ylabel('Speedup (CPU / GPU)', 'FontSize', 11, 'FontName', 'Times New Roman');
title('GPU Speedup over CPU', 'FontSize', 12, 'FontName', 'Times New Roman');
grid on;
for k = 1:num_cases
    text(M_blocks(k)/1000, speedup(k) + 0.1, sprintf('%.1fx', speedup(k)), ...
         'HorizontalAlignment', 'center', 'FontSize', 9);
end
