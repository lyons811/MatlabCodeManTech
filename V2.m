% MATLAB Script for DeepRadar2022 Analysis with Enhanced GPU Utilization
% This script demonstrates various signal processing techniques on the DeepRadar2022 dataset
% utilizing GPU processing where applicable.

% Enable GPU processing if available
useGPU = false;
if gpuDeviceCount > 0
    gpuDevice(1); % Initialize the first GPU device
    useGPU = true;
    fprintf('GPU detected and enabled for processing.\n');
else
    fprintf('No GPU detected. Processing will be performed on the CPU.\n');
end

% Load data
try
    load('X_test.mat');
    X_test = double(X_test); % Ensure double precision
    fprintf('Original X_test size: %s\n', mat2str(size(X_test)));
    
    if useGPU
        X_test_gpu = gpuArray(X_test); % Keep a GPU copy for GPU-accelerated operations
        fprintf('Data loaded onto the GPU.\n');
    else
        fprintf('Data loaded onto the CPU.\n');
    end
catch
    error('Unable to load X_test.mat. Please ensure the file is in the current directory.');
end

% Select a random signal for analysis
signalIndex = randi(size(X_test, 1));
if ndims(X_test) == 3
    signal = squeeze(X_test(signalIndex, :, :)); % [m, p]
else
    error('X_test must be a 3D array with dimensions [n, m, p].');
end

% Ensure signal has at least two columns for I and Q
if size(signal, 2) < 2
    error('Selected signal does not have at least two columns for I and Q components.');
end

I = signal(:, 1);
Q = signal(:, 2);
complexSignal = I + 1i*Q;

% Handle complexSignal based on GPU availability
if useGPU
    complexSignal_gpu = gpuArray(complexSignal); % Keep on GPU
    fprintf('Selected signal is on the GPU.\n');
else
    complexSignal_cpu = double(complexSignal); % Ensure double precision on CPU
    fprintf('Selected signal is on the CPU.\n');
end

% 2. Find abrupt changes in signal (CPU only)
if useGPU
    % Transfer complexSignal to CPU for functions that don't support GPU
    complexSignal_cpu = gather(complexSignal_gpu);
    fprintf('Transferred signal from GPU to CPU for change point detection.\n');
else
    complexSignal_cpu = complexSignal; % Already on CPU
end

changePoints = findchangepts(abs(complexSignal_cpu), 'MaxNumChanges', 5);
fprintf('Detected change points at indices: %s\n', mat2str(changePoints));

% 3. Time-frequency ridges using Spectrogram (GPU accelerated if possible)
if useGPU
    % Define window on CPU and transfer to GPU
    window_cpu = hamming(128);
    window_gpu = gpuArray(window_cpu);
    overlap = 64;
    nfft = 256;
    fs = 1024; % Sample rate

    % Compute spectrogram on GPU
    try
        [s_gpu, f, t] = spectrogram(complexSignal_gpu, window_gpu, overlap, nfft, fs, 'yaxis');
        fprintf('Spectrogram computed on GPU.\n');
        
        % Move spectrogram data to CPU for further processing
        s = gather(s_gpu);
        f = gather(f);
        t = gather(t);
    catch ME
        warning('Spectrogram computation on GPU failed: %s\nAttempting to compute on CPU.', ME.message);
        [s, f, t] = spectrogram(complexSignal_cpu, window_cpu, overlap, nfft, fs, 'yaxis');
        fprintf('Spectrogram computed on CPU.\n');
    end
else
    % Compute spectrogram on CPU
    [s, f, t] = spectrogram(complexSignal_cpu, hamming(128), 64, 256, 1024, 'yaxis');
    fprintf('Spectrogram computed on CPU.\n');
end

% Compute time-frequency ridges
[fridge, iridge] = tfridge(s, f);
fprintf('Time-frequency ridges extracted.\n');

% 4. Estimate instantaneous bandwidth (CPU only)
ibw = instbw(complexSignal_cpu, 1024); % Assuming a sample rate of 1024 Hz
fprintf('Instantaneous bandwidth estimated.\n');

% 5. Spectral analysis (CPU only)
[pxx, f_pxx] = periodogram(complexSignal_cpu, [], [], 1024);
se = spectralEntropy(pxx, f_pxx);
sf = spectralFlatness(pxx, f_pxx);
sk = spectralKurtosis(pxx, f_pxx);
ss = spectralSkewness(pxx, f_pxx);
fprintf('Spectral features calculated.\n');

% 6. Signal anomaly detection (CPU only)
detector = signalFrequencyFeatureExtractor('SampleRate', 1024);
detector.FrameSize = 64;
detector.FrameOverlapLength = 32;
detector.WelchPSD = true;
features = detector.extract(complexSignal_cpu);
fprintf('Signal features extracted for anomaly detection.\n');

% 7. Extract ROIs (CPU only)
roiLimits = [1, 256; 257, 512; 513, 768; 769, 1024]; % Example ROI limits
roi = extractsigroi(abs(complexSignal_cpu), roiLimits);
fprintf('Regions of Interest (ROIs) extracted.\n');

% Visualizations
figure('Position', [100, 100, 1800, 1200]); % Increased figure size for better layout

% 1. Spectrogram
subplot(4, 4, 1);
spectrogram(complexSignal_cpu, hamming(128), 64, 256, 1024, 'yaxis');
title('Signal Spectrogram');
xlabel('Time (s)');
ylabel('Frequency (Hz)');
colorbar;

% 2. Extracted Signal ROIs with PSD
subplot(4, 4, 2);
hold on;
colors = lines(length(roi));
for i = 1:length(roi)
    if useGPU
        roi_data_gpu = gpuArray(roi{i});
        try
            [pxx_roi_gpu, f_roi_gpu] = pwelch(roi_data_gpu, [], [], [], 1024);
            pxx_roi = gather(pxx_roi_gpu);
            f_roi = gather(f_roi_gpu);
            fprintf('PSD computed on GPU for ROI %d.\n', i);
        catch ME
            warning('PSD computation on GPU for ROI %d failed: %s\nAttempting to compute on CPU.', i, ME.message);
            [pxx_roi, f_roi] = pwelch(roi{i}, [], [], [], 1024);
        end
    else
        [pxx_roi, f_roi] = pwelch(roi{i}, [], [], [], 1024);
    end
    plot(f_roi, 10*log10(pxx_roi), 'Color', colors(i,:));
end
hold off;
title('PSD of Extracted Signal ROIs');
xlabel('Frequency (Hz)');
ylabel('Power/Frequency (dB/Hz)');
legend(cellstr(num2str((1:length(roi))')), 'Location', 'bestoutside');

% 3. Wavelet Scattering Plot (CPU only)
subplot(4, 4, 3);
sn = waveletScattering('SignalLength', numel(complexSignal_cpu), 'SamplingFrequency', 1024);
[wst_real, wstInfo] = featureMatrix(sn, real(complexSignal_cpu));
[wst_imag, ~] = featureMatrix(sn, imag(complexSignal_cpu));
wst = [wst_real; wst_imag]; % Combine real and imaginary parts
imagesc(wst);
title('Wavelet Scattering Transform (Real & Imag)');
xlabel('Scattering Path');
ylabel('Time');
colorbar;

% 4. Plot time-frequency ridge
subplot(4, 4, 4);
imagesc(t, f, log10(abs(s) + eps)); % Use log scale for better visibility
hold on;
plot(t, fridge, 'r', 'LineWidth', 2);
hold off;
axis xy;
title('Time-Frequency Ridge (Log Scale)');
xlabel('Time');
ylabel('Frequency');
colorbar;

% 5. Plot instantaneous bandwidth
subplot(4, 4, 5);
plot(ibw);
title('Instantaneous Bandwidth');
xlabel('Sample');
ylabel('Bandwidth');

% 6. Plot spectral features with color (CPU only)
subplot(4, 4, 6);
meanFeatures = mean(features, 1);
stdFeatures = std(features, 0, 1);

% Create color map
cmap = jet(256);
colormap(cmap);

% Normalize mean features to map to color scale
normalizedMean = (meanFeatures - min(meanFeatures)) / (max(meanFeatures) - min(meanFeatures));
colors_plot = interp1(linspace(0,1,256), cmap, normalizedMean, 'linear', 'extrap');

% Plot with color
for i = 1:length(meanFeatures)
    h = errorbar(i, meanFeatures(i), stdFeatures(i), 'o');
    set(h, 'Color', colors_plot(i,:), 'MarkerFaceColor', colors_plot(i,:));
    hold on;
end

title('Mean Spectral Features with Standard Deviation');
xlabel('Feature Index');
ylabel('Feature Value');
xlim([0 length(meanFeatures)+1]);

% Add colorbar
c = colorbar;
ylabel(c, 'Normalized Feature Magnitude');

% Adjust colorbar limits to match the data
caxis([0 1]); % Since normalizedMean is between 0 and 1

hold off;

% 7. STFT with Instantaneous Frequency (GPU accelerated if possible)
subplot(4, 4, 7);
if useGPU
    % Define window on CPU and transfer to GPU
    window_cpu_stft = hamming(128);
    window_gpu_stft = gpuArray(window_cpu_stft);
    overlap_stft = 64;
    nfft_stft = 256;
    fs = 1024; % Sample rate

    % Compute STFT on GPU
    try
        [S_stft_gpu, F_stft, T_stft] = stft(complexSignal_gpu, fs, 'Window', window_gpu_stft, ...
            'OverlapLength', overlap_stft, 'FFTLength', nfft_stft);
        fprintf('STFT computed on GPU.\n');

        % Move STFT data to CPU
        S_stft = gather(S_stft_gpu);
        F_stft = gather(F_stft);
        T_stft = gather(T_stft);
    catch ME
        warning('STFT computation on GPU failed: %s\nAttempting to compute on CPU.', ME.message);
        [S_stft, F_stft, T_stft] = stft(complexSignal_cpu, fs, 'Window', window_cpu_stft, ...
            'OverlapLength', overlap_stft, 'FFTLength', nfft_stft);
        fprintf('STFT computed on CPU.\n');
    end
else
    [S_stft, F_stft, T_stft] = stft(complexSignal_cpu, fs, 'Window', hamming(128), ...
        'OverlapLength', 64, 'FFTLength', 256);
    fprintf('STFT computed on CPU.\n');
end

% Compute instantaneous frequency
if useGPU
    try
        [instf_gpu, t_instf_gpu] = instfreq(complexSignal_gpu, fs);
        instf = gather(instf_gpu);
        t_instf = gather(t_instf_gpu);
    catch ME
        warning('Instantaneous frequency computation on GPU failed: %s\nAttempting to compute on CPU.', ME.message);
        [instf, t_instf] = instfreq(complexSignal_cpu, fs);
    end
else
    [instf, t_instf] = instfreq(complexSignal_cpu, fs);
end

% Plot STFT Spectrogram
imagesc(T_stft, F_stft, abs(S_stft));
axis xy;
hold on;
plot(t_instf, instf, 'r', 'LineWidth', 2);
hold off;
title('STFT with Instantaneous Frequency');
xlabel('Time (s)');
ylabel('Frequency (Hz)');
colorbar;

% 8. CWT Scalogram (CPU only)
subplot(4, 4, 8);
fb = cwtfilterbank('SignalLength', numel(complexSignal_cpu), 'SamplingFrequency', 1024);
[cfs, frq] = cwt(complexSignal_cpu, 'FilterBank', fb);
t_cwt = (0:numel(complexSignal_cpu)-1) / 1024;
cfs_mag = abs(cfs);

% Handle 3D cfs_mag
if size(cfs_mag, 3) > 1
    % Take the mean across the third dimension
    cfs_mag = mean(cfs_mag, 3);
end

% Ensure cfs_mag is oriented correctly
if size(cfs_mag, 1) ~= length(frq)
    cfs_mag = cfs_mag';
end

disp(['Size of cfs_mag: ', mat2str(size(cfs_mag))]);
disp(['Length of t_cwt: ', num2str(length(t_cwt))]);
disp(['Length of frq: ', num2str(length(frq))]);

imagesc(t_cwt, frq, cfs_mag);
axis xy;
colormap('jet');
xlabel('Time (s)'); ylabel('Frequency (Hz)');
title('CWT Scalogram');
colorbar;

% 9. MODWT (CPU only)
subplot(4, 4, 9);
wt = modwt(real(complexSignal_cpu), 'sym4', 5);
levels = size(wt, 1);
t_modwt = 1:size(wt, 2);
for i = 1:levels
    plot(t_modwt, wt(i,:) + i*2);
    hold on;
end
hold off;
yticks(2:2:levels*2);
yticklabels(1:levels);
xlabel('Time');
ylabel('Level');
title('MODWT Coefficients');

% 10. STFT Spectrogram (CPU only)
subplot(4, 4, 10);
[S_spec, F_spec, T_spec] = spectrogram(real(complexSignal_cpu), hamming(256), 128, 256, 1024);
S_db = 10*log10(abs(S_spec) + eps);
surf(T_spec, F_spec, S_db, 'EdgeColor', 'none');
axis tight; view(0, 90);
xlabel('Time (s)'); ylabel('Frequency (Hz)');
title('STFT Spectrogram');
colorbar;

% 11. Wavelet Scattering Features Difference (Real - Imag) (CPU only)
subplot(4, 4, 11);
sn = waveletScattering('SignalLength', numel(complexSignal_cpu), 'SamplingFrequency', 1024);
scatter_features_real = featureMatrix(sn, real(complexSignal_cpu));
scatter_features_imag = featureMatrix(sn, imag(complexSignal_cpu));
scatter_features_diff = scatter_features_real - scatter_features_imag;
imagesc(scatter_features_diff);
xlabel('Scattering Path');
ylabel('Time');
title('Wavelet Scattering Features (Real - Imag)');
colorbar;
colormap(jet); % Use jet colormap for better visibility of differences

% 12. PCA on the entire dataset (CPU only)
subplot(4, 4, 12);
signal_power = sum(abs(X_test).^2, 2); % [n,1]

try
    if useGPU
        X_pca_cpu = gather(reshape(X_test_gpu, size(X_test,1), [])); % Reshape to 2D for PCA
        signal_power_cpu = gather(signal_power);
    else
        X_pca_cpu = reshape(X_test, size(X_test,1), []); % [n, m*p]
        signal_power_cpu = signal_power;
    end

    % Perform PCA
    [coeff, score, latent] = pca(X_pca_cpu);

    % Ensure signal_power matches the number of points in score
    signal_power_plot = signal_power_cpu(1:size(score,1));

    % Print diagnostic information
    disp(['Size of X_pca_cpu: ', mat2str(size(X_pca_cpu))]);
    disp(['Size of score: ', mat2str(size(score))]);
    disp(['Size of signal_power_plot: ', mat2str(size(signal_power_plot))]);

    scatter(score(:,1), score(:,2), 10, signal_power_plot, 'filled');
    xlabel('First Principal Component');
    ylabel('Second Principal Component');
    title('PCA of Dataset (Colored by Signal Power)');
    colorbar;
    colormap(jet);
catch ME
    warning('%s\nTrying with a subset of data...', ME.message);
    if useGPU
        subset = gather(reshape(X_test_gpu(1:min(5000, size(X_test_gpu, 1)), :, :), ...
            min(5000, size(X_test_gpu,1)), []));
        subset_power = gather(signal_power(1:min(5000, size(X_test, 1))));
    else
        subset = reshape(X_test(1:min(5000, size(X_test, 1)), :, :), ...
            min(5000, size(X_test,1)), []);
        subset_power = signal_power(1:min(5000, size(X_test, 1)));
    end
    [coeff, score, latent] = pca(subset);

    % Print diagnostic information
    disp(['Size of score (subset): ', mat2str(size(score))]);
    disp(['Size of subset_power: ', mat2str(size(subset_power))]);

    scatter(score(:,1), score(:,2), 10, subset_power, 'filled');
    xlabel('First Principal Component');
    ylabel('Second Principal Component');
    title('PCA of Dataset Subset (Colored by Signal Power)');
    colorbar;
    colormap(jet);
end

% 13. t-SNE on a subset of the data (CPU only)
subplot(4, 4, 13);
rng default % for reproducibility
if useGPU
    subset = gather(reshape(X_test_gpu(1:1000, :, :), 1000, []));
    subset_power = gather(signal_power(1:1000));
else
    subset = reshape(X_test(1:1000, :, :), 1000, []); % [1000, m*p]
    subset_power = signal_power(1:1000);
end

% Print diagnostic information
disp(['Size of subset: ', mat2str(size(subset))]);
disp(['Size of subset_power: ', mat2str(size(subset_power))]);

% Check if subset has at least two samples
if size(subset,1) < 2
    error('Subset for t-SNE must contain at least two samples.');
end

Y = tsne(subset, 'Verbose', 1); % Add 'Verbose', 1 to show progress
scatter(Y(:,1), Y(:,2), 10, subset_power, 'filled');
xlabel('t-SNE 1'); ylabel('t-SNE 2');
title('t-SNE of Subset (Colored by Signal Power)');
colorbar;
colormap(jet);

% 14. PCA and t-SNE Enhancements (Optional GPU Steps)
% Note: Advanced GPU-based PCA and t-SNE implementations require specialized toolboxes
% or custom implementations and are not covered in this script.

% Display spectral analysis results
fprintf('Spectral Entropy: %.4f\n', se);
fprintf('Spectral Flatness: %.4f\n', sf);
fprintf('Spectral Kurtosis: %.4f\n', sk);
fprintf('Spectral Skewness: %.4f\n', ss);
