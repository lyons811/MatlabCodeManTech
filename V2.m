% MATLAB Script for DeepRadar2022 Analysis with GPU Utilization
% This script demonstrates various signal processing techniques on the DeepRadar2022 dataset
% utilizing GPU processing.

% Initialize GPU
gpuDevice(1);
fprintf('GPU enabled for processing.\n');

% Load data
try
    load('X_test.mat');
    X_test_gpu = gpuArray(double(X_test));
    fprintf('Data loaded onto the GPU. Size: %s\n', mat2str(size(X_test_gpu)));
catch
    error('Unable to load X_test.mat. Please ensure the file is in the current directory.');
end

% Select a random signal for analysis
signalIndex = randi(size(X_test_gpu, 1));
if ndims(X_test_gpu) == 3
    signal = squeeze(X_test_gpu(signalIndex, :, :)); % [m, p]
else
    error('X_test must be a 3D array with dimensions [n, m, p].');
end

% Ensure signal has at least two columns for I and Q
if size(signal, 2) < 2
    error('Selected signal does not have at least two columns for I and Q components.');
end

I = signal(:, 1);
Q = signal(:, 2);
complexSignal_gpu = I + 1i*Q;
complexSignal_cpu = gather(complexSignal_gpu);

% Find abrupt changes in signal
changePoints = findchangepts(abs(complexSignal_cpu), 'MaxNumChanges', 5);
fprintf('Detected change points at indices: %s\n', mat2str(changePoints));

% Time-frequency analysis using Spectrogram
window_gpu = gpuArray(hamming(128));
[s_gpu, f, t] = spectrogram(complexSignal_gpu, window_gpu, 64, 256, 1024, 'yaxis');
s = gather(s_gpu);
fprintf('Spectrogram computed on GPU.\n');

% Compute time-frequency ridges
[fridge, iridge] = tfridge(s, f);
fprintf('Time-frequency ridges extracted.\n');

% Estimate instantaneous bandwidth
ibw = instbw(complexSignal_cpu, 1024);
fprintf('Instantaneous bandwidth estimated.\n');

% Spectral analysis
[pxx, f_pxx] = periodogram(complexSignal_cpu, [], [], 1024);
se = spectralEntropy(pxx, f_pxx);
sf = spectralFlatness(pxx, f_pxx);
sk = spectralKurtosis(pxx, f_pxx);
ss = spectralSkewness(pxx, f_pxx);
fprintf('Spectral features calculated.\n');

% Signal anomaly detection
detector = signalFrequencyFeatureExtractor('SampleRate', 1024);
detector.FrameSize = 64;
detector.FrameOverlapLength = 32;
detector.WelchPSD = true;
features = detector.extract(complexSignal_cpu);
fprintf('Signal features extracted for anomaly detection.\n');

% Extract ROIs
roiLimits = [1, 256; 257, 512; 513, 768; 769, 1024];
roi = extractsigroi(abs(complexSignal_cpu), roiLimits);
fprintf('Regions of Interest (ROIs) extracted.\n');

% Visualizations
figure('Position', [100, 100, 1800, 1200]);

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
    roi_data_gpu = gpuArray(roi{i});
    [pxx_roi_gpu, f_roi_gpu] = pwelch(roi_data_gpu, [], [], [], 1024);
    pxx_roi = gather(pxx_roi_gpu);
    f_roi = gather(f_roi_gpu);
    plot(f_roi, 10*log10(pxx_roi), 'Color', colors(i,:));
end
hold off;
title('PSD of Extracted Signal ROIs');
xlabel('Frequency (Hz)');
ylabel('Power/Frequency (dB/Hz)');
legend(cellstr(num2str((1:length(roi))')), 'Location', 'bestoutside');

% 3. Wavelet Scattering Plot
subplot(4, 4, 3);
sn = waveletScattering('SignalLength', numel(complexSignal_cpu), 'SamplingFrequency', 1024);
[wst_real, wstInfo] = featureMatrix(sn, real(complexSignal_cpu));
[wst_imag, ~] = featureMatrix(sn, imag(complexSignal_cpu));
wst = [wst_real; wst_imag];
imagesc(wst);
title('Wavelet Scattering Transform (Real & Imag)');
xlabel('Scattering Path');
ylabel('Time');
colorbar;

% 4. Plot time-frequency ridge
subplot(4, 4, 4);
imagesc(t, f, log10(abs(s) + eps));
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

% 6. Plot spectral features with color
subplot(4, 4, 6);
meanFeatures = mean(features, 1);
stdFeatures = std(features, 0, 1);
cmap = jet(256);
colormap(cmap);
normalizedMean = (meanFeatures - min(meanFeatures)) / (max(meanFeatures) - min(meanFeatures));
colors_plot = interp1(linspace(0,1,256), cmap, normalizedMean, 'linear', 'extrap');
for i = 1:length(meanFeatures)
    h = errorbar(i, meanFeatures(i), stdFeatures(i), 'o');
    set(h, 'Color', colors_plot(i,:), 'MarkerFaceColor', colors_plot(i,:));
    hold on;
end
title('Mean Spectral Features with Standard Deviation');
xlabel('Feature Index');
ylabel('Feature Value');
xlim([0 length(meanFeatures)+1]);
c = colorbar;
ylabel(c, 'Normalized Feature Magnitude');
caxis([0 1]);
hold off;

% 7. STFT with Instantaneous Frequency
subplot(4, 4, 7);
window_gpu_stft = gpuArray(hamming(128));
[S_stft_gpu, F_stft, T_stft] = stft(complexSignal_gpu, 1024, 'Window', window_gpu_stft, ...
    'OverlapLength', 64, 'FFTLength', 256);
S_stft = gather(S_stft_gpu);
imagesc(T_stft, F_stft, abs(S_stft));
axis xy;
title('STFT');
xlabel('Time (s)');
ylabel('Frequency (Hz)');
colorbar;

% 8. CWT Scalogram
subplot(4, 4, 8);
fb = cwtfilterbank('SignalLength', numel(complexSignal_cpu), 'SamplingFrequency', 1024);
[cfs, frq] = cwt(complexSignal_cpu, 'FilterBank', fb);
t_cwt = (0:numel(complexSignal_cpu)-1) / 1024;
cfs_mag = abs(cfs);
if size(cfs_mag, 3) > 1
    cfs_mag = mean(cfs_mag, 3);
end
if size(cfs_mag, 1) ~= length(frq)
    cfs_mag = cfs_mag';
end
imagesc(t_cwt, frq, cfs_mag);
axis xy;
colormap('jet');
xlabel('Time (s)'); ylabel('Frequency (Hz)');
title('CWT Scalogram');
colorbar;

% 9. MODWT
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

% 10. STFT Spectrogram
subplot(4, 4, 10);
[S_spec, F_spec, T_spec] = spectrogram(real(complexSignal_cpu), hamming(256), 128, 256, 1024);
S_db = 10*log10(abs(S_spec) + eps);
surf(T_spec, F_spec, S_db, 'EdgeColor', 'none');
axis tight; view(0, 90);
xlabel('Time (s)'); ylabel('Frequency (Hz)');
title('STFT Spectrogram');
colorbar;

% 11. Wavelet Scattering Features Difference (Real - Imag)
subplot(4, 4, 11);
scatter_features_real = featureMatrix(sn, real(complexSignal_cpu));
scatter_features_imag = featureMatrix(sn, imag(complexSignal_cpu));
scatter_features_diff = scatter_features_real - scatter_features_imag;
imagesc(scatter_features_diff);
xlabel('Scattering Path');
ylabel('Time');
title('Wavelet Scattering Features (Real - Imag)');
colorbar;
colormap(jet);

% 12. PCA on the entire dataset
subplot(4, 4, 12);
signal_power = sum(abs(X_test_gpu).^2, 2);
X_pca = gather(reshape(X_test_gpu, size(X_test_gpu,1), []));
signal_power_cpu = gather(signal_power);
[coeff, score, latent] = pca(X_pca);
signal_power_plot = signal_power_cpu(1:size(score,1));
scatter(score(:,1), score(:,2), 10, signal_power_plot, 'filled');
xlabel('First Principal Component');
ylabel('Second Principal Component');
title('PCA of Dataset (Colored by Signal Power)');
colorbar;
colormap(jet);

% 13. t-SNE on a subset of the data
subplot(4, 4, 13);
rng default
subset = gather(reshape(X_test_gpu(1:1000, :, :), 1000, []));
subset_power = gather(signal_power(1:1000));
Y = tsne(subset, 'Verbose', 1);
scatter(Y(:,1), Y(:,2), 10, subset_power, 'filled');
xlabel('t-SNE 1'); ylabel('t-SNE 2');
title('t-SNE of Subset (Colored by Signal Power)');
colorbar;
colormap(jet);

% 14. Energy Detector Plot
subplot(4, 4, 14);
energy_threshold = 0.7 * max(abs(complexSignal_gpu).^2); % 70% of max energy as threshold
signal_energy = abs(complexSignal_gpu).^2;
detected_signals = signal_energy > energy_threshold;
t = (0:length(complexSignal_gpu)-1) / 1024; % Assuming 1024 Hz sampling rate
plot(t, signal_energy);
hold on;
plot(t, energy_threshold * ones(size(t)), 'r--');
plot(t(detected_signals), signal_energy(detected_signals), 'ro');
hold off;
title('Energy Detector');
xlabel('Time (s)');
ylabel('Signal Energy');
legend('Signal Energy', 'Threshold', 'Detected Signals');

% 16. 3D Spectrogram Plot (replacing Waterfall Plot)
subplot(4, 4, 15);
[S, F, T] = spectrogram(complexSignal_gpu, hamming(256), 128, 512, 1024);
S_magnitude = 10*log10(abs(S) + eps); % Convert to dB scale
surf(T, F, S_magnitude, 'EdgeColor', 'none');
axis tight;
view(-15, 60); % Adjust view angle
xlabel('Time (s)');
ylabel('Frequency (Hz)');
zlabel('Magnitude (dB)');
title('3D Spectrogram');
colorbar;

% Display spectral analysis results
fprintf('Spectral Entropy: %.4f\n', se);
fprintf('Spectral Flatness: %.4f\n', sf);
fprintf('Spectral Kurtosis: %.4f\n', sk);
fprintf('Spectral Skewness: %.4f\n', ss);
fprintf('Energy Detector: %d signals detected\n', sum(detected_signals));