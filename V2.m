% MATLAB Script for DeepRadar2022 Analysis
% This script demonstrates various signal processing techniques on the DeepRadar2022 dataset

% Enable GPU processing if available
useGPU = false;
if gpuDeviceCount > 0
    gpuDevice(1);
    useGPU = true;
end

% Load data
try
    load('X_test.mat');
    X_test = double(X_test); % Ensure double precision
    if useGPU
        X_test = gpuArray(X_test);
    end
catch
    error('Unable to load X_test.mat. Please ensure the file is in the current directory.');
end

% Select a random signal for analysis
signalIndex = randi(size(X_test, 1));
signal = squeeze(X_test(signalIndex, :, :));
I = signal(:, 1);
Q = signal(:, 2);
complexSignal = I + 1i*Q;

% Ensure complexSignal is in double precision and on CPU for processing
complexSignal_cpu = gather(double(complexSignal));

% 2. Find abrupt changes in signal
changePoints = findchangepts(abs(complexSignal_cpu), 'MaxNumChanges', 5);

% 3. Time-frequency ridges
[s, f, t] = spectrogram(complexSignal_cpu);
[fridge, iridge] = tfridge(s, f);

% 4. Estimate instantaneous bandwidth
ibw = instbw(complexSignal_cpu, 1024); % Assuming a sample rate of 1024 Hz

% 5. Spectral analysis
[pxx, f] = periodogram(complexSignal_cpu);
se = spectralEntropy(pxx, f);
sf = spectralFlatness(pxx, f);
sk = spectralKurtosis(pxx, f);
ss = spectralSkewness(pxx, f);

% 6. Signal anomaly detection
detector = signalFrequencyFeatureExtractor('SampleRate', 1024);
detector.FrameSize = 64;
detector.FrameOverlapLength = 32;
detector.WelchPSD = true;
features = detector.extract(complexSignal_cpu);

% 7. Extract ROIs
roiLimits = [1, 256; 257, 512; 513, 768; 769, 1024]; % Example ROI limits
roi = extractsigroi(abs(complexSignal_cpu), roiLimits);

% Visualizations
figure('Position', [100, 100, 1200, 1200]);

% Spectrogram
subplot(4, 3, 1);
spectrogram(complexSignal_cpu, hamming(128), 64, 256, 1024, 'yaxis');
title('Signal Spectrogram');
xlabel('Time (s)');
ylabel('Frequency (Hz)');
colorbar;

% Extracted Signal ROIs with PSD
subplot(4, 3, 2);
hold on;
colors = lines(length(roi));
for i = 1:length(roi)
    [pxx, f] = pwelch(roi{i}, [], [], [], 1024);
    plot(f, 10*log10(pxx), 'Color', colors(i,:));
end
hold off;
title('PSD of Extracted Signal ROIs');
xlabel('Frequency (Hz)');
ylabel('Power/Frequency (dB/Hz)');
legend(cellstr(num2str((1:length(roi))')), 'Location', 'bestoutside');

% Wavelet Scattering Plot
subplot(4, 3, 3);
sn = waveletScattering('SignalLength', numel(complexSignal_cpu), 'SamplingFrequency', 1024);
[wst_real, wstInfo] = featureMatrix(sn, real(complexSignal_cpu));
[wst_imag, ~] = featureMatrix(sn, imag(complexSignal_cpu));
wst = [wst_real; wst_imag]; % Combine real and imaginary parts
imagesc(wst);
title('Wavelet Scattering Transform (Real & Imag)');
xlabel('Scattering Path');
ylabel('Time');
colorbar;

% Plot time-frequency ridge
subplot(4, 3, 4);
imagesc(t, f, log10(abs(s) + eps)); % Use log scale for better visibility
hold on;
plot(t, fridge, 'r', 'LineWidth', 2);
hold off;
axis xy;
title('Time-Frequency Ridge (Log Scale)');
xlabel('Time');
ylabel('Frequency');
colorbar;

% Plot instantaneous bandwidth
subplot(4, 3, 5);
plot(ibw);
title('Instantaneous Bandwidth');
xlabel('Sample');
ylabel('Bandwidth');


% Plot spectral features with color
subplot(4, 3, 6);
meanFeatures = mean(features, 1);
stdFeatures = std(features, 0, 1);

% Create color map
cmap = jet(256);
colormap(cmap);

% Normalize mean features to map to color scale
normalizedMean = (meanFeatures - min(meanFeatures)) / (max(meanFeatures) - min(meanFeatures));
colors = interp1(linspace(0,1,256), cmap, normalizedMean);

% Plot with color
for i = 1:length(meanFeatures)
    h = errorbar(i, meanFeatures(i), stdFeatures(i), 'o');
    set(h, 'Color', colors(i,:), 'MarkerFaceColor', colors(i,:));
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
caxis([min(meanFeatures) max(meanFeatures)]);

hold off;


% STFT with Instantaneous Frequency
subplot(4, 3, 7);
[s, f, t] = stft(complexSignal_cpu, 1024, 'Window', hamming(128), 'OverlapLength', 64, 'FFTLength', 256);
imagesc(t, f, abs(s));
axis xy;
hold on;
[instf, t_instf] = instfreq(complexSignal_cpu, 1024);
plot(t_instf, instf, 'r', 'LineWidth', 2);
hold off;
title('STFT with Instantaneous Frequency');
xlabel('Time (s)');
ylabel('Frequency (Hz)');
colorbar;

% CWT Scalogram
subplot(4, 3, 8);
fb = cwtfilterbank('SignalLength', numel(complexSignal_cpu), 'SamplingFrequency', 1024);
[cfs, frq] = cwt(complexSignal_cpu, 'FilterBank', fb);
t = (0:numel(complexSignal_cpu)-1) / 1024;
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
disp(['Length of t: ', num2str(length(t))]);
disp(['Length of frq: ', num2str(length(frq))]);

imagesc(t, frq, cfs_mag);
axis xy;
colormap('jet');
xlabel('Time (s)'); ylabel('Frequency (Hz)');
title('CWT Scalogram');
colorbar;

% MODWT
subplot(4, 3, 9);
wt = modwt(real(complexSignal_cpu), 'sym4', 5);
levels = size(wt, 1);
t = 1:size(wt, 2);
for i = 1:levels
    plot(t, wt(i,:) + i*2);
    hold on;
end
hold off;
yticks(2:2:levels*2);
yticklabels(1:levels);
xlabel('Time');
ylabel('Level');
title('MODWT Coefficients');

% STFT Spectrogram
subplot(4, 3, 10);
[S,F,T] = spectrogram(real(complexSignal_cpu), hamming(256), 128, 256, 1024);
S_db = 10*log10(abs(S) + eps);
surf(T, F, S_db, 'EdgeColor', 'none');
axis tight; view(0, 90);
xlabel('Time (s)'); ylabel('Frequency (Hz)');
title('STFT Spectrogram');
colorbar;

% Wavelet Scattering Features Difference (Real - Imag)
subplot(4, 3, 10);
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

% Calculate signal power for coloring
signal_power = sum(abs(X_test).^2, 2);

% PCA on the entire dataset
subplot(4, 3, 11);
try
    [coeff, score, latent] = pca(gather(X_test(:,:)));
    
    % Ensure signal_power matches the number of points in score
    signal_power_plot = gather(signal_power(1:size(score,1)));
    
    % Print diagnostic information
    disp(['Size of score: ', num2str(size(score))]);
    disp(['Size of signal_power_plot: ', num2str(size(signal_power_plot))]);
    
    scatter(score(:,1), score(:,2), 10, signal_power_plot, 'filled');
    xlabel('First Principal Component');
    ylabel('Second Principal Component');
    title('PCA of Dataset (Colored by Signal Power)');
    colorbar;
    colormap(jet);
catch ME
    warning('%s\nTrying with a subset of data...', ME.message);
    subset = X_test(1:min(5000, size(X_test, 1)), :);
    subset_power = signal_power(1:min(5000, size(X_test, 1)));
    [coeff, score, latent] = pca(gather(subset));
    
    % Print diagnostic information
    disp(['Size of score (subset): ', num2str(size(score))]);
    disp(['Size of subset_power: ', num2str(size(subset_power))]);
    
    scatter(score(:,1), score(:,2), 10, gather(subset_power), 'filled');
    xlabel('First Principal Component');
    ylabel('Second Principal Component');
    title('PCA of Dataset Subset (Colored by Signal Power)');
    colorbar;
    colormap(jet);
end

% t-SNE on a subset of the data
subplot(4, 3, 12);
rng default % for reproducibility
subset = gather(X_test(1:1000,:));
subset_power = gather(signal_power(1:1000));

% Print diagnostic information
disp(['Size of subset: ', num2str(size(subset))]);
disp(['Size of subset_power: ', num2str(size(subset_power))]);

Y = tsne(subset, 'Verbose', 1); % Add 'Verbose', 1 to show progress
scatter(Y(:,1), Y(:,2), 10, subset_power, 'filled');
xlabel('t-SNE 1'); ylabel('t-SNE 2');
title('t-SNE of Subset (Colored by Signal Power)');
colorbar;
colormap(jet);

% Display spectral analysis results
fprintf('Spectral Entropy: %.4f\n', se);
fprintf('Spectral Flatness: %.4f\n', sf);
fprintf('Spectral Kurtosis: %.4f\n', sk);
fprintf('Spectral Skewness: %.4f\n', ss);