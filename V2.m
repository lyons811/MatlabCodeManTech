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

% Plot original signal
subplot(4, 3, 1);
plot(abs(complexSignal_cpu));
title('Original Signal Magnitude');
xlabel('Sample');
ylabel('Magnitude');

% Plot ROI
subplot(4, 3, 2);
hold on;
for i = 1:length(roi)
    plot(roiLimits(i,1):roiLimits(i,2), roi{i});
end
hold off;
title('Extracted Signal ROIs');
xlabel('Sample');
ylabel('Magnitude');

% Plot signal with change points
subplot(4, 3, 3);
plot(abs(complexSignal_cpu));
hold on;
plot(changePoints, abs(complexSignal_cpu(changePoints)), 'ro');
title('Signal with Change Points');
xlabel('Sample');
ylabel('Magnitude');

% Plot time-frequency ridge
subplot(4, 3, 4);
imagesc(t, f, abs(s));
hold on;
plot(t, fridge, 'r', 'LineWidth', 2);
hold off;
axis xy;
title('Time-Frequency Ridge');
xlabel('Time');
ylabel('Frequency');
colorbar;

% Plot instantaneous bandwidth
subplot(4, 3, 5);
plot(ibw);
title('Instantaneous Bandwidth');
xlabel('Sample');
ylabel('Bandwidth');

% Plot spectral features
subplot(4, 3, 6);
plot(features);
title('Spectral Features');
xlabel('Frame');
ylabel('Feature Value');

% CWT Scalogram
subplot(4, 3, 7);
fb = cwtfilterbank('SignalLength', numel(complexSignal_cpu), 'SamplingFrequency', 1024);
[cfs, frq] = cwt(complexSignal_cpu, 'FilterBank', fb);

% Create time vector
t = (0:numel(complexSignal_cpu)-1) / 1024;

% Calculate magnitude of CWT coefficients
cfs_mag = abs(cfs(:,:,1) + 1i*cfs(:,:,2));

% Plot the scalogram
imagesc(t, frq, cfs_mag);
axis xy; % To ensure low frequencies are at the bottom
colormap('jet');
xlabel('Time (s)'); ylabel('Frequency (Hz)');
title('CWT Scalogram');
colorbar;

% Display size information
disp(['Size of complexSignal_cpu: ', num2str(size(complexSignal_cpu))]);
disp(['Size of cfs: ', num2str(size(cfs))]);
disp(['Size of cfs_mag: ', num2str(size(cfs_mag))]);


% MODWT
subplot(4, 3, 8);
wt = modwt(real(complexSignal_cpu), 'sym4', 5);
% Replace plotmodwt with custom plotting
levels = size(wt, 1);
t = 1:size(wt, 2);
for i = 1:levels
    plot(t, wt(i,:) + i*2); % Offset each level for visibility
    hold on;
end
hold off;
yticks(2:2:levels*2);
yticklabels(1:levels);
xlabel('Time');
ylabel('Level');
title('MODWT Coefficients');

% Replace Mel Spectrogram with STFT
subplot(4, 3, 9);
[S,F,T] = spectrogram(real(complexSignal_cpu), hamming(256), 128, 256, 1024);
S_db = 10*log10(abs(S) + eps);  % Convert to dB scale
surf(T, F, S_db, 'EdgeColor', 'none');
axis tight; view(0, 90);
xlabel('Time (s)'); ylabel('Frequency (Hz)');
title('STFT Spectrogram');
colorbar;

% Wavelet Scattering Transform
subplot(4, 3, 10);
sn = waveletScattering('SignalLength', numel(complexSignal_cpu), 'SamplingFrequency', 1024);
scatter_features_real = featureMatrix(sn, real(complexSignal_cpu));
scatter_features_imag = featureMatrix(sn, imag(complexSignal_cpu));
scatter_features = [scatter_features_real; scatter_features_imag];  % Combine features
imagesc(scatter_features);
xlabel('Scattering Path'); ylabel('Time');
title('Wavelet Scattering Features (Real & Imag)');
colorbar;

% PCA on the entire dataset
subplot(4, 3, 11);
try
    [coeff, score, latent] = pca(gather(X_test(:,:)));
    scatter(score(:,1), score(:,2), 10, 'filled');
    xlabel('First Principal Component');
    ylabel('Second Principal Component');
    title('PCA of Dataset');
catch ME
    warning('PCA failed: %s\nTrying with a subset of data...', ME.message);
    subset = X_test(1:min(5000, size(X_test, 1)), :);
    [coeff, score, latent] = pca(gather(subset));
    scatter(score(:,1), score(:,2), 10, 'filled');
    xlabel('First Principal Component');
    ylabel('Second Principal Component');
    title('PCA of Dataset Subset');
end

% t-SNE on a subset of the data
subplot(4, 3, 12);
rng default % for reproducibility
subset = gather(X_test(1:1000,:));
Y = tsne(subset, 'Verbose', 1);  % Add 'Verbose', 1 to show progress
scatter(Y(:,1), Y(:,2), 10, 'filled');
xlabel('t-SNE 1'); ylabel('t-SNE 2');
title('t-SNE of Subset');

% Display spectral analysis results
fprintf('Spectral Entropy: %.4f\n', se);
fprintf('Spectral Flatness: %.4f\n', sf);
fprintf('Spectral Kurtosis: %.4f\n', sk);
fprintf('Spectral Skewness: %.4f\n', ss);