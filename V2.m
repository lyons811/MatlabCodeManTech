% MATLAB Script for DeepRadar2022 Analysis
% This script demonstrates various signal processing techniques on the DeepRadar2022 dataset

% Enable GPU processing
if gpuDeviceCount > 0
    gpuDevice(1);
    useGPU = true;
else
    useGPU = false;
end

% Load data
load('X_test.mat');
X_test = double(X_test); % Ensure double precision
if useGPU
    X_test = gpuArray(X_test);
end

% Select a random signal for analysis
signalIndex = randi(size(X_test, 1));
signal = squeeze(X_test(signalIndex, :, :));
I = signal(:, 1);
Q = signal(:, 2);
complexSignal = I + 1i*Q;

% Ensure complexSignal is in double precision and on CPU for findchangepts
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

% 7. Extract ROIs (added this step)
roiLimits = [1, 256; 257, 512; 513, 768; 769, 1024]; % Example ROI limits
roi = extractsigroi(abs(complexSignal_cpu), roiLimits);

% Visualizations
figure;

% Plot original signal
subplot(3, 2, 1);
plot(abs(complexSignal_cpu));
title('Original Signal Magnitude');
xlabel('Sample');
ylabel('Magnitude');

% Plot ROI
subplot(3, 2, 2);
hold on;
for i = 1:length(roi)
    plot(roiLimits(i,1):roiLimits(i,2), roi{i});
end
hold off;
title('Extracted Signal ROIs');
xlabel('Sample');
ylabel('Magnitude');

% Plot signal with change points
subplot(3, 2, 3);
plot(abs(complexSignal_cpu));
hold on;
plot(changePoints, abs(complexSignal_cpu(changePoints)), 'ro');
title('Signal with Change Points');
xlabel('Sample');
ylabel('Magnitude');

% Plot time-frequency ridge
subplot(3, 2, 4);
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
subplot(3, 2, 5);
plot(ibw);
title('Instantaneous Bandwidth');
xlabel('Sample');
ylabel('Bandwidth');

% Plot spectral features
subplot(3, 2, 6);
plot(features);
title('Spectral Features');
xlabel('Frame');
ylabel('Feature Value');

% Display spectral analysis results
fprintf('Spectral Entropy: %.4f\n', se);
fprintf('Spectral Flatness: %.4f\n', sf);
fprintf('Spectral Kurtosis: %.4f\n', sk);
fprintf('Spectral Skewness: %.4f\n', ss);