%% Advanced Signal Analysis and Classification with GPU Acceleration
% This script performs signal analysis, feature extraction, and classification
% using GPU acceleration for improved performance.

%% Initialization and Setup
try
    % Initialize GPU
    gpu = gpuDevice(1);
    fprintf('Using GPU: %s\n', gpu.Name);
catch ME
    error('GPU Initialization Error: %s', ME.message);
end

%% Data Loading
try
    fprintf('Loading data...\n');
    X_train = gpuArray(single(load('X_train.mat').X_train)); % Load directly to GPU
    Y_train = load('Y_train.mat').Y_train;
    lbl_train = load('lbl_train.mat').lbl_train;
    [num_samples, signal_length, ~] = size(X_train);
    fprintf('Data Loaded: %d samples with signal length %d.\n', num_samples, signal_length);
catch ME
    error('Data Loading Error: %s', ME.message);
end

%% Feature Extraction Setup
num_features = 12;
features_matrix = zeros(num_samples, num_features, 'single', 'gpuArray');
batch_size = 10000; % Adjust based on GPU memory
num_batches = ceil(num_samples / batch_size);

% Precompute Frequency Range
freq_range = gpuArray(single(linspace(0, 1, signal_length)));

% Define Feature Names
feature_names = {'Mean Amp', 'Max Amp', 'Std Amp', 'Spectral Centroid', 'Mean Inst Freq', ...
                 'Spectral Entropy', 'Spectral Flatness', 'Spectral Rolloff', ...
                 'Signal Energy', 'Max Autocorr', 'Zero-Cross Rate', 'STFT Mean'};

%% Feature Extraction Functions

% GPU-Compatible Feature Extraction (11 Features)
function features_gpu = extract_features_gpu(signals, freq_range)
    complex_signals = complex(signals(:,:,1), signals(:,:,2));
    fft_signals = fft(complex_signals, [], 2);
    fft_magnitude = abs(fft_signals);
    psd = (fft_magnitude.^2) / size(signals, 2);
    
    % Ensure all operations return column vectors of size [batch_size, 1]
    mean_amp = mean(abs(complex_signals), 2);
    max_amp = max(abs(complex_signals), [], 2);
    std_amp = std(abs(complex_signals), 0, 2);
    spectral_centroid = sum(freq_range .* psd, 2) ./ sum(psd, 2);
    mean_inst_freq = mean(diff(unwrap(angle(complex_signals), [], 2), 1, 2) / (2*pi*(1/size(signals, 2))), 2);
    spectral_entropy = -sum(psd ./ sum(psd, 2) .* log2(psd ./ sum(psd, 2) + eps), 2);
    spectral_flatness = exp(mean(log(psd + eps), 2)) ./ mean(psd, 2);
    spectral_rolloff = sum(cumsum(psd, 2) < 0.85 * sum(psd, 2), 2) / size(signals, 2);
    signal_energy = sum(abs(complex_signals).^2, 2);
    max_autocorr = max(abs(ifft(fft_signals .* conj(fft_signals), [], 2)), [], 2);
    zero_crossing_rate = sum(abs(diff(sign(real(complex_signals)), 1, 2)) > 0, 2) / size(signals, 2);
    
    features_gpu = [mean_amp, max_amp, std_amp, spectral_centroid, mean_inst_freq, ...
                   spectral_entropy, spectral_flatness, spectral_rolloff, ...
                   signal_energy, max_autocorr, zero_crossing_rate];
end

% CPU-Based Feature Extraction (STFT Mean)
function features_cpu = extract_features_cpu(signals)
    num_signals = size(signals, 1);
    features_cpu = zeros(num_signals, 1, 'single');
    for i = 1:num_signals
        signal = squeeze(signals(i, :, :));
        complex_signal = complex(signal(:,1), signal(:,2));
        % Compute STFT on CPU
        S = stft(complex_signal, 'Window', hamming(128, 'periodic'), 'OverlapLength', 120, 'FFTLength', 256);
        features_cpu(i) = mean(abs(S(:)));
    end
end

%% Main Processing Loop
tic;
for i = 1:num_batches
    start_idx = (i-1)*batch_size + 1;
    end_idx = min(i*batch_size, num_samples);
    batch_signals = X_train(start_idx:end_idx, :, :);
    current_batch_size = end_idx - start_idx + 1;
    
    % Extract GPU-Compatible Features
    features_gpu = extract_features_gpu(batch_signals, freq_range);
    
    % Gather to CPU for STFT Feature Extraction
    batch_signals_cpu = gather(batch_signals);
    features_cpu = extract_features_cpu(batch_signals_cpu);
    
    % Debug: Check sizes
    % Uncomment the following line to print sizes
    % fprintf('Batch %d: features_gpu size = %s, features_cpu size = %s\n', i, mat2str(size(features_gpu)), mat2str(size(features_cpu)));
    
    % Convert features_cpu to GPU before concatenation
    features_cpu_gpu = gpuArray(single(features_cpu));
    
    % Combine Features
    features_combined = [features_gpu, features_cpu_gpu];
    
    % Assign to Feature Matrix
    features_matrix(start_idx:end_idx, :) = features_combined;
    
    fprintf('Processed batch %d of %d.\n', i, num_batches);
end
processing_time = toc;
fprintf('Feature extraction completed in %.2f seconds.\n', processing_time);

%% Transfer Features to CPU and Perform PCA
features_matrix = gather(features_matrix);
[coeff, score, ~, ~, explained] = pca(features_matrix);

% Determine number of PCA components to retain 95% variance, with a minimum of 3
cumulative_variance = cumsum(explained);
num_pca_components = find(cumulative_variance >= 95, 1);
if isempty(num_pca_components)
    num_pca_components = size(features_matrix, 2);
end
num_pca_components = max(num_pca_components, 3); % Ensure at least 3 components

% Handle case where data may have fewer features
num_pca_components = min(num_pca_components, size(score, 2));

reduced_features = score(:, 1:num_pca_components);

% Inform the user about the number of PCA components retained
fprintf('Number of PCA components retained: %d\n', num_pca_components);
if num_pca_components < 3
    fprintf('Only %d PCA components available. 2D scatter plot will be used.\n', num_pca_components);
end

%% Prepare Data for Neural Network
[~, true_labels_numeric] = max(Y_train, [], 2);
true_labels = categorical(true_labels_numeric);

% Shuffle and split data
rng(1961);
shuffle_indices = randperm(num_samples);
features_matrix = features_matrix(shuffle_indices, :);
true_labels = true_labels(shuffle_indices);

validation_fraction = 0.2;
num_val = floor(validation_fraction * num_samples);
X_train_nn = features_matrix(1:end-num_val, :);
Y_train_nn = true_labels(1:end-num_val);
X_val_nn = features_matrix(end-num_val+1:end, :);
Y_val_nn = true_labels(end-num_val+1:end);

%% Define and Train Neural Network
layers = [
    featureInputLayer(size(features_matrix,2), 'Normalization', 'zscore')
    fullyConnectedLayer(64)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.3)
    fullyConnectedLayer(32)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.3)
    fullyConnectedLayer(23)
    softmaxLayer
    classificationLayer
];

options = trainingOptions('adam', ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 1024, ...
    'ValidationData', {X_val_nn, Y_val_nn}, ...
    'ValidationFrequency', 30, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'gpu');

net = trainNetwork(X_train_nn, Y_train_nn, layers, options);

%% Evaluate Network
Y_pred = classify(net, X_val_nn, 'ExecutionEnvironment', 'gpu');
accuracy = sum(Y_pred == Y_val_nn) / numel(Y_val_nn) * 100;
fprintf('Validation Accuracy: %.2f%%\n', accuracy);

%% Visualizations

% Single Signal Analysis
sample_index = randi(num_samples);
signal = gather(squeeze(X_train(sample_index, :, :)));
label = Y_train(sample_index, :);
signal_info = lbl_train(sample_index, :);

figure('Name', 'Single Signal Analysis', 'NumberTitle', 'off');

% Time Domain Plot
subplot(2, 3, 1);
time_vector = (0:signal_length-1)' / 100e6;
plot(time_vector(1:200), signal(1:200,1), 'b', time_vector(1:200), signal(1:200,2), 'r');
title('Time Domain (First 200 Samples)');
legend('I', 'Q');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

% Frequency Domain Plot
subplot(2, 3, 2);
freq_domain = abs(fftshift(fft(complex(signal(:,1), signal(:,2)))));
freq_axis = linspace(-50e6, 50e6, length(freq_domain));
plot(freq_axis, freq_domain);
title('Frequency Domain');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
grid on;

% Power Spectral Density
subplot(2, 3, 3);
[psd_vals, freq_psd] = pspectrum(complex(signal(:,1), signal(:,2)), 100e6, 'FrequencyLimits', [0 50e6]);
plot(freq_psd, 10*log10(psd_vals));
title('Power Spectral Density');
xlabel('Frequency (Hz)');
ylabel('Power/Frequency (dB/Hz)');
grid on;

% Spectrogram
subplot(2, 3, 4);
spectrogram(complex(signal(:,1), signal(:,2)), hamming(128, 'periodic'), 120, 256, 100e6, 'yaxis');
title('Spectrogram');

% Instantaneous Frequency Plot
subplot(2, 3, 5);
[inst_freq, time_inst] = instfreq(complex(signal(:,1), signal(:,2)), 100e6);
plot(time_inst, inst_freq);
title('Instantaneous Frequency');
xlabel('Time (s)');
ylabel('Frequency (Hz)');
grid on;

% Feature Display
subplot(2, 3, 6);
single_features = features_matrix(sample_index, :);
bar(single_features);
set(gca, 'XTickLabel', feature_names, 'XTick', 1:num_features);
xtickangle(45);
title('Extracted Features');
ylabel('Feature Value');

% Display Single Signal Features
[~, signal_label] = max(label);
fprintf('\nSingle Signal Analysis (Index: %d):\n', sample_index);
for f = 1:num_features
    fprintf('%s: %.4f\n', feature_names{f}, single_features(f));
end
fprintf('Signal Class: %d\n', signal_label);
fprintf('SNR: %.2f dB\n', signal_info(2));

% Multi-Signal Analysis Visualization
figure('Name', 'Multi-Signal Analysis', 'NumberTitle', 'off');

% 3D Scatter Plot of PCA Components (if at least 3 components are available)
if num_pca_components >= 3
    subplot(2, 2, 1);
    scatter3(reduced_features(:,1), reduced_features(:,2), reduced_features(:,3), 10, true_labels_numeric, 'filled');
    xlabel('PC1'); ylabel('PC2'); zlabel('PC3');
    title('PCA of Signal Features');
    colorbar;
    colormap(jet(max(true_labels_numeric)));
else
    % 2D Scatter Plot if fewer than 3 components
    subplot(2, 2, 1);
    scatter(reduced_features(:,1), reduced_features(:,2), 10, true_labels_numeric, 'filled');
    xlabel('PC1'); ylabel('PC2');
    title('PCA of Signal Features (2D)');
    colorbar;
    colormap(jet(max(true_labels_numeric)));
end

% Feature Correlation Heatmap
subplot(2, 2, 2);
correlation_matrix = corr(features_matrix);
heatmap(correlation_matrix, 'XDisplayLabels', feature_names, 'YDisplayLabels', feature_names);
title('Feature Correlation');

% Feature Distribution Boxplots
subplot(2, 2, 3);
boxplot(features_matrix(:,6), true_labels_numeric);
title('Boxplot of Spectral Entropy per Class');
xlabel('Class');
ylabel('Spectral Entropy');

subplot(2, 2, 4);
boxplot(features_matrix(:,11), true_labels_numeric);
title('Boxplot of Zero-Crossing Rate per Class');
xlabel('Class');
ylabel('Zero-Crossing Rate');

% Confusion Matrix
figure('Name', 'Confusion Matrix', 'NumberTitle', 'off');
confusionchart(Y_val_nn, Y_pred);
title('Confusion Matrix for Signal Classification');

%% Save Results
save('processedFeatures.mat', 'features_matrix', 'reduced_features', 'net', '-v7.3');

%% Cleanup
gpuDevice([]);
fprintf('Script execution completed successfully.\n');
