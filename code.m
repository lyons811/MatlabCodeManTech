% Improved MATLAB Code for Signal Processing and Feature Extraction with AI Integration

%% Initialization and Setup
try
    % Check for available GPUs
    gpu_info = gpuDeviceCount;
    if gpu_info == 0
        error('No GPU device found. Please ensure a compatible GPU is available.');
    end
    gpu_dev = gpuDevice(1); % Initialize the first GPU
    fprintf('Using GPU Device #%d: %s\n', gpu_dev.Index, gpu_dev.Name);
catch ME
    fprintf('GPU Initialization Error: %s\n', ME.message);
    return;
end

%% Load Data
try
    fprintf('Loading data...\n');
    % Load training data
    X_train = load('X_train.mat').X_train; % [469200 x 1024 x 2]
    Y_train = load('Y_train.mat').Y_train; % [469200 x 23]
    lbl_train = load('lbl_train.mat').lbl_train; % [469200 x 6]
    num_samples = size(X_train, 1);
    signal_length = size(X_train, 2);
    fprintf('Data Loaded: %d samples with signal length %d.\n', num_samples, signal_length);
catch ME
    fprintf('Data Loading Error: %s\n', ME.message);
    return;
end

%% Transfer Data to GPU
try
    fprintf('Transferring data to GPU...\n');
    X_train_gpu = gpuArray(single(X_train)); % Convert to single precision
    % Y_train_gpu will be kept on CPU for neural network training
    clear X_train; % Free CPU memory
    fprintf('Data transferred to GPU.\n');
catch ME
    fprintf('Data Transfer Error: %s\n', ME.message);
    return;
end

%% Feature Extraction Parameters
num_features = 12; % Adjusted number of features based on implemented features
features_matrix_gpu = gpuArray.zeros(num_samples, num_features, 'single'); % Use single precision to save memory
batch_size_main = 5000;  % Adjust based on GPU memory
num_batches = ceil(num_samples / batch_size_main);
fprintf('Starting feature extraction with %d batches...\n', num_batches);

% Define Feature Names Globally
feature_names = {'Mean Amp', 'Max Amp', 'Std Amp', 'Spectral Centroid', 'Mean Inst Freq', ...
                'Spectral Entropy', 'Spectral Flatness', 'Spectral Rolloff', ...
                'Signal Energy', 'Max Autocorr', 'Zero-Cross Rate', 'STFT Mean'};

%% Precompute Frequency Range for Spectral Centroid
freq_range = linspace(0, 1, signal_length); % [1 x 1024] Row vector
freq_range_gpu = gpuArray(freq_range); % Transfer to GPU once

%% Define Feature Extraction Function
% It is recommended to define this as a separate function file for better efficiency.
% For illustration, it's kept as a nested function.
% Fixed: Handle 'stft' correctly by gathering to CPU

function features = process_multiple_signals_gpu(signals, signal_length, num_features, freq_range_gpu, feature_names)
    % Convert I/Q to complex signals
    complex_signals = complex(signals(:,:,1), signals(:,:,2)); % [batch_size x signal_length]
    
    % Perform FFT
    fft_signals = fft(complex_signals, [], 2);
    fft_magnitude = abs(fft_signals);
    
    % Power Spectral Density
    psd = (fft_magnitude.^2) / signal_length;
    
    % Basic Features
    mean_amp = mean(abs(complex_signals), 2);
    max_amp = max(abs(complex_signals), [], 2);
    std_amp = std(abs(complex_signals), 0, 2);
    
    % Spectral Centroid
    spectral_centroid = sum(freq_range_gpu .* psd, 2) ./ sum(psd, 2);
    
    % Instantaneous Frequency
    inst_phase = unwrap(angle(complex_signals), [], 2);
    inst_freq = diff(inst_phase, 1, 2) / (2*pi*(1/signal_length));
    mean_inst_freq = mean(inst_freq, 2);
    
    % Spectral Entropy
    psd_norm = psd ./ sum(psd, 2);
    spectral_entropy = -sum(psd_norm .* log2(psd_norm + eps), 2);
    
    % Spectral Flatness
    spectral_flatness = geomean(psd, 2) ./ mean(psd, 2);
    
    % Spectral Rolloff (85%)
    cumsum_psd = cumsum(psd, 2);
    thresholds = 0.85 * sum(psd, 2);
    rolloff_bins = sum(cumsum_psd < thresholds, 2);
    spectral_rolloff = rolloff_bins / signal_length;
    
    % Signal Energy
    signal_energy = sum(abs(complex_signals).^2, 2);
    
    % Max Autocorrelation Value
    autocorr_vals = ifft(fft_signals .* conj(fft_signals), [], 2);
    max_autocorr = max(abs(autocorr_vals), [], 2);
    
    % Zero-Crossing Rate
    zero_crossings = sum(abs(diff(sign(real(complex_signals)), 1, 2)) > 0, 2);
    zero_crossing_rate = zero_crossings / signal_length;
    
    % Deep Learning STFT Features
    try
        % Gather complex_signals to CPU for STFT processing
        complex_signals_cpu = gather(complex_signals); % [batch_size x signal_length]
        
        % Initialize STFT features array
        stft_features_mean_cpu = zeros(size(complex_signals_cpu,1),1,'single');
        
        window = hamming(128, 'periodic'); % Define window vector
        overlap = 120;
        fft_length = 256;
        for idx = 1:size(complex_signals_cpu,1)
            signal = complex_signals_cpu(idx,:);
            % Compute STFT
            [s, ~, ~] = stft(signal, 'Window', window, 'OverlapLength', overlap, 'FFTLength', fft_length);
            % Compute mean magnitude
            stft_features_mean_cpu(idx) = mean(abs(s(:)));
        end
        
        % Transfer STFT features back to GPU
        stft_features_mean = gpuArray(single(stft_features_mean_cpu)); % [batch_size x 1]
    catch ME
        fprintf('STFT Feature Extraction Error: %s\n', ME.message);
        stft_features_mean = gpuArray.zeros(size(complex_signals,1),1,'single'); % Fallback to zeros if STFT fails
    end
    
    % Compile Features
    features = [mean_amp, max_amp, std_amp, spectral_centroid, mean_inst_freq, ...
                spectral_entropy, spectral_flatness, spectral_rolloff, ...
                signal_energy, max_autocorr, zero_crossing_rate, stft_features_mean];
end

%% Single Signal Analysis and Label Extraction
try
    fprintf('Performing single signal analysis...\n');
    % Select a random sample
    sample_index = randi(num_samples);
    
    % Extract and gather the sample signal
    batch_signal_gpu = X_train_gpu(sample_index, :, :);
    signal = gather(squeeze(batch_signal_gpu)); % [1024 x 2]
    
    % Extract the label
    label = Y_train(sample_index, :); % [1 x 23]
    
    signal_info = lbl_train(sample_index, :); % Assuming lbl_train is on CPU
    
    % Time Vector
    time_vector = (0:signal_length-1)' / 100e6; % Assuming sampling frequency of 100 MHz
    
    % Number of Samples to Plot
    num_samples_to_plot = min(200, signal_length);
    
    % Create Figure for Single Signal
    figure('Name', 'Single Signal Analysis', 'NumberTitle', 'off');
    
    % Time Domain Plot
    subplot(2, 3, 1);
    plot(time_vector(1:num_samples_to_plot), signal(1:num_samples_to_plot,1), 'b', ...
         time_vector(1:num_samples_to_plot), signal(1:num_samples_to_plot,2), 'r');
    title('Time Domain (First 200 Samples)');
    legend('I', 'Q');
    xlabel('Time (s)');
    ylabel('Amplitude');
    grid on;
    
    % Frequency Domain Plot
    subplot(2, 3, 2);
    freq_domain = abs(fftshift(fft(complex(signal(:,1), signal(:,2)))));
    freq_axis = linspace(-50e6, 50e6, length(freq_domain)); % Assuming f = 100 MHz
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
    spectrogram(complex(signal(:,1), signal(:,2)), hamming(128), 120, 256, 100e6, 'yaxis');
    title('Spectrogram');
    
    % Instantaneous Frequency Plot
    subplot(2, 3, 5);
    [inst_freq, time_inst] = instfreq(complex(signal(:,1), signal(:,2)), 100e6);
    plot(time_inst, inst_freq);
    title('Instantaneous Frequency');
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    grid on;
    
    % Feature Extraction for the single sample
    single_features_gpu = process_multiple_signals_gpu(X_train_gpu(sample_index, :, :), signal_length, num_features, freq_range_gpu, feature_names);
    single_features = gather(single_features_gpu); % [1 x num_features]
    
    % Feature Display
    subplot(2, 3, 6);
    bar(single_features);
    set(gca, 'XTickLabel', feature_names, 'XTick',1:num_features);
    xtickangle(45);
    title('Extracted Features');
    ylabel('Feature Value');
    
    % Display Single Signal Features
    [~, signal_label] = max(label, [], 2); % Assuming one-hot encoding
    fprintf('\nSingle Signal Analysis (Index: %d):\n', sample_index);
    for f = 1:num_features
        fprintf('%s: %.4f\n', feature_names{f}, single_features(f));
    end
    fprintf('Signal Class: %d\n', signal_label);
    fprintf('SNR: %.2f dB\n', signal_info(2));
catch ME
    fprintf('Single Signal Analysis and Label Extraction Error: %s\n', ME.message);
end

%% Main Processing Loop
tic;
for i = 1:num_batches
    start_idx = (i-1)*batch_size_main + 1;
    end_idx = min(i*batch_size_main, num_samples);
    current_batch_size = end_idx - start_idx + 1;
    
    % Extract current batch
    batch_signals = X_train_gpu(start_idx:end_idx, :, :); % [batch_size x 1024 x 2]
    
    % Process batch
    batch_features = process_multiple_signals_gpu(batch_signals, signal_length, num_features, freq_range_gpu, feature_names); % [batch_size x num_features]
    
    % Store features
    features_matrix_gpu(start_idx:end_idx, 1:size(batch_features,2)) = batch_features;
    
    fprintf('Processed batch %d of %d.\n', i, num_batches);
end
toc;
fprintf('Feature extraction completed.\n');

%% Transfer Features to CPU
try
    fprintf('Transferring features to CPU...\n');
    features_matrix = gather(features_matrix_gpu); % [469200 x num_features]
    clear features_matrix_gpu; % Only clear features_matrix_gpu
    fprintf('Features transferred to CPU.\n');
catch ME
    fprintf('Data Gathering Error: %s\n', ME.message);
    return;
end

try
    fprintf('Performing PCA for dimensionality reduction...\n');
    
    % Estimate the rank of the data matrix
    tol = 1e-6;  % Tolerance for singular values
    s = svd(features_matrix);
    rank_estimate = sum(s > tol * s(1));
    
    % Perform PCA with the estimated rank
    [coeff, score, ~, ~, explained] = pca(features_matrix, 'NumComponents', rank_estimate);
    
    % Determine the number of components to keep
    num_pca_components = min(3, rank_estimate);
    reduced_features = score(:, 1:num_pca_components);
    
    fprintf('PCA completed. Using %d components. Explained variance: %.2f%%\n', ...
            num_pca_components, sum(explained(1:num_pca_components)));
catch ME
    fprintf('PCA Error: %s\n', ME.message);
    reduced_features = [];
end
%% Anomaly Detection using Isolation Forest
try
    fprintf('Performing anomaly detection using Isolation Forest...\n');
    
    % Define contamination factor (proportion of anomalies expected)
    contamination = 0.1; % Adjust as needed
    
    % Create the Isolation Forest model
    iforest = isolationForest('ContaminationFraction', contamination, 'NumLearners', 100);
    
    % Fit the model and compute anomaly scores
    [~, anomaly_scores] = iforest.fit(features_matrix);
    
    % Determine the anomaly threshold
    anomaly_threshold = iforest.threshold;
    
    % Convert scores to anomaly flags (1 for anomaly, 0 for normal)
    anomaly_flags = anomaly_scores > anomaly_threshold;
    
    fprintf('Anomaly detection completed. Found %d anomalies (%.2f%% of the data).\n', ...
            sum(anomaly_flags), 100*sum(anomaly_flags)/numel(anomaly_flags));
catch ME
    fprintf('Anomaly Detection Error: %s\n', ME.message);
    anomaly_flags = [];
    anomaly_scores = [];
end

%% Train a Deep Neural Network for Signal Classification
try
    fprintf('Training a deep neural network for signal classification...\n');
    % Convert labels to categorical
    [~, true_labels_numeric] = max(Y_train, [], 2); % [469200 x 1]
    true_labels = categorical(true_labels_numeric);
    
    % Split data into training and validation sets
    validation_fraction = 0.2;
    num_val = floor(validation_fraction * num_samples);
    rng(1961); % For reproducibility
    idx = randperm(num_samples);
    X_train_nn = features_matrix(idx(1:end-num_val), :);
    Y_train_nn = true_labels(idx(1:end-num_val));
    X_val_nn = features_matrix(idx(end-num_val+1:end), :);
    Y_val_nn = true_labels(idx(end-num_val+1:end));
    
    % Define the network architecture
    layers = [
        featureInputLayer(size(features_matrix,2), 'Normalization', 'zscore')
        fullyConnectedLayer(64)
        reluLayer
        dropoutLayer(0.3)
        fullyConnectedLayer(32)
        reluLayer
        dropoutLayer(0.3)
        fullyConnectedLayer(23) % Number of classes
        softmaxLayer
        classificationLayer
    ];
    
    % Training options
    options = trainingOptions('adam', ...
        'MaxEpochs', 20, ...
        'MiniBatchSize', 1024, ...
        'ValidationData', {X_val_nn, Y_val_nn}, ...
        'ValidationFrequency', 30, ...
        'Verbose', true, ...
        'Plots', 'training-progress');
    
    % Train the network
    net = trainNetwork(X_train_nn, Y_train_nn, layers, options);
    fprintf('Neural network training completed.\n');
    
    % Save the trained network
    save('trainedSignalNet.mat', 'net');
catch ME
    fprintf('Neural Network Training Error: %s\n', ME.message);
end

%% Multi-Signal Analysis Visualization
try
    fprintf('Creating multi-signal analysis visualizations...\n');
    figure('Name', 'Multi-Signal Analysis', 'NumberTitle', 'off');
    
    % 3D Scatter Plot of PCA Components
    subplot(2, 3, 1);
    true_labels_numeric = double(true_labels);
    if size(reduced_features,2) >= 3
        scatter3(reduced_features(:,1), reduced_features(:,2), reduced_features(:,3), 20, true_labels_numeric, 'filled');
        xlabel('PC1'); ylabel('PC2'); zlabel('PC3');
    elseif size(reduced_features,2) == 2
        scatter(reduced_features(:,1), reduced_features(:,2), 20, true_labels_numeric, 'filled');
        xlabel('PC1'); ylabel('PC2');
    else
        scatter(reduced_features(:,1), zeros(num_samples,1), 20, true_labels_numeric, 'filled');
        xlabel('PC1'); ylabel('PC2 (Not Available)');
    end
    title('PCA of Signal Features');
    grid on;
    colorbar;
    colormap jet;
    
    % Anomaly Scores Distribution
    subplot(2, 3, 2);
    if ~isempty(anomaly_scores)
        histogram(anomaly_scores, 'BinMethod', 'auto');
        title('Anomaly Scores Distribution');
        xlabel('Anomaly Score');
        ylabel('Frequency');
        grid on;
    else
        title('Anomaly Scores Distribution');
        text(0.5, 0.5, 'Anomaly Scores Not Available', 'HorizontalAlignment', 'center');
    end
    
    % Feature Correlation Heatmap
    subplot(2, 3, 3);
    correlation_matrix = corr(features_matrix);
    heatmap(correlation_matrix, 'Title', 'Feature Correlation', 'XDisplayLabels', feature_names, 'YDisplayLabels', feature_names);
    
    % Spectral Entropy Histogram
    subplot(2, 3, 4);
    histogram(features_matrix(:,6), 'BinMethod', 'auto');
    title('Histogram of Spectral Entropy');
    xlabel('Spectral Entropy');
    ylabel('Frequency');
    grid on;
    
    % Zero-Crossing Rate Histogram
    subplot(2, 3, 5);
    histogram(features_matrix(:,11), 'BinMethod', 'auto');
    title('Histogram of Zero-Crossing Rate');
    xlabel('Zero-Crossing Rate');
    ylabel('Frequency');
    grid on;
    
    % Anomaly Highlight in PCA Plot
    subplot(2, 3, 6);
    if ~isempty(anomaly_flags) && ~isempty(reduced_features)
        scatter3(reduced_features(:,1), reduced_features(:,2), reduced_features(:,3), 10, anomaly_flags, 'filled');
        xlabel('PC1'); ylabel('PC2'); zlabel('PC3');
        title('Anomalies Highlighted in PCA Space');
        colormap(gca, [0 0 1; 1 0 0]); % Blue for normal, Red for anomalies
        colorbar('Ticks', [0, 1], 'TickLabels', {'Normal', 'Anomaly'});
    else
        title('Anomalies Highlighted in PCA Space');
        text(0.5, 0.5, 'Anomalies Not Detected', 'HorizontalAlignment', 'center');
    end
    
   % Display Multi-Signal Analysis Results
    fprintf('\nMulti-Signal Analysis:\n');
    fprintf('Number of samples analyzed: %d\n', num_samples);
    fprintf('Number of features extracted per signal: %d\n', num_features);
    if ~isempty(anomaly_scores)
        fprintf('Mean Anomaly Score: %.4f\n', mean(anomaly_scores));
        fprintf('Number of detected anomalies: %d\n', sum(anomaly_flags));
    end
    
    if ~isempty(anomaly_scores)
        % Identify Top 5 Anomalies
        [sorted_scores, anomaly_indices_sorted] = sort(anomaly_scores, 'descend');
        top_anomalies = anomaly_indices_sorted(1:min(5, length(anomaly_indices_sorted)));
        fprintf('Top 5 anomalous signal indices: %s\n', mat2str(top_anomalies));
        
        % Display Top 5 Anomalous Signals
        for idx = 1:length(top_anomalies)
            fprintf('Anomaly %d: Signal Index %d with Score %.4f\n', idx, top_anomalies(idx), sorted_scores(idx));
        end
    end
catch ME
    fprintf('Multi-Signal Analysis Visualization Error: %s\n', ME.message);
end

%% Save Processed Data and Results
try
    fprintf('Saving processed data and results...\n');
    % Prepare variables to save
    save_vars = {'features_matrix', 'reduced_features', 'anomaly_scores', 'anomaly_flags'};
    % Check if anomaly_scores and anomaly_flags exist
    if ~exist('anomaly_scores', 'var') || isempty(anomaly_scores)
        save_vars = setdiff(save_vars, {'anomaly_scores', 'anomaly_flags'});
    end
    save('processedFeatures.mat', save_vars{:});
    fprintf('All data and results saved successfully.\n');
catch ME
    fprintf('Saving Error: %s\n', ME.message);
end

%% Cleanup GPU
reset(gpu_dev);
fprintf('GPU resources have been released.\n');

%% Conclusion
fprintf('Script execution completed successfully.\n');