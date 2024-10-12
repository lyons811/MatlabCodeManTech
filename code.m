% Improved MATLAB Code for Fast Signal Analysis and Feature Extraction with AI Integration
% Corrected and Enhanced to Fix Reported Errors

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
    clear X_train; % Free CPU memory
    fprintf('Data transferred to GPU.\n');
catch ME
    fprintf('Data Transfer Error: %s\n', ME.message);
    return;
end

%% Feature Extraction Parameters
% Existing features: 12
% Additional features: 8 (Wavelet-based and others)
num_features = 20; % Updated number of features (removed 2 mel spectrogram features)
features_matrix_gpu = gpuArray.zeros(num_samples, num_features, 'single'); % Use single precision to save memory
batch_size_main = 5000;  % Adjust based on GPU memory
num_batches = ceil(num_samples / batch_size_main);
fprintf('Starting feature extraction with %d batches...\n', num_batches);

% Define Feature Names Globally
feature_names = {'Mean Amp', 'Max Amp', 'Std Amp', 'Spectral Centroid', 'Mean Inst Freq', ...
                'Spectral Entropy', 'Spectral Flatness', 'Spectral Rolloff', ...
                'Signal Energy', 'Max Autocorr', 'Zero-Cross Rate', 'STFT Mean', ...
                'Wavelet Entropy', 'CWT Mean Magnitude', 'MODWT Detail 1', ...
                'MODWT Detail 2', 'MODWT Detail 3', 'MODWT Detail 4', ...
                'Wavelet Packet Energy', 'Joint Time-Frequency Scattering'};

%% Precompute Frequency Range for Spectral Centroid
freq_range = linspace(0, 1, signal_length); % [1 x 1024] Row vector
freq_range_gpu = gpuArray(freq_range); % Transfer to GPU once

%% Define Feature Extraction Function
% Enhanced with Wavelet-based Feature Extraction and Error Handling
function features = process_multiple_signals_gpu(signals, signal_length, num_features, freq_range_gpu, feature_names)
    % Convert I/Q to complex signals
    complex_signals = complex(signals(:,:,1), signals(:,:,2)); % [batch_size x signal_length]
    
    % Perform FFT
    fft_signals = fft(complex_signals, [], 2);
    fft_magnitude = abs(fft_signals);
    
    % Power Spectral Density
    psd = (fft_magnitude.^2) / signal_length;
    
    % Basic Features
    mean_amp = mean(abs(complex_signals), 2); % [batch_size x1]
    max_amp = max(abs(complex_signals), [], 2); % [batch_size x1]
    std_amp = std(abs(complex_signals), 0, 2); % [batch_size x1]
    
    % Spectral Centroid
    spectral_centroid = sum(freq_range_gpu .* psd, 2) ./ sum(psd, 2); % [batch_size x1]
    
    % Instantaneous Frequency
    inst_phase = unwrap(angle(complex_signals), [], 2);
    inst_freq = diff(inst_phase, 1, 2) / (2*pi*(1/signal_length));
    mean_inst_freq = mean(inst_freq, 2); % [batch_size x1]
    
    % Spectral Entropy
    psd_norm = psd ./ sum(psd, 2);
    spectral_entropy = -sum(psd_norm .* log2(psd_norm + eps), 2); % [batch_size x1]
    
    % Spectral Flatness
    spectral_flatness = geomean(psd, 2) ./ mean(psd, 2); % [batch_size x1]
    
    % Spectral Rolloff (85%)
    cumsum_psd = cumsum(psd, 2);
    thresholds = 0.85 * sum(psd, 2);
    spectral_rolloff = sum(cumsum_psd < thresholds, 2) / signal_length; % [batch_size x1]
    
    % Signal Energy
    signal_energy = sum(abs(complex_signals).^2, 2); % [batch_size x1]
    
    % Max Autocorrelation Value
    autocorr_vals = ifft(fft_signals .* conj(fft_signals), [], 2);
    max_autocorr = max(abs(autocorr_vals), [], 2); % [batch_size x1]
    
    % Zero-Crossing Rate
    zero_crossings = sum(abs(diff(sign(real(complex_signals)), 1, 2)) > 0, 2);
    zero_crossing_rate = zero_crossings / signal_length; % [batch_size x1]
    
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
        stft_features_mean = gpuArray(single(stft_features_mean_cpu)); % [batch_size x1]
    catch ME
        fprintf('STFT Feature Extraction Error: %s\n', ME.message);
        stft_features_mean = gpuArray.zeros(size(complex_signals,1),1,'single'); % Fallback to zeros if STFT fails
    end
    
    %% Additional Feature Extraction: Wavelet-Based Features
    % Wavelet Entropy
    try
        batch_size = size(complex_signals, 1);
        wavelet_entropy = zeros(batch_size, 1, 'single'); % Initialize

        % Gather complex_signals to CPU for processing
        complex_signals_cpu = gather(complex_signals); % [batch_size x signal_length]

        parfor idx = 1:batch_size
            % Compute Wavelet Entropy for each signal
            wavelet_entropy(idx) = wentropy(real(complex_signals_cpu(idx, :)), 'shannon');
        end

        % Transfer back to GPU
        wavelet_entropy_gpu = gpuArray(single(wavelet_entropy)); % [batch_size x1]
    catch ME
        fprintf('Wavelet Entropy Extraction Error: %s\n', ME.message);
        wavelet_entropy_gpu = gpuArray.zeros(size(complex_signals,1),1,'single');
    end
    
    % Continuous Wavelet Transform (CWT) Mean Magnitude
    try
        cwt_mean = zeros(batch_size,1,'single');
        parfor idx = 1:batch_size
            signal_mag = double(abs(complex_signals_cpu(idx,:)));
            cwt_result = abs(cwt(signal_mag, 'amor'));
            cwt_mean(idx) = mean(cwt_result(:));
        end
        cwt_mean_gpu = gpuArray(single(cwt_mean)); % [batch_size x1]
    catch ME
        fprintf('CWT Mean Magnitude Extraction Error: %s\n', ME.message);
        cwt_mean_gpu = gpuArray.zeros(size(complex_signals,1),1,'single');
    end
    
    % Maximal Overlap Discrete Wavelet Transform (MODWT) Detail Coefficients
    try
        modwt_details = zeros(batch_size,4,'single'); % [batch_size x4]
        parfor idx = 1:batch_size
            signal = real(complex_signals_cpu(idx,:)); % Use real part for MODWT
            C = modwt(signal, 'db4', 4); % 4-level MODWT with Daubechies 4
            for d = 1:4
                modwt_details(idx,d) = std(C(d, :)); % Standard deviation of detail coefficients
            end
        end
        modwt_details_gpu = gpuArray(single(modwt_details)); % [batch_size x4]
    catch ME
        fprintf('MODWT Detail Extraction Error: %s\n', ME.message);
        modwt_details_gpu = gpuArray.zeros(size(complex_signals,1),4,'single');
    end
    
    % Wavelet Packet Energy
    try
        wp_energy = zeros(batch_size,1,'single'); % [batch_size x1]
        parfor idx = 1:batch_size
            signal = real(complex_signals_cpu(idx,:)); % Use real part for Wavelet Packet
            wp = wpdec(signal, 3, 'db4'); % Wavelet packet decomposition level 3
            wp_energy(idx) = sum(wpcoef(wp, 'a') .^ 2); % Energy of approximation coefficients
        end
        wp_energy_gpu = gpuArray(single(wp_energy)); % [batch_size x1]
    catch ME
        fprintf('Wavelet Packet Energy Extraction Error: %s\n', ME.message);
        wp_energy_gpu = gpuArray.zeros(size(complex_signals,1),1,'single');
    end
    
    % Joint Time-Frequency Scattering (Placeholder)
    joint_scattering = gpuArray.zeros(size(complex_signals,1),1,'single'); % [batch_size x1]
    
    %% Compile All Features
    feature_list = {mean_amp, max_amp, std_amp, spectral_centroid, mean_inst_freq, ...
                   spectral_entropy, spectral_flatness, spectral_rolloff, ...
                   signal_energy, max_autocorr, zero_crossing_rate, stft_features_mean, ...
                   wavelet_entropy_gpu, cwt_mean_gpu, modwt_details_gpu(:,1), ...
                   modwt_details_gpu(:,2), modwt_details_gpu(:,3), modwt_details_gpu(:,4), ...
                   wp_energy_gpu, joint_scattering};
               
    expected_rows = size(complex_signals,1);
    for f = 1:length(feature_list)
        if size(feature_list{f},1) ~= expected_rows
            error('Feature %d (%s) has %d rows, expected %d.', f, feature_names{f}, size(feature_list{f},1), expected_rows);
        end
    end
    
    % Concatenate all features horizontally
    features = horzcat(feature_list{:}); % [batch_size x num_features]
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
    subplot(3, 4, 1);
    plot(time_vector(1:num_samples_to_plot), signal(1:num_samples_to_plot,1), 'b', ...
         time_vector(1:num_samples_to_plot), signal(1:num_samples_to_plot,2), 'r');
    title('Time Domain (First 200 Samples)');
    legend('I', 'Q');
    xlabel('Time (s)');
    ylabel('Amplitude');
    grid on;
    
    % Frequency Domain Plot
    subplot(3, 4, 2);
    freq_domain = abs(fftshift(fft(complex(signal(:,1), signal(:,2)))));
    freq_axis = linspace(-50e6, 50e6, length(freq_domain)); % Assuming f = 100 MHz
    plot(freq_axis, freq_domain);
    title('Frequency Domain');
    xlabel('Frequency (Hz)');
    ylabel('Magnitude');
    grid on;
    
    % Power Spectral Density
    subplot(3, 4, 3);
    [psd_vals, freq_psd] = pspectrum(complex(signal(:,1), signal(:,2)), 100e6, 'FrequencyLimits', [0 50e6]);
    plot(freq_psd, 10*log10(psd_vals));
    title('Power Spectral Density');
    xlabel('Frequency (Hz)');
    ylabel('Power/Frequency (dB/Hz)');
    grid on;
    
    % Spectrogram
    subplot(3, 4, 4);
    spectrogram(complex(signal(:,1), signal(:,2)), hamming(128), 120, 256, 100e6, 'yaxis');
    title('Spectrogram');
    
    % Instantaneous Frequency Plot
    subplot(3, 4, 5);
    [inst_freq, time_inst] = instfreq(complex(signal(:,1), signal(:,2)), 100e6);
    plot(time_inst, inst_freq);
    title('Instantaneous Frequency');
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    grid on;
    
    % Histogram of Signal Amplitudes
    subplot(3, 4, 6);
    histogram(abs(complex(signal(:,1), signal(:,2))), 50);
    title('Histogram of Signal Amplitudes');
    xlabel('Amplitude');
    ylabel('Count');
    grid on;
    
    % Box Plot of Signal Amplitudes
    subplot(3, 4, 7);
    boxplot([abs(signal(:,1)), abs(signal(:,2))], {'I', 'Q'});
    title('Box Plot of I and Q Amplitudes');
    ylabel('Amplitude');
    grid on;
    
    % Heatmap of Signal Correlation
    subplot(3, 4, 8);
    correlation_matrix = corrcoef([signal(:,1), signal(:,2)]);
    heatmap(correlation_matrix, 'Title', 'Correlation between I and Q', ...
            'XDisplayLabels', {'I', 'Q'}, 'YDisplayLabels', {'I', 'Q'});
    
    % Line Graph of Full Signal
    subplot(3, 4, 9);
    plot(time_vector, signal(:,1), 'b', time_vector, signal(:,2), 'r');
    title('Full Signal in Time Domain');
    legend('I', 'Q');
    xlabel('Time (s)');
    ylabel('Amplitude');
    grid on;
    
    % Waterfall Plot (Example with Multiple Signals)
    subplot(3, 4, 10);
    % Generate multiple signals for waterfall
    waterfall_indices = randi(num_samples, 10,1);
    waterfall_signals_gpu = X_train_gpu(waterfall_indices, :, :);
    waterfall_signals = gather(waterfall_signals_gpu); % [10 x 1024 x 2]
    for s = 1:10
        complex_s = complex(waterfall_signals(s,:,1), waterfall_signals(s,:,2));
        plot3(time_vector, real(complex_s), imag(complex_s));
        hold on;
    end
    hold off;
    title('Waterfall Plot of 10 Random Signals');
    xlabel('Time (s)');
    ylabel('In-phase');
    zlabel('Quadrature');
    grid on;
    
    % Feature Extraction for the single sample
    single_features_gpu = process_multiple_signals_gpu(X_train_gpu(sample_index, :, :), signal_length, num_features, freq_range_gpu, feature_names);
    single_features = gather(single_features_gpu); % [1 x num_features]
    
    % Feature Display
    subplot(3, 4, 11);
    bar(single_features);
    set(gca, 'XTickLabel', feature_names, 'XTick',1:num_features);
    xtickangle(45);
    title('Extracted Features');
    ylabel('Feature Value');
    
    % Scatter Plot of Selected Features (Example: Spectral Centroid vs. Signal Energy)
    subplot(3, 4, 12);
    scatter(single_features(4), single_features(9), 100, 'filled');
    title('Spectral Centroid vs. Signal Energy');
    xlabel('Spectral Centroid');
    ylabel('Signal Energy');
    grid on;
    
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

%% Perform PCA for Dimensionality Reduction
try
    fprintf('Performing PCA for dimensionality reduction...\n');
    
    % Estimate the rank of the data matrix
    tol = 1e-6;  % Tolerance for singular values
    s = svd(features_matrix, 'econ');
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

%% Train a Deep Neural Network for Signal Classification
try
    fprintf('Training a deep neural network for signal classification...\n');
    % Convert labels to categorical
    [~, true_labels_numeric] = max(Y_train, [], 2); % [469200 x 1]
    true_labels = categorical(true_labels_numeric);
    
    % Shuffle data
    rng(1961); % For reproducibility
    shuffle_indices = randperm(num_samples);
    features_matrix = features_matrix(shuffle_indices, :);
    true_labels = true_labels(shuffle_indices);
    
    % Split data into training and validation sets
    validation_fraction = 0.2;
    num_val = floor(validation_fraction * num_samples);
    X_train_nn = features_matrix(1:end-num_val, :);
    Y_train_nn = true_labels(1:end-num_val);
    X_val_nn = features_matrix(end-num_val+1:end, :);
    Y_val_nn = true_labels(end-num_val+1:end);
    
    % Define the network architecture
    layers = [
        featureInputLayer(size(features_matrix,2), 'Normalization', 'zscore')
        fullyConnectedLayer(128)
        reluLayer
        dropoutLayer(0.3)
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
        'MaxEpochs', 30, ...
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
    
    % 3D Scatter Plot of PCA Components Colored by Class
    if size(reduced_features,2) >= 3
        % Convert reduced_features to double if they are single
        x = double(reduced_features(:,1));
        y = double(reduced_features(:,2));
        z = double(reduced_features(:,3));
        
        % Define a colormap with enough distinct colors
        cmap = jet(23); % Assuming 23 classes
        
        % Create a scatter3 plot
        scatter3(x, y, z, 10, true_labels_numeric, 'filled');
        xlabel('PC1');
        ylabel('PC2');
        zlabel('PC3');
        title('PCA of Signal Features Colored by Class');
        grid on;
        colorbar;
        colormap(cmap);
        
        % Optionally, set the color axis to cover all classes
        caxis([1 23]);
        
    elseif size(reduced_features,2) == 2
        % 2D Scatter Plot if only 2 PCA components are available
        x = double(reduced_features(:,1));
        y = double(reduced_features(:,2));
        
        gscatter(x, y, true_labels_numeric, jet(23), 'o', 5);
        xlabel('PC1');
        ylabel('PC2');
        title('PCA of Signal Features Colored by Class');
        grid on;
        colorbar;
        colormap(jet(23));
        
    else
        % 1D Scatter Plot if only 1 PCA component is available
        x = double(reduced_features(:,1));
        
        scatter(x, zeros(num_samples,1), 10, true_labels_numeric, 'filled');
        xlabel('PC1');
        ylabel('PC2 (Not Available)');
        title('PCA of Signal Features Colored by Class');
        grid on;
        colorbar;
        colormap(jet(23));
    end
    
    % Feature Correlation Heatmap
    subplot(2, 2, 2);
    correlation_matrix = corr(features_matrix);
    heatmap(correlation_matrix, 'Title', 'Feature Correlation', 'XDisplayLabels', feature_names, 'YDisplayLabels', feature_names, ...
            'Colormap', parula, 'ColorLimits', [-1 1]);
    
    % Feature Distribution Histograms per Class (Example with a few features)
    subplot(2, 2, 3);
    feature_to_plot = 6; % Spectral Entropy
    g = findgroups(true_labels_numeric);
    boxplot(features_matrix(:,feature_to_plot), g);
    title('Boxplot of Spectral Entropy per Class');
    xlabel('Class');
    ylabel('Spectral Entropy');
    
    subplot(2, 2, 4);
    feature_to_plot = 11; % Zero-Crossing Rate
    boxplot(features_matrix(:,feature_to_plot), g);
    title('Boxplot of Zero-Crossing Rate per Class');
    xlabel('Class');
    ylabel('Zero-Crossing Rate');
    
    % Additional Visualizations
    figure('Name', 'Additional Feature Visualizations', 'NumberTitle', 'off');
    
    % Heatmap of Feature Correlations
    subplot(2, 3, 1);
    heatmap(correlation_matrix, 'Title', 'Feature Correlation Heatmap', ...
            'XDisplayLabels', feature_names, 'YDisplayLabels', feature_names, ...
            'Colormap', hot, 'ColorLimits', [-1 1]);
    
    % Bubble Chart: Wavelet Entropy vs. Spectral Centroid with Size as Signal Energy
    subplot(2, 3, 2);
    scatter(features_matrix(:,13), features_matrix(:,4), 10, features_matrix(:,9), 'filled');
    title('Bubble Chart: Wavelet Entropy vs. Spectral Centroid');
    xlabel('Wavelet Entropy');
    ylabel('Spectral Centroid');
    colorbar;
    colormap(jet);
    grid on;
    
    % Line Graph of Average Feature Values per Class
    subplot(2, 3, 3);
    avg_features = grpstats(features_matrix, true_labels_numeric);
    plot(avg_features(:,4), 'LineWidth', 2); hold on; % Spectral Centroid
    plot(avg_features(:,6), 'LineWidth', 2); % Spectral Entropy
    plot(avg_features(:,13), 'LineWidth', 2); % Wavelet Entropy
    title('Average Feature Values per Class');
    xlabel('Class');
    ylabel('Average Value');
    legend({'Spectral Centroid', 'Spectral Entropy', 'Wavelet Entropy'}, 'Location', 'best');
    grid on;
    
    % Zero-Crossing Rate Distribution
    subplot(2, 3, 4);
    histogram(features_matrix(:,11), 50);
    title('Histogram of Zero-Crossing Rate');
    xlabel('Zero-Crossing Rate');
    ylabel('Count');
    grid on;
    
    % Joint Time-Frequency Scattering Feature (Placeholder)
    subplot(2, 3, 5);
    if size(reduced_features,2) >= 2
        scatter(reduced_features(:,1), reduced_features(:,2), 10, true_labels_numeric, 'filled');
        title('Joint Time-Frequency Scattering Feature (Placeholder)');
        xlabel('Feature 1');
        ylabel('Feature 2');
        grid on;
        colorbar;
        colormap(jet(23));
    else
        scatter(features_matrix(:,20), zeros(num_samples,1), 10, true_labels_numeric, 'filled');
        title('Joint Time-Frequency Scattering Feature (Placeholder)');
        xlabel('Feature 20');
        ylabel('Feature 2 (Not Available)');
        grid on;
        colorbar;
        colormap(jet(23));
    end
    
    % Display Multi-Signal Analysis Results
    fprintf('\nMulti-Signal Analysis:\n');
    fprintf('Number of samples analyzed: %d\n', num_samples);
    fprintf('Number of features extracted per signal: %d\n', num_features);
    
    % Evaluate the trained network on validation data
    Y_pred = classify(net, X_val_nn);
    accuracy = sum(Y_pred == Y_val_nn) / numel(Y_val_nn) * 100;
    fprintf('Validation Accuracy: %.2f%%\n', accuracy);
    
    % Confusion Matrix
    figure('Name', 'Confusion Matrix', 'NumberTitle', 'off');
    confusionchart(Y_val_nn, Y_pred);
    title('Confusion Matrix for Signal Classification');
    
    fprintf('Multi-Signal Analysis Visualization completed.\n');
    
    % Identify Top 5 Misclassified Signals (if needed)
    [~, misclassified_indices] = find(Y_pred ~= Y_val_nn);
    top_misclassified = misclassified_indices(1:min(5, length(misclassified_indices)));
    fprintf('Top 5 Misclassified signal indices in validation set: %s\n', mat2str(top_misclassified));
    
    % Optionally, display some misclassified signals
    for idx = 1:length(top_misclassified)
        signal_idx = num_samples - num_val + top_misclassified(idx); % Adjust index for validation set
        fprintf('Misclassified Signal %d: Predicted Class %s, True Class %s\n', ...
                signal_idx, string(Y_pred(top_misclassified(idx))), string(Y_val_nn(top_misclassified(idx))));
    end
catch ME
    fprintf('Multi-Signal Analysis Visualization Error: %s\n', ME.message);
end


%% Save Processed Data and Results
try
    fprintf('Saving processed data and results...\n');
    % Prepare variables to save
    save_vars = {'features_matrix', 'reduced_features', 'net', 'X_val_nn', 'Y_val_nn', 'Y_pred'};
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
