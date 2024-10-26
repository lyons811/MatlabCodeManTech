% Initialize GPU
g = gpuDevice(1);
reset(g);
fprintf('GPU enabled for processing. Available GPU memory: %.2f GB\n', g.AvailableMemory/1e9);

% Define modulation classes for DeepSig RadioML 2018.01A
classes = {'OOK', 'ASK4', 'ASK8', 'BPSK', 'QPSK', 'PSK8', 'PSK16', 'PSK32', ...
           'APSK16', 'APSK32', 'APSK64', 'APSK128', 'QAM16', 'QAM32', 'QAM64', ...
           'QAM128', 'QAM256', 'AM_SSB_WC', 'AM_SSB_SC', 'AM_DSB_WC', ...
           'AM_DSB_SC', 'FM', 'GMSK', 'OQPSK'};

try
    % Open the HDF5 file
    filename = 'GOLD_XYZ_OSC.0001_1024.hdf5';
    
    % Verify file access
    if ~exist(filename, 'file')
        error('File not found: %s', filename);
    end
    fprintf('File exists and is accessible.\n');
    
    % Open file with H5F
    file_id = H5F.open(filename, 'H5F_ACC_RDONLY', 'H5P_DEFAULT');
    fprintf('Successfully opened HDF5 file.\n');
    
    % Set number of samples to read
    num_samples = 1000;
    
    % Read X dataset
    dset_x_id = H5D.open(file_id, '/X');
    fprintf('Successfully opened X dataset.\n');
    
    % Get X dataset space and dimensions
    space_id = H5D.get_space(dset_x_id);
    [~, dims] = H5S.get_simple_extent_dims(space_id);
    fprintf('Original X dataset dimensions: %s\n', mat2str(dims));
    
    % Define the hyperslab selection for X
    start = [0 0 0];  % Start from beginning
    count = [num_samples 1024 2];  % Read num_samples frames
    
    % Select hyperslab in the file
    H5S.select_hyperslab(space_id, 'H5S_SELECT_SET', start, [], [], count);
    
    % Create memory space for X
    memspace = H5S.create_simple(3, count, []);
    
    % Read X data
    fprintf('Reading X data...\n');
    X_data = H5D.read(dset_x_id, 'H5ML_DEFAULT', memspace, space_id, 'H5P_DEFAULT');
    X_data = permute(reshape(X_data, count), [3 2 1]); % Reshape to [2 1024 num_samples]
    fprintf('X data successfully read. Size: %s\n', mat2str(size(X_data)));
    
    % Read Y dataset (one-hot encoded labels)
    fprintf('\nReading Y dataset...\n');
    dset_y_id = H5D.open(file_id, '/Y');
    space_y_id = H5D.get_space(dset_y_id);
    [~, dims_y] = H5S.get_simple_extent_dims(space_y_id);
    fprintf('Y dataset dimensions: %s\n', mat2str(dims_y));
    
    % Define hyperslab for Y
    start_y = [0 0];  % Start from beginning
    count_y = [num_samples 24];  % Read num_samples frames, all 24 modulation types
    
    % Create memory space for Y
    memspace_y = H5S.create_simple(2, count_y, []);
    
    % Select hyperslab for Y
    H5S.select_hyperslab(space_y_id, 'H5S_SELECT_SET', start_y, [], [], count_y);
    
    % Read Y data
    Y_data = H5D.read(dset_y_id, 'H5ML_DEFAULT', memspace_y, space_y_id, 'H5P_DEFAULT');
    fprintf('Y data successfully read. Size: %s\n', mat2str(size(Y_data)));
    
    % Read Z dataset (SNR values)
    fprintf('\nReading Z dataset...\n');
    dset_z_id = H5D.open(file_id, '/Z');
    space_z_id = H5D.get_space(dset_z_id);
    [~, dims_z] = H5S.get_simple_extent_dims(space_z_id);
    fprintf('Z dataset dimensions: %s\n', mat2str(dims_z));

    % Define hyperslab for Z with correct dimensionality
    start_z = [0, 0];         % Start from beginning in both dimensions
    count_z = [num_samples, 1]; % Read num_samples rows and 1 column

    % Create memory space for Z
    memspace_z = H5S.create_simple(2, count_z, []);

    % Select hyperslab for Z
    H5S.select_hyperslab(space_z_id, 'H5S_SELECT_SET', start_z, [], [], count_z);

    % Read Z data
    Z_data = H5D.read(dset_z_id, 'H5ML_DEFAULT', memspace_z, space_z_id, 'H5P_DEFAULT');
    Z_data = reshape(Z_data, [num_samples, 1]); % Reshape to [num_samples, 1]
    fprintf('Z data successfully read. Size: %s\n', mat2str(size(Z_data)));
    
    % Move data to GPU if needed
    X_gpu = gpuArray(X_data);
    Y_gpu = gpuArray(Y_data);
    Z_gpu = gpuArray(Z_data);

    % Get modulation type and SNR for first sample
    [~, modIndex] = max(Y_data(:,1)); % Get modulation type for first sample
    snr = Z_data(1); % Get SNR for first sample

    % Create complex signal for first frame
    complexSignal = X_data(1,:,1) + 1j*X_data(2,:,1);
    complexSignal_gpu = gpuArray(complexSignal);
    
    % Clean up HDF5 resources
    H5S.close(space_id);
    H5S.close(memspace);
    H5S.close(space_y_id);
    H5S.close(memspace_y);
    H5S.close(space_z_id);
    H5S.close(memspace_z);
    H5D.close(dset_x_id);
    H5D.close(dset_y_id);
    H5D.close(dset_z_id);
    H5F.close(file_id);
    
catch ME
    fprintf('Error: %s\n', ME.message);
    fprintf('Error details:\n');
    disp(getReport(ME, 'extended'));
    
    % Clean up HDF5 resources if they exist
    if exist('space_id', 'var'), H5S.close(space_id); end
    if exist('memspace', 'var'), H5S.close(memspace); end
    if exist('space_y_id', 'var'), H5S.close(space_y_id); end
    if exist('memspace_y', 'var'), H5S.close(memspace_y); end
    if exist('space_z_id', 'var'), H5S.close(space_z_id); end
    if exist('memspace_z', 'var'), H5S.close(memspace_z); end
    if exist('dset_x_id', 'var'), H5D.close(dset_x_id); end
    if exist('dset_y_id', 'var'), H5D.close(dset_y_id); end
    if exist('dset_z_id', 'var'), H5D.close(dset_z_id); end
    if exist('file_id', 'var'), H5F.close(file_id); end
    
    if exist('g', 'var')
        delete(g);
    end
    return;
end

% Display data summary
fprintf('\nFinal Data Summary:\n');
fprintf('X data shape: %s\n', mat2str(size(X_data)));
fprintf('Y data shape: %s\n', mat2str(size(Y_data)));
fprintf('Z data shape: %s\n', mat2str(size(Z_data)));

% Display success message
fprintf('\nScript completed successfully!\n');

% Get modulation type and SNR for selected signal
fprintf('Analyzing signal:\nModulation: %s\nSNR: %.2f dB\n', classes{modIndex}, snr);

% Find abrupt changes in signal
changePoints = findchangepts(abs(complexSignal), 'MaxNumChanges', 5);
fprintf('Detected change points at indices: %s\n', mat2str(changePoints));

% Time-frequency analysis using Spectrogram
complexSignal = complexSignal - mean(complexSignal); % Remove DC
complexSignal = complexSignal ./ max(abs(complexSignal)); % Normalize

[b, a] = butter(6, [0.1 0.4]); % Adjust frequencies based on your signal
complexSignal = filtfilt(b, a, complexSignal);

window_gpu = gpuArray(hamming(128));
[s_gpu, f, t] = spectrogram(complexSignal_gpu, window_gpu, 64, 256, 1024, 'yaxis');
s = gather(s_gpu);
fprintf('Spectrogram computed on GPU.\n');

% Compute time-frequency ridges
[fridge, iridge] = tfridge(s, f);
fprintf('Time-frequency ridges extracted.\n');

% Estimate instantaneous bandwidth
ibw = instbw(complexSignal, 1024);
fprintf('Instantaneous bandwidth estimated.\n');

% Spectral analysis
[pxx, f_pxx] = periodogram(complexSignal, [], [], 1024);
se = spectralEntropy(pxx, f_pxx);
sf = spectralFlatness(pxx, f_pxx);
sk = spectralKurtosis(pxx, f_pxx);
ss = spectralSkewness(pxx, f_pxx);
fprintf('Spectral features calculated.\n');

% Signal feature extraction
detector = signalFrequencyFeatureExtractor('SampleRate', 1024);
detector.FrameSize = 64;
detector.FrameOverlapLength = 32;
detector.WelchPSD = true;
features = detector.extract(complexSignal);
fprintf('Signal features extracted for anomaly detection.\n');

% Extract ROIs
roiLimits = [1, 256; 257, 512; 513, 768; 769, 1024];
roi = extractsigroi(abs(complexSignal), roiLimits);
fprintf('Regions of Interest (ROIs) extracted.\n');

% Create figure for plots
fig = figure('Name', 'Signal Analysis - DeepSig RadioML 2018.01A', 'Position', [100 100 1200 800]);

% Plot 1: Enhanced Spectrogram
subplot(4, 4, 1);
window_size = 256;
noverlap = 192;
nfft = 512;
fs = 1024; % Sampling frequency

% Compute spectrogram
[s, f, t] = spectrogram(complexSignal, hamming(window_size), noverlap, nfft, fs, 'yaxis');

% Convert to dB scale
s_db = 10*log10(abs(s) + eps);

% Dynamic Clipping based on data
min_db = min(s_db(:));
max_db = max(s_db(:));
clim_range = [min_db, max_db]; % Adjust as needed

% Plot using imagesc
imagesc(t, f, s_db);
axis xy;
title(sprintf('Signal Spectrogram\n%s, SNR: %.2f dB', classes{modIndex}, snr));
xlabel('Time (s)');
ylabel('Frequency (Hz)');
colormap(jet);
caxis(clim_range); % Set color limits dynamically
colorbar;

% Plot 2: PSD of Extracted Signal ROIs
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
title(sprintf('PSD of Extracted Signal ROIs\n%s', classes{modIndex}));
xlabel('Frequency (Hz)');
ylabel('Power/Frequency (dB/Hz)');
legend(arrayfun(@(x) sprintf('ROI %d', x), 1:length(roi), 'UniformOutput', false), 'Location', 'bestoutside');

% Plot 3: 3D Wavelet Scattering Transform
subplot(4, 4, 3);
[wst_real, wstInfo] = featureMatrix(sn, real(complexSignal));
[wst_imag, ~] = featureMatrix(sn, imag(complexSignal));
wst = [wst_real; wst_imag];
[X_wst, Y_wst] = meshgrid(1:size(wst,2), 1:size(wst,1));
surf(X_wst, Y_wst, wst, 'EdgeColor', 'none');
title('3D Wavelet Scattering Transform');
xlabel('Scattering Path');
ylabel('Time');
zlabel('Magnitude');
view(-45, 45);
colorbar;
lighting phong;
camlight('headlight');
material([0.7 0.9 0.3 1]);
shading interp;

% --- Replacement for Plot 4: Instantaneous Frequency (IF) Plot ---
subplot(4, 4, 4);
try
    % Define Sampling Frequency
    fs = 1024; % Hz (as per your dataset)
    dt = 1 / fs; % Sampling interval

    % Compute the instantaneous phase directly from the complex signal
    instantaneousPhase = unwrap(angle(complexSignal));

    % Compute the instantaneous frequency by differentiating the phase
    instantaneousFrequency = diff(instantaneousPhase) / (2*pi*dt); % Hz

    % Create a time vector for the instantaneous frequency
    t_if = (0:length(instantaneousFrequency)-1) * dt;

    % Plot the Instantaneous Frequency
    plot(t_if, instantaneousFrequency, 'b', 'LineWidth', 1.5);
    title('Instantaneous Frequency (IF)');
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    grid on;
    hold on;

    % Detect and mark change points in the Instantaneous Frequency
    changePoints_if = findchangepts(instantaneousFrequency, 'MaxNumChanges', 5);
    plot(t_if(changePoints_if), instantaneousFrequency(changePoints_if), 'ro', 'MarkerSize', 6, 'LineWidth', 2);
    hold off;

    % Add annotations for modulation type and SNR
    dim = [0.15 0.6 0.3 0.3]; % Position of the annotation box [x y w h]
    annotation('textbox', dim, 'String', {sprintf('Modulation: %s', classes{modIndex}), sprintf('SNR: %.2f dB', snr)}, ...
        'FitBoxToText', 'on', 'BackgroundColor', 'w', 'EdgeColor', 'k', 'FontSize', 8);

    fprintf('Instantaneous Frequency plot successfully generated.\n');

catch ME
    fprintf('Error in Instantaneous Frequency analysis: %s\n', ME.message);
    fprintf('Error details:\n%s\n', getReport(ME, 'extended'));
    subplot(4, 4, 4);
    text(0.5, 0.5, 'Error in IF Analysis', ...
        'HorizontalAlignment', 'center', 'Color', 'red', 'FontSize', 12);
    axis off;
end


% Plot 5: Instantaneous Bandwidth
subplot(4, 4, 5);
plot(ibw, 'LineWidth', 1.5);
title('Instantaneous Bandwidth');
xlabel('Sample');
ylabel('Bandwidth');

% Plot 6: 3D Spectral Features
subplot(4, 4, 6);
[X_feat, Y_feat] = meshgrid(1:size(features,2), 1:size(features,1));
surf(X_feat, Y_feat, features, 'EdgeColor', 'none');
title('3D Spectral Features');
xlabel('Feature Index');
ylabel('Time Window');
zlabel('Feature Value');
view(-45, 45);
colorbar;
lighting phong;
camlight('headlight');
colormap(jet);
shading interp;

% Plot 7: 3D STFT
subplot(4, 4, 7);
window_gpu_stft = gpuArray(hamming(128));
[S_stft_gpu, F_stft, T_stft] = stft(complexSignal_gpu, 1024, 'Window', window_gpu_stft, ...
    'OverlapLength', 64, 'FFTLength', 256);
S_stft = gather(S_stft_gpu);

% Create meshgrid for 3D surface
[T_mesh, F_mesh] = meshgrid(T_stft, F_stft);
S_stft_mag = 20*log10(abs(S_stft) + eps); % Convert to dB scale

% Create 3D surface plot
surf(T_mesh, F_mesh, S_stft_mag, 'EdgeColor', 'none');
view(-45, 45);
lighting phong;
camlight('headlight');
material([0.7 0.9 0.3 1]);
shading interp;

title('3D STFT');
xlabel('Time (s)');
ylabel('Frequency (Hz)');
zlabel('Magnitude (dB)');
colorbar;

% Plot 8: CWT Scalogram (3D)
subplot(4, 4, 8);
fb = cwtfilterbank('SignalLength', numel(complexSignal), 'SamplingFrequency', 1024);
[cfs, frq] = cwt(complexSignal, 'FilterBank', fb);
t_cwt = (0:numel(complexSignal)-1) / 1024;
cfs_mag = abs(cfs);
if size(cfs_mag, 3) > 1
    cfs_mag = mean(cfs_mag, 3);
end
if size(cfs_mag, 1) ~= length(frq)
    cfs_mag = cfs_mag';
end

% Create meshgrid for 3D plotting
[T, F] = meshgrid(t_cwt, frq);

% Create 3D surface plot
surf(T, F, cfs_mag, 'EdgeColor', 'none');
colormap('jet');
xlabel('Time (s)');
ylabel('Frequency (Hz)');
zlabel('Magnitude');
title('CWT Scalogram (3D)');
colorbar;
view(-45, 45);
lighting gouraud;

% Plot 9: MODWT Coefficients
subplot(4, 4, 9);
wt = modwt(real(complexSignal), 'sym4', 5);
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
grid on;

% Plot 10: STFT Spectrogram (3D)
subplot(4, 4, 10);
[S_spec, F_spec, T_spec] = spectrogram(real(complexSignal), hamming(256), 128, 256, 1024);
S_db = 10*log10(abs(S_spec) + eps);
surf(T_spec, F_spec, S_db, 'EdgeColor', 'none');
axis tight;
view(-15, 60); % Adjust view angle
xlabel('Time (s)');
ylabel('Frequency (Hz)');
zlabel('Magnitude (dB)');
title('STFT Spectrogram (3D)');
colorbar;
lighting phong;
camlight('headlight');
material([0.7 0.9 0.3 1]);
shading interp;

% Plot 11: 3D Wavelet Scattering Features Difference
subplot(4, 4, 11);
scatter_features_real = featureMatrix(sn, real(complexSignal));
scatter_features_imag = featureMatrix(sn, imag(complexSignal));
scatter_features_diff = scatter_features_real - scatter_features_imag;

% Create meshgrid for 3D visualization
[X_scat, Y_scat] = meshgrid(1:size(scatter_features_diff,2), 1:size(scatter_features_diff,1));

% Create 3D surface plot
surf(X_scat, Y_scat, scatter_features_diff, 'EdgeColor', 'none');
view(-45, 45);
lighting phong;
camlight('headlight');
material([0.7 0.9 0.3 1]);
shading interp;
title('3D Wavelet Scattering Features (Real - Imag)');
xlabel('Scattering Path');
ylabel('Time');
zlabel('Difference Magnitude');
colorbar;
colormap(jet);
grid on;

% Plot 12: Enhanced Signal Constellation Diagram
subplot(4, 4, 12);
signal_power = abs(complexSignal).^2;
signal_power_db = 10*log10(signal_power + eps); % Avoid log(0)

scatter(real(complexSignal), imag(complexSignal), 20, signal_power_db, 'filled');
grid on;
xlabel('In-phase Component (I)');
ylabel('Quadrature Component (Q)');
title('Signal Constellation');
colormap(gca, jet);
c = colorbar;
c.Label.String = 'Signal Power (dB)';
axis equal;
axis square;

% Add annotation for modulation type
text(min(real(complexSignal)), max(imag(complexSignal)), ...
    sprintf('Modulation: %s\nSNR: %.1f dB', classes{modIndex}, snr), ...
    'VerticalAlignment', 'top', 'FontSize', 8);

% Cyclostationary Feature Analysis for DeepSig RadioML 2018.01A
subplot(4, 4, 13);

try
    % Parameters for cyclic spectral analysis
    window_length = 256;
    overlap = window_length / 2;
    nfft = 512;

    % Generate lag values within a suitable range for radio signals
    max_lag = 64;  % Original value; adjust if necessary
    lag_vector = -max_lag:max_lag;

    % Initialize cyclic correlation matrix
    cyclic_corr = zeros(length(lag_vector), nfft / 2 + 1);

    % Ensure complexSignal is a column vector
    complexSignal = complexSignal(:);

    % Precompute frequency vector
    dummy_product = complexSignal .* conj(complexSignal);  % Zero lag
    [~, f] = pwelch(dummy_product, hamming(window_length), overlap, nfft, 1024);
    f = f(:).';  % Ensure row vector

    % Compute cyclic correlation for different lags
    for idx = 1:length(lag_vector)
        lag = lag_vector(idx);

        % Shift signal based on lag
        if lag >= 0
            padded_signal = [zeros(lag, 1); complexSignal(1:end - lag)];
        else
            padded_signal = [complexSignal(-lag + 1:end); zeros(-lag, 1)];
        end

        % Conjugate multiplication for correlation
        product = complexSignal .* conj(padded_signal);

        % Compute power spectral density
        [Sxx, ~] = pwelch(product, hamming(window_length), overlap, nfft, 1024);

        % Store one-sided PSD
        Sxx_one_sided = Sxx(1:nfft / 2 + 1).';  % Transpose to row vector
        cyclic_corr(idx, :) = Sxx_one_sided;
    end

    % Convert to dB scale and normalize
    cyclic_corr_db = 10 * log10(abs(cyclic_corr) + eps);
    cyclic_corr_db = cyclic_corr_db - min(cyclic_corr_db(:));
    cyclic_corr_db = cyclic_corr_db / max(cyclic_corr_db(:));

    % Create meshgrid for plotting
    [X, Y] = meshgrid(f, lag_vector);

    % Verify dimensions
    if ~isequal(size(X), size(Y), size(cyclic_corr_db))
        error('Dimensions of X, Y, and cyclic_corr_db do not match.');
    end

    % Create 3D surface plot
    surf(X, Y, cyclic_corr_db, 'EdgeColor', 'none');
    colormap(jet);
    colorbar;
    c = colorbar;
    c.Label.String = 'Normalized Cyclic Correlation (dB)';
    xlabel('Frequency (Hz)');
    ylabel('Lag');
    zlabel('Normalized Cyclic Correlation (dB)');
    title(sprintf('3D Cyclic Spectral Analysis\n%s Modulation', classes{modIndex}));

    % Adjust view angle for better perception
    view(45, 30);

    % Add lighting for depth perception
    camlight('right');
    lighting phong;

    % Optional: Add peak marker
    [~, max_idx] = max(cyclic_corr_db(:));
    [max_row, max_col] = ind2sub(size(cyclic_corr_db), max_idx);
    hold on;
    plot3(f(max_col), lag_vector(max_row), cyclic_corr_db(max_row, max_col), 'w*', 'MarkerSize', 10, 'LineWidth', 2);
    hold off;

    % Add SNR annotation
    text(min(f), max(lag_vector), max(cyclic_corr_db(:)), ...
        sprintf('SNR: %.1f dB', snr), ...
        'VerticalAlignment', 'bottom', 'Color', 'white', ...
        'FontSize', 8, 'BackgroundColor', 'black');

    % Print analysis metrics
    fprintf('\nCyclic Analysis Metrics:\n');
    fprintf('Maximum correlation: %.2f\n', max(cyclic_corr_db(:)));
    fprintf('Frequency resolution: %.2f Hz\n', f(2) - f(1));
    fprintf('Number of lag points: %d\n', length(lag_vector));

catch ME
    % Error handling
    fprintf('Error in cyclostationary analysis: %s\n', ME.message);
    fprintf('Error details:\n%s\n', getReport(ME, 'extended'));
    text(0.5, 0.5, 'Error in analysis', ...
        'HorizontalAlignment', 'center', 'Color', 'red');
    axis([0 1 0 1]);
end

% Adjust appearance
set(gca, 'XTickMode', 'auto', 'XTickLabelMode', 'auto');
set(gca, 'Position', get(gca, 'OuterPosition') - ...
    get(gca, 'TightInset') * [-1 0 1 0; 0 -1 0 1; 0 0 1 0; 0 0 0 1]);


% Plot 14: Energy Detector
subplot(4, 4, 14);
energy_threshold = 0.7 * max(abs(complexSignal_gpu).^2); % 70% of max energy
signal_energy = abs(complexSignal_gpu).^2;
detected_signals = signal_energy > energy_threshold;
t = (0:length(complexSignal_gpu)-1) / 1024; % Assuming 1024 Hz sampling rate
plot(t, signal_energy, 'b', 'LineWidth', 1.5);
hold on;
plot(t, energy_threshold * ones(size(t)), 'r--', 'LineWidth', 1.5);
plot(t(detected_signals), signal_energy(detected_signals), 'ro', 'MarkerSize', 4);
hold off;
title('Energy Detector');
xlabel('Time (s)');
ylabel('Signal Energy');
legend('Signal Energy', 'Threshold', 'Detected Signals');
grid on;

% Plot 15: 3D Spectrogram Plot
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
colormap(jet);
colorbar;
lighting phong;
camlight('headlight');
material([0.7 0.9 0.3 1]);
shading interp;

% Find all axes except colorbars
all_axes = findall(fig, 'type', 'axes');
for ax = all_axes'
    if ~strcmp(get(ax, 'Tag'), 'Colorbar')  % Skip colorbars
        pos = get(ax, 'Position');
        set(ax, 'Position', [pos(1) pos(2) pos(3)*1.1 pos(4)]);  % Make plots 10% wider
        set(ax, 'XTickLabelRotation', 45);  % Rotate x labels for better readability
    end
end

% Adjust overall figure layout
set(fig, 'Units', 'normalized');
set(fig, 'Position', [0.1 0.1 0.8 0.8]);  % Use 80% of screen width/height

% Reduce spacing between subplots
spacing = 0.02;  % Reduced spacing
margins = 0.1;   % Margins around edges
height = (1 - 2*margins - 3*spacing) / 4;  % Height for each subplot
width = (1 - 2*margins - 3*spacing) / 4;   % Width for each subplot

% Apply tighter layout adjustments as needed
% This section can be customized further based on specific needs