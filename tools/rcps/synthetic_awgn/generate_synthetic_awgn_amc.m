function generate_synthetic_awgn_amc(output_root, samples_per_class, frame_len, seed, snr_csv)
% Generate a clean-paired AWGN AMC dataset in the CSRR annotation format.
%
% Example:
%   generate_synthetic_awgn_amc('/tmp/synthetic_awgn_smoke', 20, 128, 2026, '-20,-18,0,18')

if nargin < 1 || isempty(output_root)
    output_root = '/home/citybuster/Data/RCPS/processed/synthetic_awgn_amc_v1';
end
if nargin < 2 || isempty(samples_per_class)
    samples_per_class = 1000;
end
if nargin < 3 || isempty(frame_len)
    frame_len = 128;
end
if nargin < 4 || isempty(seed)
    seed = 2026;
end
if nargin < 5 || isempty(snr_csv)
    snr_csv = '-20,-18,-16,-14,-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12,14,16,18';
end

snrs = str2double(strsplit(char(snr_csv), ','));
mods = {'8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', ...
        '4PAM', '16QAM', '64QAM', 'QPSK', 'WBFM'};

rng(seed, 'twister');
ensure_dir(output_root);
ensure_dir(fullfile(output_root, 'iq'));
ensure_dir(fullfile(output_root, 'clean'));

train_list = struct([]);
val_list = struct([]);
test_list = struct([]);
clean_index = 0;
sample_index = 0;
snr_errors = [];

for mod_idx = 1:numel(mods)
    mod_name = mods{mod_idx};
    order = randperm(samples_per_class);
    n_train = floor(0.70 * samples_per_class);
    n_val = floor(0.15 * samples_per_class);
    train_ids = order(1:n_train);
    val_ids = order(n_train + 1:n_train + n_val);

    for local_id = 1:samples_per_class
        clean_index = clean_index + 1;
        clean_id = clean_index - 1;
        clean = synth_modulation(mod_name, frame_len);
        clean = clean ./ sqrt(mean(abs(clean).^2) + eps);
        clean_iq = single([real(clean); imag(clean)]);
        clean_file = sprintf('%012d.npy', clean_id);
        write_npy_float32(fullfile(output_root, 'clean', clean_file), clean_iq);

        if any(train_ids == local_id)
            split = 'train';
        elseif any(val_ids == local_id)
            split = 'validation';
        else
            split = 'test';
        end

        for snr_idx = 1:numel(snrs)
            snr_db = snrs(snr_idx);
            noisy = add_awgn(clean, snr_db);
            noisy_iq = single([real(noisy); imag(noisy)]);
            file_name = sprintf('%012d.npy', sample_index);
            write_npy_float32(fullfile(output_root, 'iq', file_name), noisy_iq);

            noise = noisy - clean;
            measured = 10 * log10(mean(abs(clean).^2) / (mean(abs(noise).^2) + eps));
            snr_errors(end + 1) = measured - snr_db; %#ok<AGROW>

            item = struct( ...
                'file_name', file_name, ...
                'clean_file_name', clean_file, ...
                'clean_id', clean_id, ...
                'sample_idx', sample_index, ...
                'modulation', mod_name, ...
                'snr', snr_db, ...
                'seed', seed, ...
                'channel_type', 'awgn', ...
                'has_clean_signal', true);

            if strcmp(split, 'train')
                train_list = append_struct(train_list, item);
            elseif strcmp(split, 'validation')
                val_list = append_struct(val_list, item);
            else
                test_list = append_struct(test_list, item);
            end
            sample_index = sample_index + 1;
        end
    end
end

metainfo = struct();
metainfo.modulations = mods;
metainfo.snrs = num2cell(snrs);
metainfo.generator = 'generate_synthetic_awgn_amc.m';
metainfo.frame_len = frame_len;
metainfo.samples_per_class = samples_per_class;
metainfo.seed = seed;
metainfo.channel_type = 'awgn';
metainfo.has_clean_signal = true;

write_annotation(fullfile(output_root, 'train.json'), metainfo, train_list);
write_annotation(fullfile(output_root, 'validation.json'), metainfo, val_list);
write_annotation(fullfile(output_root, 'test.json'), metainfo, test_list);

manifest = struct();
manifest.output_root = output_root;
manifest.modulations = mods;
manifest.snrs = num2cell(snrs);
manifest.samples_per_class = samples_per_class;
manifest.frame_len = frame_len;
manifest.seed = seed;
manifest.total_clean = clean_index;
manifest.total_noisy = sample_index;
manifest.mean_snr_error_db = mean(abs(snr_errors));
manifest.max_snr_error_db = max(abs(snr_errors));
manifest.matlab_version = version;
manifest.communications_toolbox = license('test', 'communication_toolbox');
write_text(fullfile(output_root, 'generator_manifest.json'), jsonencode(manifest, 'PrettyPrint', true));

fprintf('Synthetic AWGN AMC generated at %s\n', output_root);
fprintf('  clean samples: %d\n', clean_index);
fprintf('  noisy samples: %d\n', sample_index);
fprintf('  mean |SNR error|: %.4f dB\n', manifest.mean_snr_error_db);
fprintf('  max |SNR error|: %.4f dB\n', manifest.max_snr_error_db);
end

function x = synth_modulation(mod_name, frame_len)
switch mod_name
    case 'BPSK'
        x = psk_symbols(2, frame_len);
    case 'QPSK'
        x = psk_symbols(4, frame_len);
    case '8PSK'
        x = psk_symbols(8, frame_len);
    case '4PAM'
        levels = [-3, -1, 1, 3] / sqrt(5);
        x = levels(randi(numel(levels), 1, frame_len));
    case '16QAM'
        x = qam_symbols(4, frame_len);
    case '64QAM'
        x = qam_symbols(8, frame_len);
    case 'CPFSK'
        x = fsk_like(frame_len, false);
    case 'GFSK'
        x = fsk_like(frame_len, true);
    case 'AM-DSB'
        m = message_signal(frame_len);
        x = 1 + 0.65 * m;
    case 'AM-SSB'
        m = message_signal(frame_len);
        x = analytic_signal(m);
    case 'WBFM'
        m = message_signal(frame_len);
        x = exp(1j * 2.2 * cumsum(m));
    otherwise
        error('Unsupported modulation: %s', mod_name);
end
x = reshape(x, 1, []);
x = x(1:frame_len);
x = x - mean(x);
if mean(abs(x).^2) < 1e-8
    x = x + 1e-3 * randn(size(x));
end
end

function x = psk_symbols(M, n)
k = randi([0, M - 1], 1, n);
phase = 2 * pi * k / M + pi / M;
x = exp(1j * phase);
end

function x = qam_symbols(side, n)
levels = -(side - 1):2:(side - 1);
i = levels(randi(side, 1, n));
q = levels(randi(side, 1, n));
x = i + 1j * q;
x = x / sqrt(mean(abs(x).^2));
end

function x = fsk_like(n, smooth)
symbols = 2 * randi([0, 1], 1, n) - 1;
if smooth
    g = exp(-((-4:4).^2) / 5);
    g = g / sum(g);
    symbols = conv(symbols, g, 'same');
end
phase = cumsum(0.55 * pi * symbols);
x = exp(1j * phase);
end

function m = message_signal(n)
w = randn(1, n + 16);
kernel = ones(1, 9) / 9;
m = conv(w, kernel, 'same');
m = m(9:8 + n);
m = m / (max(abs(m)) + eps);
end

function z = analytic_signal(x)
n = numel(x);
X = fft(x);
h = zeros(1, n);
if mod(n, 2) == 0
    h(1) = 1;
    h(n / 2 + 1) = 1;
    h(2:n / 2) = 2;
else
    h(1) = 1;
    h(2:(n + 1) / 2) = 2;
end
z = ifft(X .* h);
end

function y = add_awgn(x, snr_db)
signal_power = mean(abs(x).^2);
noise_power = signal_power / (10^(snr_db / 10));
noise = sqrt(noise_power / 2) * (randn(size(x)) + 1j * randn(size(x)));
y = x + noise;
end

function out = append_struct(arr, item)
if isempty(arr)
    out = item;
else
    out = [arr, item]; %#ok<AGROW>
end
end

function write_annotation(path, metainfo, data_list)
payload = struct();
payload.metainfo = metainfo;
payload.data_list = data_list;
write_text(path, jsonencode(payload, 'PrettyPrint', true));
end

function write_text(path, text)
fid = fopen(path, 'w');
if fid < 0
    error('Cannot open %s for writing.', path);
end
cleaner = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', text);
end

function ensure_dir(path)
if ~exist(path, 'dir')
    mkdir(path);
end
end

function write_npy_float32(path, data)
data = single(data);
shape = size(data);
if numel(shape) ~= 2
    error('write_npy_float32 expects a 2-D array.');
end
header = sprintf('{''descr'': ''<f4'', ''fortran_order'': False, ''shape'': (%d, %d), }', shape(1), shape(2));
header_len_no_pad = length(header) + 1;
pad_len = mod(16 - mod(10 + header_len_no_pad, 16), 16);
header = [header, repmat(' ', 1, pad_len), sprintf('\n')];
fid = fopen(path, 'w', 'ieee-le');
if fid < 0
    error('Cannot open %s for writing.', path);
end
cleaner = onCleanup(@() fclose(fid));
fwrite(fid, [147, double('NUMPY')], 'uint8');
fwrite(fid, [1, 0], 'uint8');
fwrite(fid, uint16(length(header)), 'uint16');
fwrite(fid, header, 'char');
fwrite(fid, data.', 'single');
end
