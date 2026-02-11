clear; clc;
rng('default');

% ================= Configuration from Paper Appendix D =================
N = 101;
xf = linspace(0, 1, N);
len = length(xf);
num_per_category = 1000; % Paper uses 1000 samples per category

% Initialize storage
% We will generate 4000 total samples:
% 1. Distributed Force (GRF)
% 2. Concentrated Force (Fourier Dirac)
% 3. Global Imposed Displacement (GRF)
% 4. Localized Imposed Displacement (Fourier Step)

total_samples = 4 * num_per_category;
data_matrix = zeros(total_samples, len);
type_list = zeros(total_samples, 1); 
% Type ID Key:
% 1 = Force BC (Distributed/GRF)
% 2 = Force BC (Concentrated/Fourier)
% 3 = Disp BC (Global/GRF)
% 4 = Disp BC (Localized/Fourier)

fprintf('Generating dataset based on Table D1...\n');

% ================= CATEGORY 1 & 3: GRF (Smooth) =================
% Table D1: "mean-zero, length-scale l=0.2" 
l = 0.2; 
cov = zeros(len, len);
for i = 1:len
    for j = 1:len
        cov(i, j) = exp(-0.5*((xf(i)-xf(j))^2) / l^2);
    end
end
mu = zeros(1, len);

% Generate 2000 GRF samples (1000 for Force, 1000 for Disp)
grf_pool = mvnrnd(mu, cov, 2 * num_per_category);

% --- 1. 定义缩放系数 (根据物理估算) ---
% 目标位移: 8mm = 0.008m
% 材料 E=12.5, 所以力的量级约为 0.1
SCALE_FORCE = 0.1;   % 用于 Type 1 (分布力)
SCALE_DISP  = 0.008; % 用于 Type 3 (全局位移)

% --- 2. Type 1: 分布力 (使用前 1000 条数据) ---
% 逻辑：取出数据 -> 归一化到 [-1, 1] -> 乘以力的系数
raw_data_force = grf_pool(1:1000, :);
% 逐行归一化 (确保每条曲线最大值都是 1，消除 GRF 随机幅度的不可控性)
max_vals_force = max(abs(raw_data_force), [], 2); 
norm_data_force = raw_data_force ./ max_vals_force; 
% 赋值并缩放
data_matrix(1:1000, :) = norm_data_force * SCALE_FORCE; 
type_list(1:1000) = 1;

% --- 3. Type 3: 全局位移 (使用后 1000 条数据) ---
% 逻辑：取出数据 -> 归一化到 [-1, 1] -> 乘以位移系数
raw_data_disp = grf_pool(1001:2000, :);
% 逐行归一化
max_vals_disp = max(abs(raw_data_disp), [], 2);
norm_data_disp = raw_data_disp ./ max_vals_disp;
% 赋值并缩放 (直接控制最大位移为 8mm)
data_matrix(2001:3000, :) = norm_data_disp * SCALE_DISP; 
type_list(2001:3000) = 3;

% ================= CATEGORY 2: Concentrated Force (Fourier Dirac) =================
% Table D1: "Fourier polynomials... absolute value < 400" 
fprintf('Generating Concentrated Forces (Fourier Dirac)...\n');
num_terms = 40; % Sufficient terms for sharp spike
for i = 1:num_per_category
    x0 = 0.1 + 0.8 * rand(); % Random location
    amp = (2*rand()-1) * 0.1; % Magnitude < 400
    
    signal = zeros(1, len);
    for k = 1:num_terms
        % Windowed Cosine Sum (Dirac approximation)
        w_k = 0.5 * (1 + cos(pi*k/num_terms));
        signal = signal + w_k * cos(k*pi*(xf - x0));
    end
    % Normalize and scale
    signal = signal / max(abs(signal)) * amp; 
    
    idx = 1000 + i;
    data_matrix(idx, :) = signal;
    type_list(idx) = 2;
end

% ================= CATEGORY 4: Localized Disp (Fourier Step) =================
% Table D1: "Fourier polynomials... absolute value < 100" 
fprintf('Generating Localized Displacements (Fourier Step)...\n');
for i = 1:num_per_category
    x_start = 0.1 + 0.5 * rand();
    width = 0.1 + 0.2 * rand();
    x_end = x_start + width;
    amp = (2*rand()-1) * 0.008; % Magnitude < 100
    
    signal = zeros(1, len);
    for k = 1:2:num_terms % Odd terms for box/step shape
        w_k = 0.5 * (1 + cos(pi*k/num_terms));
        term = (sin(k*pi*(xf - x_start)) - sin(k*pi*(xf - x_end))) / k;
        signal = signal + w_k * term;
    end
    signal = signal / max(abs(signal)) * amp;
    
    idx = 3000 + i;
    data_matrix(idx, :) = signal;
    type_list(idx) = 4;
end

% ================= SHUFFLE & SAVE =================
rp = randperm(total_samples);
f_bc = data_matrix(rp, :);
f_type = type_list(rp);

% Save 'f_bc' (curves) and 'f_type' (1-4 labels)
save('bc_source.mat', 'f_bc', 'f_type');

fprintf('Saved 4000 samples to bc_source.mat.\n');
fprintf('Type Distribution:\n');
fprintf('1 (Dist Force): %d\n', sum(f_type==1));
fprintf('2 (Conc Force): %d\n', sum(f_type==2));
fprintf('3 (Glob Disp):  %d\n', sum(f_type==3));
fprintf('4 (Loc Disp):   %d\n', sum(f_type==4));