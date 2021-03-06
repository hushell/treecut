function eval_BSDS300_ave(metric, eval_path)
% BSDS300 - ave

if nargin < 2
    eval_path  = './output/grid_eval/'; 
end

grid_metric = ['grid_' metric];
strid = 4;
%nalg = 2;

% TC
scals = [1e-3 9e-4 8e-4 7e-4 6e-4 5e-4 4e-4 3e-4 2e-4 1e-4];
nr = length(scals);
ps = [exp(linspace(log(0.0001), log(0.09), 5)) exp(linspace(log(0.1), log(0.79), 35)) exp(linspace(log(0.8), log(0.89), 30))               exp(linspace(log(0.9), log(0.9999), 30))];
ns = length(ps);

% UCM
thres = 0.01:0.01:1.00;

%% train on train ave_subj (ODS, OIS) 
fprintf('train on train ave_subj (ODS, OIS):\n');
% Alg 1 - UCM, Alg 2 - TC
% ==> ODS: Alg 1: COV_g = 0.618004, p_g = 0.195168, scal_g = 0.001000; 
% ==> OIS: Alg 1: COV_g = 0.677248
% ==> ODS: Alg 2: COV_g = 0.619642, p_g = 0.967764, scal_g = 0.000400; 
% ==> OIS: Alg 2: COV_g = 0.680502

dataset = 'train';
load(['data/BSDS_' dataset '_all_files.mat'], 'all_files');
eval_dir  = [eval_path '/' dataset '/'];
nis = length(all_files);

% get nalg
[~,name] = fileparts(all_files(1).name);
fil_nam = [eval_dir 'grid_img_' num2str(1) '_' name '.mat'];
temp = load(fil_nam, grid_metric);
grid_res = getfield(temp, grid_metric);
[~,~,~,nalg] = size(grid_res);

grid_COV_train_ave = zeros(nis,nr,ns,nalg); % nscal, nps, nalg
COV_OIS_train_all = zeros(nis,nalg);

for i = 1:strid:nis
    img_s = i;
    img_t = i+strid-1;
    for j = img_s:img_t
        [~,name] = fileparts(all_files(j).name);
        fil_nam = [eval_dir 'grid_img_' num2str(j) '_' name '.mat'];
        temp = load(fil_nam, grid_metric);
        grid_res = getfield(temp, grid_metric);
        [~,~,nsub,~] = size(grid_res);
        % ODS
        grid_COV_train_ave(j,:,:,:) = squeeze(sum(grid_res,3))./nsub;
        
        % OIS
        %fprintf('==> img %d:\n', j);
        for a = 1:nalg
            temp = squeeze(grid_COV_train_ave(j,:,:,a));
            [COV_OIS_train_all(j,a),I] = opt(temp,metric);
            [ri,ji] = ind2sub(size(temp),I);
            %fprintf('Alg %d: COV = %f, p = %f, scal = %f; \n', ...
            %    a, COV_OIS_train_all(j,a), ps(ji), scals(ri));
        end
    end
end

COV_ODS_train_ave = zeros(nalg,1);
j_ODS_train_ave = zeros(nalg,1); % p or k
r_ODS_train_ave = zeros(nalg,1); % scal
COV_OIS_train_ave = zeros(nalg,1);

sc = 6;
for i = 1:nalg
    temp = squeeze(sum(grid_COV_train_ave(:,:,:,i),1))./nis;
    %[COV_ODS_train_ave(i),I] = opt(temp,metric);
    %[r_ODS_train_ave(i),j_ODS_train_ave(i)] = ind2sub(size(temp),I);
    [COV_ODS_train_ave(i),I] = opt(temp,metric,2,sc);
    r_ODS_train_ave(i) = sc;
    j_ODS_train_ave(i) = I;

    fprintf('ave ==> ODS: Alg %d: %s_g = %f, p_g = %f, scal_g = %f; \n', ...
        i, metric, COV_ODS_train_ave(i), ps(j_ODS_train_ave(i)), scals(r_ODS_train_ave(i)));
    COV_OIS_train_ave(i) = mean(COV_OIS_train_all(:,i));
    fprintf('ave ==> OIS: Alg %d: %s_g = %f\n', i, metric, COV_OIS_train_ave(i));
end


%% val on val ave_subj (ODS, OIS)
fprintf('val on val ave_subj (ODS, OIS):\n');
% ave ==> ODS: Alg 1: COV_g = 0.588763, p_g = 0.220399, scal_g = 0.001000; 
% ave ==> OIS: Alg 1: COV_g = 0.645582
% ave ==> ODS: Alg 2: COV_g = 0.588590, p_g = 0.971283, scal_g = 0.000500; 
% ave ==> OIS: Alg 2: COV_g = 0.651440
dataset = 'val';
load(['data/BSDS_' dataset '_all_files.mat'], 'all_files');
eval_dir  = [eval_path '/' dataset '/'];

nis = length(all_files);

grid_COV_val_ave = zeros(nis,nr,ns,nalg); % nscal, nps, nalg
COV_OIS_val_all = zeros(nis,nalg);

for i = 1:strid:nis
    img_s = i;
    img_t = i+strid-1;
    for j = img_s:img_t
        [~,name] = fileparts(all_files(j).name);
        fil_nam = [eval_dir 'grid_img_' num2str(j) '_' name '.mat'];
        temp = load(fil_nam, grid_metric);
        grid_res = getfield(temp, grid_metric);
        [~,~,nsub,~] = size(grid_res);
        % ODS
        grid_COV_val_ave(j,:,:,:) = squeeze(sum(grid_res,3))./nsub;
        
        % OIS
        %fprintf('==> img %d:\n', j);
        for a = 1:nalg
            temp = squeeze(grid_COV_val_ave(j,:,:,a));
            [COV_OIS_val_all(j,a),I] = opt(temp,metric);
            [ri,ji] = ind2sub(size(temp),I);
            %fprintf('Alg %d: COV = %f, p = %f, scal = %f; \n', ...
            %    a, COV_OIS_val_all(j,a), ps(ji), scals(ri));
        end
    end
end

COV_ODS_val_ave = zeros(nalg,1);
j_ODS_val_ave = zeros(nalg,1); % p or k
r_ODS_val_ave = zeros(nalg,1); % scal
COV_OIS_val_ave = zeros(nalg,1);

for i = 1:nalg
    temp = squeeze(sum(grid_COV_val_ave(:,:,:,i),1))./nis;
    [COV_ODS_val_ave(i),I] = opt(temp,metric);
    [r_ODS_val_ave(i),j_ODS_val_ave(i)] = ind2sub(size(temp),I);

    fprintf('ave ==> ODS: Alg %d: %s_g = %f, p_g = %f, scal_g = %f; \n', ...
        i, metric, COV_ODS_val_ave(i), ps(j_ODS_val_ave(i)), scals(r_ODS_val_ave(i)));
    COV_OIS_val_ave(i) = mean(COV_OIS_val_all(:,i));
    fprintf('ave ==> OIS: Alg %d: %s_g = %f\n', i, metric, COV_OIS_val_ave(i));
end


%% train on val ave_subj (ODS)
fprintf('train on val ave_subj (ODS):\n');
% ave ==> ODS: Alg 1: COV_g = 0.587282; 
% ave ==> ODS: Alg 2: COV_g = 0.583039;
for i = 1:nalg
    COV = mean(grid_COV_val_ave(:,r_ODS_train_ave(i),j_ODS_train_ave(i),i)); 
    fprintf('ave ==> ODS: Alg %d: %s_g = %f; \n', ...
        i, metric, COV);
end


function [val, I] = opt(x, metric, free_dim, fixed_val)
if nargin < 3
    free_dim = 0;
    fixed_val = 0;
end

if strcmp(metric,'VOI') == 1
    x(x==0) = +Inf;
    [val, I] = mymin(x, free_dim, fixed_val);
else
    [val, I] = mymax(x, free_dim, fixed_val);
end

function [val, I] = mymax(x, free_dim, fixed_val)
if nargin < 2
    free_dim = 0;
end

if free_dim == 0
    [val, I] = max(x(:));
else
    if nargin < 3
        error('fixed_val on fixed_dim should be specified!');
    end
    [val, I] = max(x, [], free_dim);
    val = val(fixed_val);
    I = I(fixed_val);
end

function [val, I] = mymin(x, free_dim, fixed_val)
if nargin < 2
    free_dim = 0;
end

if free_dim == 0
    [val, I] = min(x(:));
else
    if nargin < 3
        error('fixed_val on fixed_dim should be specified!');
    end
    [val, I] = min(x, [], free_dim);
    val = val(fixed_val);
    I = I(fixed_val);
end
