function eval_BSDS500_ave(metric, eval_path)
% BSDS500 - ave

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

%% train+val on train+val ave_subj (ODS, OIS) 
fprintf('train+val on train+val ave_subj (ODS, OIS):\n');
% Alg 1 - UCM, Alg 2 - TC
% ave ==> ODS: Alg 1: COV_g = 0.607764, p_g = 0.195168, scal_g = 0.001000; 
% ave ==> OIS: Alg 1: COV_g = 0.666693
% ave ==> ODS: Alg 2: COV_g = 0.608666, p_g = 0.971283, scal_g = 0.000500; 
% ave ==> OIS: Alg 2: COV_g = 0.670814

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

grid_COV_train_ave = zeros(nis+100,nr,ns,nalg); % nscal, nps, nalg
COV_OIS_train_all = zeros(nis+100,nalg);

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
            [COV_OIS_train_all(j,a),I] = opt(temp(:),metric);
            [ri,ji] = ind2sub(size(temp),I);
            %fprintf('Alg %d: COV = %f, p = %f, scal = %f; \n', ...
            %    a, COV_OIS_train_all(j,a), ps(ji), scals(ri));
        end
    end
end

dataset = 'val';
load(['data/BSDS_' dataset '_all_files.mat'], 'all_files');
eval_dir  = [eval_path '/' dataset '/'];

nis2 = length(all_files);

for i = 1:strid:nis2
    img_s = i;
    img_t = i+strid-1;
    for j = img_s:img_t
        [~,name] = fileparts(all_files(j).name);
        fil_nam = [eval_dir 'grid_img_' num2str(j) '_' name '.mat'];
        temp = load(fil_nam, grid_metric);
        grid_res = getfield(temp, grid_metric);
        [~,~,nsub,~] = size(grid_res);

        jj = j + nis;

        % ODS
        grid_COV_train_ave(jj,:,:,:) = squeeze(sum(grid_res,3))./nsub;
        
        % OIS
        %fprintf('==> img %d:\n', j);
        for a = 1:nalg
            temp = squeeze(grid_COV_train_ave(jj,:,:,a));
            [COV_OIS_train_all(jj,a),I] = max(temp(:));
            [ri,ji] = ind2sub(size(temp),I);
            %fprintf('Alg %d: COV = %f, p = %f, scal = %f; \n', ...
            %    a, COV_OIS_train_all(jj,a), ps(ji), scals(ri));
        end
    end
end

COV_ODS_train_ave = zeros(nalg,1);
j_ODS_train_ave = zeros(nalg,1); % p or k
r_ODS_train_ave = zeros(nalg,1); % scal
COV_OIS_train_ave = zeros(nalg,1);

for i = 1:nalg
    temp = squeeze(sum(grid_COV_train_ave(:,:,:,i),1))./(nis+nis2);
    [COV_ODS_train_ave(i),I] = opt(temp(:),metric);
    [r_ODS_train_ave(i),j_ODS_train_ave(i)] = ind2sub(size(temp),I);
    fprintf('ave ==> ODS: Alg %d: %s_g = %f, p_g = %f, scal_g = %f; \n', ...
        i, metric, COV_ODS_train_ave(i), ps(j_ODS_train_ave(i)), scals(r_ODS_train_ave(i)));
    COV_OIS_train_ave(i) = mean(COV_OIS_train_all(:,i));
    fprintf('ave ==> OIS: Alg %d: %s_g = %f\n', i, metric, COV_OIS_train_ave(i));
end


%% test on test ave_subj (ODS, OIS)
fprintf('test on test ave_subj (ODS, OIS):\n');
% ave ==> ODS: Alg 1: COV_g = 0.588307, p_g = 0.234213, scal_g = 0.001000; 
% ave ==> OIS: Alg 1: COV_g = 0.647085
% ave ==> ODS: Alg 2: COV_g = 0.594165, p_g = 0.985488, scal_g = 0.000700; 
% ave ==> OIS: Alg 2: COV_g = 0.650406
dataset = 'test';
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
            [COV_OIS_val_all(j,a),I] = opt(temp(:),metric);
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
    [COV_ODS_val_ave(i),I] = opt(temp(:),metric);
    [r_ODS_val_ave(i),j_ODS_val_ave(i)] = ind2sub(size(temp),I);
    fprintf('ave ==> ODS: Alg %d: %s_g = %f, p_g = %f, scal_g = %f; \n', ...
        i, metric, COV_ODS_val_ave(i), ps(j_ODS_val_ave(i)), scals(r_ODS_val_ave(i)));
    COV_OIS_val_ave(i) = mean(COV_OIS_val_all(:,i));
    fprintf('ave ==> OIS: Alg %d: %s_g = %f\n', i, metric, COV_OIS_val_ave(i));
end


%% train+val on test ave_subj (ODS)
fprintf('train+val on test ave_subj (ODS):\n');
% ave ==> ODS: Alg 1: COV_g = 0.587303; 
% ave ==> ODS: Alg 2: COV_g = 0.592394; 
for i = 1:nalg
    COV = mean(grid_COV_val_ave(:,r_ODS_train_ave(i),j_ODS_train_ave(i),i)); 
    fprintf('ave ==> ODS: Alg %d: %s_g = %f; \n', ...
        i, metric, COV);
end


function [val, I] = opt(x, metric)
if strcmp(metric,'VOI') == 1
    x(x==0) = +Inf;
    [val, I] = min(x);
else
    [val, I] = max(x);
end

