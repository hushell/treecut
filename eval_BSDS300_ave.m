function eval_BSDS300_ave(metric)
% BSDS300 - ave

grid_metric = ['grid_' metric];

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

img_dir  = ['./data/images/' dataset '/'];
ucm2_dir = ['./data/ucm2/' dataset '/'];
gt_dir   = ['./data/groundTruth/' dataset '/'];
tree_dir = ['./output/trees/' dataset '/'];
pt_dir   = ['./output/processed_trees/' dataset '/'];
eval_dir  = ['./output/grid_eval/' dataset '/'];

all_files = dir(ucm2_dir);
mat       = arrayfun(@(x) ~isempty(strfind(x.name, '.mat')), all_files);
all_files = all_files(logical(mat));

nis = length(all_files);
strid = 4;
nalg = 2;

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
        for a = 1:2
            temp = squeeze(grid_COV_train_ave(j,:,:,a));
            [COV_OIS_train_all(j,a),I] = opt(temp(:),metric);
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

for i = 1:2
    temp = squeeze(sum(grid_COV_train_ave(:,:,:,i),1))./nis;
    [COV_ODS_train_ave(i),I] = opt(temp(:),metric);
    [r_ODS_train_ave(i),j_ODS_train_ave(i)] = ind2sub(size(temp),I);
    fprintf('ave ==> ODS: Alg %d: COV_g = %f, p_g = %f, scal_g = %f; \n', ...
        i, COV_ODS_train_ave(i), ps(j_ODS_train_ave(i)), scals(r_ODS_train_ave(i)));
    COV_OIS_train_ave(i) = mean(COV_OIS_train_all(:,i));
    fprintf('ave ==> OIS: Alg %d: COV_g = %f\n', i, COV_OIS_train_ave(i));
end


%% val on val ave_subj (ODS, OIS)
fprintf('val on val ave_subj (ODS, OIS):\n');
% ave ==> ODS: Alg 1: COV_g = 0.588763, p_g = 0.220399, scal_g = 0.001000; 
% ave ==> OIS: Alg 1: COV_g = 0.645582
% ave ==> ODS: Alg 2: COV_g = 0.588590, p_g = 0.971283, scal_g = 0.000500; 
% ave ==> OIS: Alg 2: COV_g = 0.651440
dataset = 'val';

img_dir  = ['./data/images/' dataset '/'];
ucm2_dir = ['./data/ucm2/' dataset '/'];
gt_dir   = ['./data/groundTruth/' dataset '/'];
tree_dir = ['./output/trees/' dataset '/'];
pt_dir   = ['./output/processed_trees/' dataset '/'];
eval_dir  = ['./output/grid_eval/' dataset '/'];

all_files = dir(ucm2_dir);
mat       = arrayfun(@(x) ~isempty(strfind(x.name, '.mat')), all_files);
all_files = all_files(logical(mat));

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
        for a = 1:2
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

for i = 1:2
    temp = squeeze(sum(grid_COV_val_ave(:,:,:,i),1))./nis;
    [COV_ODS_val_ave(i),I] = opt(temp(:),metric);
    [r_ODS_val_ave(i),j_ODS_val_ave(i)] = ind2sub(size(temp),I);
    fprintf('ave ==> ODS: Alg %d: COV_g = %f, p_g = %f, scal_g = %f; \n', ...
        i, COV_ODS_val_ave(i), ps(j_ODS_val_ave(i)), scals(r_ODS_val_ave(i)));
    COV_OIS_val_ave(i) = mean(COV_OIS_val_all(:,i));
    fprintf('ave ==> OIS: Alg %d: COV_g = %f\n', i, COV_OIS_val_ave(i));
end


%% train on val ave_subj (ODS)
fprintf('train on val ave_subj (ODS):\n');
% ave ==> ODS: Alg 1: COV_g = 0.587282; 
% ave ==> ODS: Alg 2: COV_g = 0.583039;
for i = 1:2
    COV = mean(grid_COV_val_ave(:,r_ODS_train_ave(i),j_ODS_train_ave(i),i)); 
    fprintf('ave ==> ODS: Alg %d: COV_g = %f; \n', ...
        i, COV);
end


function [val, I] = opt(x, metric)
if strcmp(metric,'VOI') == 1
    x(x==0) = +Inf;
    [val, I] = min(x);
else
    [val, I] = max(x);
end

