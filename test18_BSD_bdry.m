clear
close all
addpath bdry_bench

dataset = 'train';

img_dir  = ['./data/images/' dataset '/'];
ucm2_dir = ['./data/ucm2/' dataset '/'];
gt_dir   = ['./data/groundTruth/' dataset '/'];
tree_dir = ['./output/trees/' dataset '/'];
pt_dir   = ['./output/processed_trees/' dataset '/'];
out_dir  = ['./output/bdry/' dataset '/'];

g_subjects = [1102:1117 1119 1121:1124 1126:1130 1132];
iids_train = load(fullfile('data/iids_train.txt')); %

nis = 200;
iids_ind = zeros(length(g_subjects),nis); % 
for i = 1:length(g_subjects)
    s = g_subjects(i);
    load(['data/gt_' num2str(s) '.mat']);
    iids_ind(i,:) = ismember(iids_train, all_iids);
end
sub_sel = [1105 1109 1123];
n_sub = length(sub_sel);
subs = ismember(g_subjects, sub_sel)';
iids_sel = iids_ind(subs,:); % 
iids_inter = sum(iids_sel,1) == n_sub; %

N = 50; % n samples
for si = 1:n_sub
    fprintf('--------------------\n');
    fprintf('Subject %d\n', sub_sel(si));
    
    load(['mat_log/BSDS_test13_' num2str(sub_sel(si)) '.mat']);
    
    % samples
    for j = 1:nis
        if iids_sel(si,j) == 0 
            continue 
        end

        fprintf('Image %d\n', j);
        iid = iids_train(j);
        name = num2str(iid);
        load([tree_dir name '_tree.mat']); % tree thres_arr
        load([ucm2_dir name '.mat']); % ucm2
        load([gt_dir name '.mat']); % gt
        img = imread([img_dir name '.jpg']); % img
        ucm = ucm2(3:2:end, 3:2:end); % ucm
        segMap = bwlabel(ucm <= 0, 4); % seg

        % gt
        subjects = g_subjects(logical(iids_ind(:,j)'));
        gt_msk = ismember(subjects, sub_sel(si)); 
        assert(sum(gt_msk) == 1);
        gt_sub = cell(1,1);
        gt_sub{1}.Boundaries = double(groundTruth{gt_msk}.Boundaries);
        
        % sample
        aftTree = all_aftTree{j,3};
        samples = post_sample(aftTree, N);
        ave_img = zeros(size(segMap));
        for n = 1:N
            numSegs = size(samples,2);
            labMap = zeros(size(segMap));
            for i = 1:numSegs
                labMap(segMap == i) = samples(n,i);
            end
            el = strel('diamond',1);
            for i = 1:2
               tmp = imdilate(labMap,el);
               labMap(labMap == 0) = tmp(labMap == 0);
            end

            bdry = seg2bdry(labMap, 'imageSize');
            ave_img = ave_img + bdry;
        end % n
        
        ave_img = ave_img / N;

        evFile = [out_dir, name, '_subj_', num2str(sub_sel(si)), '_ev1.txt'];
        evaluation_bdry_image(ave_img, gt_sub, evFile);
    end % j

    collect_eval_bdry(out_dir);
    plot_eval(out_dir);
end % s
