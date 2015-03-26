clear
close all

dataset = 'test';

img_dir  = ['./data/images/' dataset '/'];
ucm2_dir = ['./data/ucm2/' dataset '/'];
gt_dir   = ['./data/groundTruth/' dataset '/'];
tree_dir = ['./output/ucm_trees/' dataset '/'];
pt_dir   = ['./output/ucm_processed_trees/' dataset '/'];

all_files = dir(ucm2_dir);
mat       = arrayfun(@(x) ~isempty(strfind(x.name, '.mat')), all_files);
all_files = all_files(logical(mat));

all_perAnnoR = cell(1,numel(all_files));
all_perAnnoR_UCM = cell(1,numel(all_files));

cumCntR = 0;
cumSumR = 0;
cumCntRUCM = 0;
cumSumRUCM = 0;

for i = 1:numel(all_files)
    % prepare data
    [~,name] = fileparts(all_files(i).name);
    load([tree_dir name '_tree.mat']); % tree
    load([ucm2_dir name '.mat']); % ucm2
    load([gt_dir name '.mat']); % gt
    img = imread([img_dir name '.jpg']); % img
    
    ucm = ucm2(3:2:end, 3:2:end);
    segMap = bwlabel(ucm <= 0, 4);
    
    nsegs = numel(groundTruth);
    for s = 1:nsegs
        groundTruth{s}.Segmentation = double(groundTruth{s}.Segmentation);
    end
    
    % preprocess tree
    thisTreePath = [pt_dir name '_tree.mat'];
    thisTree = tree_preprocess(thisTreePath, thisTree, img, segMap);
    
    % test treecut
    scal = 1e-3;
    n_r = 1;
    n_s = 50;
    log_ps = linspace(-2.5,-0.005,n_s); % exp(-2.5) ~ 0.08, exp(-0.1) = 0.9
    log_ps = [log_ps log(0.99) log(0.999) log(0.9999) log(0.99999) log(0.999999)];
    n_s = n_s + 5;

    i_labMap = cell(1,n_s);
    for j = 1:n_s
        p = exp(log_ps(j));
        [~,segLabels] = inference_temp(thisTree, p, scal);

        numSegs = length(segLabels);
        labMap = zeros(size(segMap));
        for s = 1:numSegs
            labMap(segMap == s) = segLabels(s);
        end

        i_labMap{j} = labMap; 
    end

    perAnnoR = zeros(1,nsegs);
    for a = 1:nsegs
        perGT = groundTruth(a);
        bestR = 0;
        bestCntR = 0;
        bestSumR = 0;
        bestj = 0;

        for j = 1:n_s
            labMap = i_labMap{j};
            [cntR, sumR] = covering_rate_ois(labMap, perGT);

            R = cntR ./ (sumR + (sumR==0));
            if R > bestR
                bestR = R;
                bestCntR = cntR;
                bestSumR = sumR;
                bestj = j;
            end
        end % j
        
        cumCntR = cumCntR + bestCntR;
        cumSumR = cumSumR + bestSumR;

        perAnnoR(a) = bestR;

        fprintf('TC: img %d, anno %d -- perAnnoCOV = %f at p = %f\n', i, a, bestR, exp(log_ps(bestj)));
    end % a

    all_perAnnoR{i} = perAnnoR;

    % test UCM
    el = strel('diamond',1);

    perAnnoR = zeros(1,nsegs);
    for a = 1:nsegs
        perGT = groundTruth(a);
        bestR = 0;
        bestCntR = 0;
        bestSumR = 0;
        bestj = 0;

        for j = 1:length(thres_arr)
            k = thres_arr(j);
            labMap = bwlabel(ucm <= k, 4);
            
            for m = 1:2
               tmp = imdilate(labMap,el);
               labMap(labMap == 0) = tmp(labMap == 0);
            end
            
            [cntR, sumR] = covering_rate_ois(labMap, perGT);

            R = cntR ./ (sumR + (sumR==0));
            if R > bestR
                bestR = R;
                bestCntR = cntR;
                bestSumR = sumR;
                bestj = j;
            end
        end % j
        
        cumCntRUCM = cumCntRUCM + bestCntR;
        cumSumRUCM = cumSumRUCM + bestSumR;

        fprintf('UCM: img %d, anno %d -- perAnnoCOV = %f at p = %f\n', i, a, bestR, thres_arr(bestj));
    end % a

    all_perAnnoR_UCM{i} = perAnnoR;
        
end % i 

R_TC = cumCntR ./ (cumSumR + (cumSumR==0));
R_UCM = cumCntRUCM ./ (cumSumRUCM + (cumSumRUCM==0));
fprintf('*** ave_COV_TC = %f, ave_COV_UCM = %f\n', R_TC, R_UCM);

save('BSDS_test11.mat');
