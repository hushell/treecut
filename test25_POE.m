clear
%close all

pt_path = 'output/processed_trees/';

% prepare data
load output/trees/train/100075_tree.mat % tree
img = imread('data/images/train/100075.jpg');
load data/ucm2/train/100075.mat % ucm2
ucm = ucm2(3:2:end, 3:2:end);
segMap = bwlabel(ucm <= 0, 4);
load data/groundTruth/train/100075.mat % gt
nsegs = numel(groundTruth);
for s = 1:nsegs
    groundTruth{s}.Segmentation = double(groundTruth{s}.Segmentation);
end

% p = 0.9, scale = 0.9
thisTreePath = [pt_path 'train/100075_tree.mat'];
thisTree = tree_preprocess(thisTreePath, thisTree, img, segMap);

gauss_lliks = thisTree.llik;

% POE
block = [15 15];
diam = floor(block(1)/2);
cid = ceil(prod(block)/2);
im1 = rgb2gray(img);
[nrow,ncol] = size(im1);
[xs,ttt] = myim2col(im1,block);
centres = ttt(cid,:);
xs = double(xs)/255;
xs = xs';

poe_lliks = zeros(thisTree.numTotalNodes,1);
for i = 540:thisTree.numTotalNodes
    leaves = thisTree.leafsUnder{i};
    pixs = find(ismember(segMap, leaves) == 1);
    patches = xs(ismember(centres, pixs),:);
    
    if isempty(patches)
        pixs = [pixs-diam;pixs+diam;pixs-(nrow*diam);pixs+(nrow*diam)];
        patches = xs(ismember(centres, pixs),:);
    end
    
    w = poe_learn(patches, block, 0);
    
    llik = poe_logprob(w,patches,block);
    poe_lliks(i) = llik;
    fprintf('llik(%d) = %f\n', i, llik);
end

thisTree.llik = poe_lliks;

% [aftTree,segLabels] = inference_temp(thisTree, 0.9, 1e-4);
% 
% [PRI, VOI, labMap] = eval_seg(segMap, segLabels, groundTruth);
% [cntR, sumR] = covering_rate_ois(labMap, groundTruth);
% COV = cntR ./ (sumR + (sumR==0));
% fprintf('PRI = %.2f, VOI = %.2f, COV = %.2f, nSeg = %d\n', PRI, VOI, COV, numel(unique(segLabels)));


ps = [exp(linspace(log(0.0001), log(0.09), 5)) exp(linspace(log(0.1), log(0.79), 35)) exp(linspace(log(0.8), log(0.89), 30)) exp(linspace(log(0.9), log(0.9999), 30))];
n_s = length(ps);

scals = [1e-3 5e-4 1e-4 5e-5 1e-5 1e-6];
n_r = length(scals);

        grid_PRI  = zeros(n_r,n_s);
        grid_VOI  = zeros(n_r,n_s);
        grid_nLab = zeros(n_r,n_s);
        grid_COV  = zeros(n_r,n_s);

        for j = 1:n_s
            for r = 1:n_r
                p = ps(j);
                scal = scals(r);

                [~,segLabels] = inference_temp(thisTree, p, scal); % thisTree
                
                [PRI, VOI, labMap] = eval_seg(segMap, segLabels, groundTruth);
                grid_PRI(r,j) = PRI;
                grid_VOI(r,j) = VOI;
                grid_nLab(r,j) = numel(unique(segLabels));
                
                [cntR, sumR] = covering_rate_ois(labMap, groundTruth);
                COV = cntR ./ (sumR + (sumR==0));
                grid_COV(r,j) = COV;
                fprintf('(%d,%d): COV = %f\n', j, r, COV); 
            end % r
        end % j
        
        [C,I] = max(max(grid_COV(:,:)))  
        [A,B] = ind2sub(size(grid_COV),I)