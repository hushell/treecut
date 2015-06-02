function test26_POE_paral(dataset, img_s, img_t)
if nargin < 3
    dataset = 'train';
    img_range = [1 2];
end

img_range = [str2num(img_s):str2num(img_t)];

fprintf('===== %s set =====\n', dataset);
fprintf('===== %d to %d=====\n', img_range(1), img_range(end));

%% 
img_dir  = ['./data/images/' dataset '/'];
ucm2_dir = ['./data/ucm2/' dataset '/'];
gt_dir   = ['./data/groundTruth/' dataset '/'];
tree_dir = ['./output/trees/' dataset '/'];
pt_dir   = ['./output/processed_trees/' dataset '/'];
poe_dir  = ['./output/poe/' dataset '/'];

all_files = dir(ucm2_dir);
mat       = arrayfun(@(x) ~isempty(strfind(x.name, '.mat')), all_files);
all_files = all_files(logical(mat));

for i = img_range
    tic;

    % data
    [~,name] = fileparts(all_files(i).name);
    iid = str2double(name);
    temp = load([tree_dir name '_tree.mat']); % tree thres_arr
    thisTree = temp.thisTree;
    temp = load([ucm2_dir name '.mat']); % ucm2
    ucm2 = temp.ucm2;
    img = imread([img_dir name '.jpg']); % img
    ucm = ucm2(3:2:end, 3:2:end); % ucm
    segMap = bwlabel(ucm <= 0, 4); % seg

    % preprocess tree
    thisTreePath = [pt_dir name '_tree.mat'];
    thisTree = tree_preprocess(thisTreePath, thisTree, img, segMap);

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
    for n = 1:thisTree.numTotalNodes
        leaves = thisTree.leafsUnder{n};
        pixs = find(ismember(segMap, leaves) == 1);
        patches = xs(ismember(centres, pixs),:);
        
        if isempty(patches)
            pixs = [pixs-diam;pixs+diam;pixs-(nrow*diam);pixs+(nrow*diam)];
            patches = xs(ismember(centres, pixs),:);
        end
        
        w = poe_learn(patches, block, 0);
        
        llik = poe_logprob(w,patches,block);
        poe_lliks(n) = llik;
        fprintf('llik(%d) = %f\n', n, llik);
    end % n

    save([poe_dir 'poelliks_' num2str(block(1)) 'x' num2str(block(2)) '_img_' num2str(i) '_' name '.mat'],'poe_lliks');
    fprintf('img %d_%d takes %f sec.\n', i, iid, toc);
end % i

