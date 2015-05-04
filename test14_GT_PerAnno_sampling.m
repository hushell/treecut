clear
close all

dataset = 'train';

img_dir  = ['./data/images/' dataset '/'];
ucm2_dir = ['./data/ucm2/' dataset '/'];
gt_dir   = ['./data/groundTruth/' dataset '/'];
tree_dir = ['./output/trees/' dataset '/'];
pt_dir   = ['./output/processed_trees/' dataset '/'];
smp_dir  = ['./output/samples/' dataset '/'];

%subjects = [1102:1117 1119 1121:1124 1126:1130 1132];
subjects = [1105 1123];

for s = subjects 
    fprintf('--------------------\n');
    fprintf('Subject %d\n', s);

    load(['data/gt_' num2str(s) '.mat']);
    
    nis = numel(all_iids);
    nsegs = numel(all_segs{1});
    nsubj = 1; 
    
    load(['mat_log/BSDS_test13_' num2str(s) '.mat']);
    
    % samples
    N = 100;
    for j = 1:nis
        fprintf('Image %d\n', j);
        % prepare data
        iid = all_iids(j);
        name = num2str(iid);
        load([ucm2_dir name '.mat']); % ucm2
        ucm = ucm2(3:2:end, 3:2:end); % ucm
        segMap = bwlabel(ucm <= 0, 4); % seg
        img = imread([img_dir name '.jpg']); % img

        aftTree = all_aftTree{j,3};
        samples = post_sample(aftTree, N);
        ave_img = zeros(size(segMap));
        for n = 1:N
            %[PRI, VOI, labMap] = eval_seg(segMap, samples{n}, groundTruth);
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

            if n <= 2
                scratch = vis_seg2(labMap, img);
                imwrite(scratch, [smp_dir name '_sample_' num2str(n) '.png']);
            end
        end % n
        
        ave_img = ave_img / N;
        imwrite(ave_img*255, [smp_dir name '_sample_ave_bry.png']);
    end % j
end % s
