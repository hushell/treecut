%

dataset = 'test';

ucm2_dir = ['./data/ucm2/' dataset '/'];
tree_dir = ['./output/ucm_trees/' dataset '/'];

all_files = dir(ucm2_dir);
mat       = arrayfun(@(x) ~isempty(strfind(x.name, '.mat')), all_files);
all_files = all_files(logical(mat));

for i = 1:numel(all_files)
    tic
    load([ucm2_dir '/' all_files(i).name], 'ucm2');
    [thisTree,thres_arr] = buildBSRTree2(ucm2);
    [~,name] = fileparts(all_files(i).name);
    save([tree_dir name '_tree.mat'], 'thisTree', 'thres_arr');
    fprintf([tree_dir name '_tree.mat' 'saved in %f sec.\n'], toc);
end
