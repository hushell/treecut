function segTree = tree_preprocess(processed, segTree, img, segMap)

if exist(processed, 'file')
    %fprintf('*** %s exists!\n', processed);
    %fprintf('>');
    load(processed);
    return
end

numLeafNodes = sum(unique(segMap) > 0);
numTotalNodes = size(segTree.kids,1);
segTree.numLeafNodes = numLeafNodes;
segTree.numTotalNodes = numTotalNodes;
img = im2double(img);
img1 = img(:,:,1);
img2 = img(:,:,2);
img3 = img(:,:,3);

%% leafs of each subtree: leafsUnder
numLeafsUnder = ones(numTotalNodes,1);
leafsUnder = cell(numTotalNodes,1);
% leafs
for s = 1:numLeafNodes
    leafsUnder{s} = s;
    segTree.leafsUnder{s} = leafsUnder{s};
end
% internals
for n = numLeafNodes+1:numTotalNodes
    kids = segTree.getKids(n);
    numLeafsUnder(n) = numLeafsUnder(kids(1))+numLeafsUnder(kids(2));
    leafsUnder{n} = [leafsUnder{kids(1)} leafsUnder{kids(2)}];
end
segTree.leafsUnder = leafsUnder;

%% loglik of each node
segTree.allik = zeros(numTotalNodes,1);
segTree.llik = zeros(numTotalNodes,1);
for i = 1:numTotalNodes
    pidx = ismember(segMap, leafsUnder{i}); % pixel index
    rgb = double([img1(pidx), img2(pidx), img3(pidx)]); % m x 3
    [allik,ll] = gaussian_loglik(rgb);
    segTree.allik(i) = allik;
    segTree.llik(i) = ll;
    %fprintf('node %d: allik = %e, llik = %e\n', i, allik, ll);
end
%fprintf('--------------\n');

%% save
save(processed, 'segTree');
