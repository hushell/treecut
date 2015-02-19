function segLabels = shuffle_labels(segLabels)
% make labels less continuous

n = numel(segLabels);
[newLabels,~,idx] = unique(segLabels);

for i = 1:length(newLabels)
    newl = newLabels(i) + randi([1,n]);
    while ismember(newl,newLabels)
        newl = newl + randi([1,n]);
    end
    newLabels(i) = newl;
end

segLabels = newLabels(idx);
