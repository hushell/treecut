function generate_subject_data(dataset, subjects)
%
if nargin < 2
    subjects = [1102:1117 1119 1121:1124 1126:1130 1132];
end

iids_train = load(['data/iids_' dataset '.txt']);

all_iids = [];
all_segs = {};
all_uids = {};
for i = 1:numel(iids_train)
    iid = iids_train(i);
    [segs,uids] = readSegs('color',iid);
    
    %disp(ismember(subjects, uids));
    if all(ismember(subjects, uids))
        msk = ismember(uids, subjects);
        all_iids = [all_iids; iid];
        all_segs = [all_segs; {segs(msk)}];
        all_uids = [all_uids; {uids(msk)}];
    end
    
    %save(['data/groundTruth/train_uid/' num2str(iid) '.mat'], 'segs', 'uids'); 
end

save(['data/gt_' num2str(subjects) '_' dataset '.mat'], 'all_iids', 'all_segs', 'all_uids');
