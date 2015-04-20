%

iids_train = load(fullfile('data/iids_train.txt'));
subjects = [1105, 1123];

all_iids = [];
all_segs = {};
all_uids = {};
for i = 1:numel(iids_train)
    iid = iids_train(i);
    [segs,uids] = readSegs('color',iid);
    
    disp(ismember(subjects, uids));
    if all(ismember(subjects, uids))
        msk = ismember(uids, subjects);
        all_iids = [all_iids; iid];
        all_segs = [all_segs; {segs(msk)}];
        all_uids = [all_uids; {uids(msk)}];
    end
    
    %save(['data/groundTruth/train_uid/' num2str(iid) '.mat'], 'segs', 'uids'); 
end

save('data/gt_1105_1123.mat', 'all_iids', 'all_segs', 'all_uids');