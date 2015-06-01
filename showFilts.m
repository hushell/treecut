function []=showFilts(ws,siz)
% modified from Yair Weiss's code

nFilts=size(ws,2);
N=ceil(sqrt(nFilts));
for i=1:nFilts
    subplot(N,N,i);
    show(reshape(ws(:,i),siz));
    axis off;
end

function []=show(im)
imagesc(im);colormap(gray);axis image

