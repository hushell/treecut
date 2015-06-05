classdef tree
    properties
        nodeNames;
        nodeLabels=[];
        numLeafNodes;
        numTotalNodes;
        % parent pointers
        pp = [];
        % the parent pointers do not save which is the left and right child of each node, hence:
        % numNodes x 2 matrix of kids, [0 0] for leaf nodes
        kids = [];
        ucm = [];
        allik = [];
        llik = [];
        activeNodes = [];
        E = [];
        M = [];
        v = [];
        posterior = [];
        marginals = [];
        leafsUnder = {};
    end
    
    methods
        function id = getTopNode(obj)
            id = find(obj.pp==0);
        end
        
        function kids = getKids(obj,node)
            %kids = find(obj.pp==node);
            kids = obj.kids(node,:);
        end

        function p = getParent(obj,node)
            %kids = find(obj.pp==node);
            if node>0
                p = obj.pp(node);
            else
                p=-1;
            end
        end        
        
        %TODO: maybe compute leaf-node-ness once and then just check for it
        function l = isLeaf(obj,node)
            l = ~any(obj.pp==node);
        end        
        
        function plotTree(obj,postpone)
            %TREEPLOT Plot picture of tree.
            %   TREEPLOT(p) plots a picture of a tree given a row vector of
            %   parent pointers, with p(i) == 0 for a root and labels on each node.
            %
            %   Example:
            %      myTreeplot([2 4 2 0 6 4 6],{'i' 'like' 'labels' 'on' 'pretty' 'trees' '.'})
            %   returns a binary tree with labels.
            %
            %   Copyright 1984-2004 The MathWorks, Inc.
            %   $Revision: 5.12.4.2 $  $Date: 2004/06/25 18:52:28 $
            %   Modified by Richard @ Socher . org to display text labels
            %   Modified by Shell Hu
            %   TODO: 1) visualize forest; 2) vis cuts; to check if
            %   multiple optimal solutions have the same number of cuts
                
            if nargin < 2
                postpone = zeros(length(obj.nodeNames));
            end
            
            p = obj.pp';
            [x,y,h]=treelayout(p);
            f = find(p~=0);
            ppf = p(f);
            X = [x(f); x(ppf); NaN(size(f))];
            Y = [y(f); y(ppf); NaN(size(f))];
            X = X(:);
            Y = Y(:);
            
            n = length(p);
            if n < 500,
                plot (x,y,'wo',X,Y,'-','color',[153/255 102/255 0],'linewidth',2);
            else
                plot (X, Y, 'r-');
            end;
            %(['height = ' int2str(h)]);
            %axis([0 1 0 1]);
            axis off
            
            if ~isempty(obj.nodeNames)
                for l=1:length(obj.nodeNames)

            		if postpone(l)
            		    pcolor = [1 1 0];
            		else
            		    %pcolor = [1 1 .6];
                        pcolor = [87/255 157/255 28/255];
            		end

                    if isnumeric(obj.nodeNames(l))
                        text(x(l),y(l),num2str(obj.nodeNames(l)),'Interpreter','none',...
                            'HorizontalAlignment','center','FontSize',8,'BackgroundColor',pcolor)
                    else
                        text(x(l),y(l),obj.nodeNames{l},'Interpreter','none',...
                            'HorizontalAlignment','center','FontSize',8,'BackgroundColor',pcolor)
                    end
                    if ~isempty(obj.nodeLabels)
                        if iscell(obj.nodeNames)
                            text(x(l),y(l),[labels{l} '(' obj.nodeLabels{l} ')'],'Interpreter','none',...
                                'HorizontalAlignment','center','FontSize',8,'BackgroundColor',pcolor)
                        else
                            % for numbers
                            if isnumeric(obj.nodeLabels(l))
%                                if isinteger(obj.nodeLabels(l))
                                     allL = obj.nodeLabels(:,l);
                                     allL = find(allL);
                                     if isempty(allL)
                                         text(x(l),y(l),[num2str(obj.nodeNames(l))],'Interpreter','none',...
                                         'HorizontalAlignment','center','FontSize',8,'BackgroundColor',pcolor)
                                     else
                                         text(x(l),y(l),[num2str(obj.nodeNames(l)) ' (' mat2str(allL) ')'],'Interpreter','none',...
                                             'HorizontalAlignment','center','FontSize',8,'BackgroundColor',pcolor)
                                     end
                                   
%                                else
%                                    text(x(l),y(l),[obj.nodeLabels(l) ' ' num2str(obj.nodeLabels(l),'%.1f') ],'Interpreter','none',...
%                                        'HorizontalAlignment','center','FontSize',8,'BackgroundColor',pcolor)
%                                end
                                 % change to font size 6 for nicer tree prints
                            else
                                text(x(l),y(l),[obj.nodeNames{l}],'Interpreter','none',...
                                    'HorizontalAlignment','center','FontSize',8,'BackgroundColor',pcolor)
                            end
                        end
                    end
                end
            end

        end % plotTree
        
        function plotForest(obj, g_thres, a_n)
            if nargin < 3
                a_n = obj.activeNodes;
            end
            
            if nargin < 2
                a_n = obj.activeNodes;
                g_thres = 1;
            end

            assert(length(a_n) == obj.numTotalNodes);
            colMap = hsv(sum(a_n));
            
            p = obj.pp';
            [x,y,h]=treelayout(p);
            f = find(p~=0);
            ppf = p(f);
            X = [x(f); x(ppf); NaN(size(f))];
            Y = [y(f); y(ppf); NaN(size(f))];
            X = X(:);
            Y = Y(:);
            
            % plot tree
            n = length(p);
            if n < 500,
                plot (x,y,'wo',X,Y,'-','color',[153/255 102/255 0],'linewidth',2);
            else
                plot (X, Y, 'r-');
            end;
            hold on 
            
            %% plot active nodes
            %i = 1;
            %nn = find(a_n == 1);
            %for n = nn'
            %    plot (x(n), y(n), 'LineStyle', 'o', 'LineWidth', 4, 'Color', colMap(i,:)); hold on
            %    i = i + 1;
            %end
            
            %% plot nodes corresponding to ucm threshold
            %if ~isempty(obj.ucm) 
            %    ucm_flag = zeros(obj.numLeafNodes,1);
            %    nn = find(obj.ucm == g_thres);
            %    nn = nn(end);
            %
            %    for n = nn:-1:1
            %        if any(ucm_flag(obj.leafsUnder{n}))
            %            continue
            %        end
            %        plot (x(n), y(n), 'LineStyle', '+', 'MarkerSize', 20, 'Color', 'k'); hold on
            %        ucm_flag(obj.leafsUnder{n}) = 1;
            %    end
            %end
            
            hold off

            %xlabel(['height = ' int2str(h)]);
            %axis([0 1 0 1]);
            axis off
            
            if ~isempty(obj.nodeNames)
                for l=1:length(obj.nodeNames)

                    if a_n(l) == 1
                        pcolor = [255/255 51/255 51/255];
                    else
                        pcolor = [87/255 157/255 28/255];
                    end

                    if isnumeric(obj.nodeNames(l))
                        text(x(l),y(l),num2str(obj.nodeNames(l)),'Interpreter','none',...
                            'HorizontalAlignment','center','FontSize',8,'BackgroundColor',pcolor)
                    else
                        text(x(l),y(l),obj.nodeNames{l},'Interpreter','none',...
                            'HorizontalAlignment','center','FontSize',8,'BackgroundColor',pcolor)
                    end
                    
                end
            end
        end % plotForest

        function plotTree3D(obj, img, segs, style)

            if nargin < 4
                style = 0;
            end

            numLeafNodes = obj.numLeafNodes;
            numTotalNodes = obj.numTotalNodes;
            nodeHeight = numLeafNodes;
            
            numLeafsUnder = ones(numLeafNodes,1);
            leafsUnder = cell(numLeafNodes,1);
            for s = 1:numLeafNodes
                leafsUnder{s} = s;
                nodeHeight(s) = 1;
            end
                
            for n = numLeafNodes+1:numTotalNodes
                kids = obj.getKids(n);
                numLeafsUnder(n) = numLeafsUnder(kids(1))+numLeafsUnder(kids(2));
                leafsUnder{n} = [leafsUnder{kids(1)} leafsUnder{kids(2)}];
                nodeHeight(n) = max(nodeHeight(kids(1)), nodeHeight(kids(2))) + 3;
            end
            
            nodeHeight = nodeHeight - 1;
            orderedKids = obj.kids;
            
            % generate centroids by using regionprop()
            lCent = getCentroidSuperpixels(segs,1);
            pCent = zeros(numTotalNodes-numLeafNodes, 2);
            pn = 1;
            for node = numLeafNodes+1:numTotalNodes
                leafIndex = leafsUnder{node};
                tsegs = zeros(size(segs));
                for li = 1:numel(leafIndex)
                    tsegs = tsegs + bsxfun(@eq, segs, leafIndex(li));
                end
                pCent(pn,:) = getCentroidSuperpixels(tsegs);
                pn = pn + 1;
            end
            Cent = [lCent; pCent];
            [xs,ys] = size(segs);
            Cent(:,2) = xs - Cent(:,2); % NOTE: the new coordinate system is centered at left-bottom corner
            
            % height limit
            hLimit = numTotalNodes;
            zLimit = 300;
            uz = zLimit / (hLimit-numLeafNodes);
            
            %run('~/working/software/vlfeat-0.9.16/toolbox/vl_setup.m');
            %[sx,sy]=vl_grad(double(segs), 'type', 'forward') ;
            %s = sx | sy;
            %s = bwmorph(s, 'clean', Inf);
            %s = bwmorph(s, 'thicken', 1.5);
            %s = bwmorph(s, 'bridge', Inf);
            %s = bwmorph(s, 'fill', Inf);
            %s = find(s) ;
            %imp = img ;
            %%imp([s s+numel(img(:,:,1)) s+2*numel(img(:,:,1))]) = 0;
            %imp(s) = 255;
            %imp(s+numel(img(:,:,1))) = 51;
            %imp(s+2*numel(img(:,:,1))) = 153;
            
            figure;
            surface(zeros(xs,ys),flipdim(img,1),...
               'FaceColor','texturemap',...
               'EdgeColor','none',...
               'CDataMapping','scaled');
            %colormap(summer)
            view(-35,17)
            xlabel('x'); ylabel('y'); zlabel('z');
            axis([0 ys 0 xs 0 zLimit])
            axis equal
            %axis image
            hold on;
            
            % TODO: support more colors
            colmap = [...
            102/255   51/255         0;... % 1 brown
            1.0000    0              1;... % 2 magenta
            0         0              1;... % 3 blue
            1         149/255   14/255;... % 4 Yellow    
            0/255     132/255  209/255;... % 5 sky blue
            153/255   51/255    102/255;...% 6 Bordeaux
            1.0000         0         0;... % 7 red
            ];

            %g_col = [0.4196    0.5569    0.1373];
            g_col = [0 153/255 0];
            %g_col = [87/255 157/255 28/255];
            
            % A+ \cup A: >= 1  A-: 0  
            if style == 1
                govern = obj.activeNodes .* 7;
            elseif style > 1
                govern = obj.activeNodes;
                colrs = 1:sum(govern);
                govern(govern >= 1) = colrs;
            else
                govern = zeros(numTotalNodes,1);
            end

            for node = numTotalNodes:-1:numLeafNodes+1
                kid1 = orderedKids(node,1);
                kid2 = orderedKids(node,2);

                if govern(node) >= 1 && style > 1
                    govern([kid1 kid2]) = govern(node); 
                end

                point = [Cent(node,:), nodeHeight(node)*uz];
                point1 = [Cent(kid1,:), nodeHeight(kid1)*uz];
                point2 = [Cent(kid2,:), nodeHeight(kid2)*uz];

                if govern(node) && style == 1
                    line3d(point, point1, '-', 2, g_col, g_col, 32); hold on;
                    line3d(point, point2, '-', 2, g_col, g_col, 32); hold on;
                    col = colmap(govern(node),:);
                    line3d(point, point, '-', 2, col, col, 32); hold on;
                elseif govern(node) && style > 1
                    col = colmap(govern(node),:);
                    line3d(point, point1, '-', 2, col, col, 32); hold on;
                    line3d(point, point2, '-', 2, col, col, 32); hold on;
                else
                    line3d(point, point1, '-', 2, g_col, g_col, 32); hold on;
                    line3d(point, point2, '-', 2, g_col, g_col, 32); hold on;
                end
            end
            
            for node = numLeafNodes:-1:1
                point = [Cent(node,:), nodeHeight(node)*uz];
                
                if govern(node)
                    col = colmap(govern(node),:);
                    line3d(point, point, '-', 2, col, col, 32); hold on;
                end
            end

            hold off;
            axis off;
            grid off;

        end % plotTree3D
        
    end % methods
end % class

function centroids = getCentroidSuperpixels(segs, inside)
if nargin < 2
    inside = 0;
end

s = regionprops(segs, 'Centroid');
centroids = cat(1, s.Centroid);

if inside
    pixlist = regionprops(segs, 'PixelList');

    for i = 1:length(pixlist)
        temp = bsxfun(@minus, centroids(i,:), pixlist(i).PixelList);
        temp = temp.^2;
        temp = sum(temp,2);
    
        if all(temp > 1)
            npix = size(pixlist(i).PixelList, 1);
            centroids(i,:) = pixlist(i).PixelList(ceil(npix/3),:);
        end
    end
end

end % getCentroidSuperpixels


function line3d(st, ed, varargin)

linespec = '-b';
linewid = 1;

if nargin > 2
    linespec = varargin{1}; % '-'
end
if nargin > 3
    linewid = varargin{2}; % 2
end
if nargin > 4
    %linespec = [linespec, '.'];
    linecol = varargin{3};
end
if nargin > 5
    markface = varargin{4};
    markedge = markface;
end
if nargin > 6
    marksize = varargin{5};
end

x1 = st(1); y1 = st(2); z1 = st(3);
x2 = ed(1); y2 = ed(2); z2 = ed(3);

a = [x1;x2];
b = [y1;y2];
c = [z1;z2];

if nargin <= 4
    plot3(a, b, c, linespec, 'LineWidth', linewid, 'Marker', '.');
elseif nargin == 5
    plot3(a, b, c, linespec, 'LineWidth', linewid, ...
        'Color', linecol, 'Marker', '.');
elseif nargin == 6
    plot3(a, b, c, linespec, 'LineWidth', linewid, 'Color', linecol, ...
        'Marker', '.', 'MarkerEdgeColor', markedge, 'MarkerFaceColor', markface);
elseif nargin >= 7
    plot3(a, b, c, linespec, 'LineWidth', linewid, 'Color', linecol, ...
        'Marker', '.', 'MarkerEdgeColor', markedge, 'MarkerFaceColor', markface, ...
        'MarkerSize', marksize);
end

end % line3d
