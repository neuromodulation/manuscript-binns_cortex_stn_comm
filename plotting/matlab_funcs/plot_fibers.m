function [cs,reps] = plot_fibers(fc,c,lw,ind,reps)

if ~exist('ind','var')
    ind = [];
end

if ~exist('lw','var')
    lw = .1;
end

% reps is a repetition of each index - allows you to colour fibers
% according to their density for example in group analyses

if exist('reps','var')
   limits = reps./max(reps);
end

if ischar(fc)
    fname = fc;
    clear fc
    load(fname)   
        if ~exist('fc','var')
            if ~isempty(ind)
                ids = ind;
                disp('replacing file fiber indices with custom ones');
            end
            for a = 1:length(ids)
                fc{a} = fibers(fibers(:,4)==ids(a),:);
            end
           
        end
elseif ~iscell(fc)
    fibers = fc;
    i = unique(fibers(:,4));
    fc={};
    for a =1:length(i)
        fc{a} = fibers(fibers(:,4)==i(a),:);
    end
end

if size(c,1)>1
    % nx=linspace(1,size(c,1),length(fc));
    % c = [interp1(1:size(c,1),c(:,1),nx'),interp1(1:size(c,1),c(:,2),nx'),interp1(1:size(c,1),c(:,3),nx')];
    if size(c, 1) ~= size(fc, 2)
        error('number of colours does not match number of fibres being plotted')
    end
elseif size(c,1)==1
    c = repmat(c,[length(fc),1]);  
end

cs = [];
for a = 1:length(fc)
    [x,y,z]=tubeplot([fc{a}(:,1),fc{a}(:,2),fc{a}(:,3)]',lw,30);
    h(a) = surf(x,y,z,'edgecolor','none','FaceColor', c(a, :), 'FaceAlpha',0.8);
%     set(h,'color','k');
%     if a>60
%     keyboard;
%     end
    %h = patch(fc{a}(:,1),fc{a}(:,2),fc{a}(:,3),'linewidth',.1);
    %set(h,'FaceAlpha',0.5,'LineWidth',0.1);
    hold on
end
cs = unique(sort(cs),'rows');
axis equal 
axis off