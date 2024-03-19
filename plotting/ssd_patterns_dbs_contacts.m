%% Plot SSD patterns for DBS contacts (Figure 2C)

repo_path = 'Path_to\Manuscript_repository';
funcs_fpath = fullfile(repo_fpath, 'plotting\matlab_funcs');
addpath(funcs_fpath);
addpath('Path_to\wjn_toolbox_tsbinns');
addpath(genpath('Path_to\spm12'));
addpath(genpath('Path_to\leaddbs'));

folderpath_analysis = 'Path_to\Project\Analysis\Results\BIDS_01_Berlin_Neurophys\sub-multi\ses-multi';

%% SSD patterns MED OFF (Figure 2C)

results = readtable(fullfile(folderpath_analysis, 'ssd_patterns_low_beta_combined-MedOffOn.csv'));

fig = ea_mnifigure;

colors = colorlover(5);
wjn_plot_surface(fullfile('meshes', 'STN_bl.nii'), "#FF8000", 0, 0.3);

off = strcmp(results.med, 'Off');

results_off = results.ssd_topographies(off);

[contact_colours, colours_rgb] = map_values_to_cmap(results_off, 'viridis');

radius = 0.5;
[x, y, z] = sphere(100);
for row=1:height(results_off)
    coords = str2num(results.ch_coords{row}) * 1000;
    ax_sphere = surf(coords(1) + (x.*radius), ...
                     coords(2) + (y.*radius), ...
                     coords(3) + (z.*radius));
    set(ax_sphere, 'LineStyle', 'none', 'facecolor', contact_colours(row, :), 'facealpha', 1);
end

ax_sphere = surf(12.58 + (x.*.7), ...
                 -13.41 + (y.*.7), ...
                 -5.87 + (z.*.7));
set(ax_sphere, 'LineStyle', 'none', 'facecolor', '#CA830F', 'facealpha', 1);

exportgraphics(gcf, fullfile(repo_fpath, 'figures\EL008_Power_locations.png'), 'Resolution', 300);
