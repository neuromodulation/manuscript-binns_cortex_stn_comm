%% Plot hyperdirect fibres coloured according to MIC patterns (Figure 4a)

repo_path = 'Path_to\Manuscript_repository';
funcs_fpath = fullfile(repo_fpath, 'plotting\matlab_funcs');
addpath(funcs_fpath);
addpath('Path_to\wjn_toolbox_tsbinns');
addpath(genpath('Path_to\spm12'));
addpath(genpath('Path_to\leaddbs'));

folderpath_analysis = 'Path_to\Project\Analysis\Results\BIDS_01_Berlin_Neurophys\sub-multi\ses-multi';

%% Load the spatial pattern values and plot fibres

fband = 'high_beta';  % low_beta, high_beta

results = readtable(fullfile(folderpath_analysis, 'mic_patterns_summed_hyperdirect_fibres-MedOffOn.csv'));
fband_results = table2array(results(:, sprintf('%s_weights', fband)));

fibre_atlas = load(fullfile(coherence_fpath, 'coherence\fibre_atlases\holographic_hyperdirect_filtered.mat')).fibers;

fibre_mask = ismember(fibre_atlas(:, 4), results.fibre_ids);
fibres_to_plot = fibre_atlas(fibre_mask, :);

fig = ea_mnifigure;

colors = colorlover(5);
wjn_plot_surface(fullfile('meshes', 'STN_bl.nii'), "#FF8000", 0, 1);

bound_results = vertcat(fband_results, [-1.5; 1.5]); % append weights range
[fibre_colours, colours_rgb] = map_values_to_cmap(bound_results, 'viridis');
plot_fibers(fibres_to_plot, fibre_colours(1:end-2, :), 0.3);


%% Save the figure

exportgraphics(gcf, fullfile(coherence_fpath, 'figures\mic_patterns_weighted_hyperdirect_fibres-MedOffOn.png'), 'Resolution', 1000);


%% Create & save colourmap

cmap_fig = figure;
cmap = colormap(colours_rgb);
cbar = colorbar('XTick', [0, 1], 'XTickLabel', {'Low', 'High'});
exportgraphics(gcf, fullfile(coherence_fpath, 'figures\mic_patterns_weighted_hyperdirect_fibres_cbar-MedOffOn.pdf'), 'ContentType', 'vector');
