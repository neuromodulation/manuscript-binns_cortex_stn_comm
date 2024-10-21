%% Plot indirect pathway nuclei coloured according to fMRI-MIC LME coefficients (Figure 4b)

repo_path = 'Path_to\Manuscript_repository';
funcs_fpath = fullfile(repo_fpath, 'plotting\matlab_funcs');
addpath(funcs_fpath);
addpath('Path_to\wjn_toolbox_tsbinns');
addpath(genpath('Path_to\spm12'));
addpath(genpath('Path_to\leaddbs'));

folderpath_analysis = 'Path_to\Project\Analysis\Results\BIDS_01_Berlin_Neurophys\sub-multi\ses-multi';

%% Load the fMRI connectivity values and extract the ECoG-BG indirect pathway nuclei values

fband = 'low_beta'; % low_beta, high_beta
rois = ["Putamen", "Caudate", "GPe", "STN"];
results = readtable(fullfile(folderpath_analysis, 'mic_patterns_fmri_lme_coeffs.csv'));

fband_results = results(strcmp(results.fband, fband), :);
roi_entries = [];
for roi_i = 1:length(rois)
    roi_entries(end+1) = find(strcmp(fband_results.roi, rois(roi_i)));
end
fband_results = results(roi_entries, :);
coeff_pad = (max(fband_results.coeff) - min(fband_results.coeff)) * 0.1;
fband_results.coeff(end+1) = max(fband_results.coeff) + coeff_pad;
fband_results.coeff(end+1) = min(fband_results.coeff) - coeff_pad;
[roi_colours, colours_rgb] = map_values_to_cmap(fband_results.coeff);
roi_colours = hex2rgb(roi_colours);

% plot putamen, GPe, and STN
fig = ea_mnifigure;
wjn_plot_surface(fullfile('meshes', 'striatum_rh.gii'), roi_colours(1, :), 0, 1);
wjn_plot_surface(fullfile('meshes', 'GPe_rh.gii'), roi_colours(3, :), 0, 1);
wjn_plot_surface(fullfile('meshes', 'STN_rh.gii'), roi_colours(4, :), 0, 1);
camzoom(0.11)
campos([-101.4880, -44, 6.6314]);
camdolly(0, 0.25, 0);

% plot caudate separately
fig_caudate = ea_mnifigure;
wjn_plot_surface(fullfile('meshes', 'striatum_rh.gii'), roi_colours(2, :), 0, 1);
camzoom(0.11)
campos([-101.4880, -44, 6.6314]);
camdolly(0, 0.25, 0);

fig_backdrop = ea_mnifigure;
camzoom(0.11)
campos([-101.4880, -44, 6.6314]);
camdolly(0, 0.25, 0);


%% Save the figure

exportgraphics(fig, fullfile(coherence_fpath, 'figures\mic_patterns_fmri_lme_coeffs-MedOffOn.png'), 'Resolution', 1000);
exportgraphics(fig_caudate, fullfile(coherence_fpath, 'figures\mic_patterns_fmri_lme_coeffs_caudate-MedOffOn.png'), 'Resolution', 1000);


%% Create & save colourmap

cmap_fig = figure;
cmap = colormap(colours_rgb);
cbar = colorbar('XTick', [0, 1], 'XTickLabel', [min(fband_results.coeff), max(fband_results.coeff)]);
exportgraphics(gcf, fullfile(coherence_fpath, 'figures\mic_patterns_fmri_lme_coeffs_cbar-MedOffOn.pdf'), 'ContentType', 'vector');
