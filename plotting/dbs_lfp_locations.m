%% Plot DBS leads in STN (Figure 2a)

repo_path = 'Path_to\Manuscript_repository';
addpath('Path_to\wjn_toolbox_tsbinns');
addpath(genpath('Path_to\spm12'));
addpath(genpath('Path_to\leaddbs'));

folderpath_analysis = 'Path_to\Project\Analysis\Results\BIDS_01_Berlin_Neurophys\sub-multi\ses-multi';


%% Plot DBS leads

lead_group

% 1. Set group directory (can be any folder).
% 2. Under "General settings", set "Which subcortical atlas to use:" to "DISTAL Minimal (Ewert 2017)"
% 3. Under patients click "Add", go to folder containing lead reconstructions (e.g., "derivatives\LFP_leads"),
%    and choose the folders for each subject you want to visualise (each folder must contain an "ea_reconstruction.mat" file).
% 4. Select all entries of the "Patients" tab and under "3D Options" click "Visualize 3D".


% Y=-18 for anatomy slice looks good; select only the STN structure to be shown.


%% Save the figure

exportgraphics(gcf, fullfile(repo_path, 'figures', 'dbs_lead_locations.png'), 'Resolution', 1000);
