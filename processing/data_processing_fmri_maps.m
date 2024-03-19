% Create whole-brain fMRI connectivity maps seeded from channels

repo_path = 'Path_to\Manuscript_repository';
addpath('Path_to\wjn_toolbox_tsbinns');
addpath(genpath('Path_to\spm12'));
addpath(genpath('Path_to\leaddbs'));

folderpath_preprocessing = 'Path_to\Project\Preprocessing\Data\BIDS_01_Berlin_Neurophys';
folderpath_processing = 'Path_to\Project\Processing\Data\BIDS_01_Berlin_Neurophys';
folderpath_analysis = 'Path_to\Project\Analysis\Results\BIDS_01_Berlin_Neurophys\sub-multi\ses-multi';


%% Settings

coords_fpath = "ECoG_LFP_coords_projected.csv";
ecog_roi_radius = 5; % sphere radius (mm)
lfp_roi_radius = 3; % sphere radius (mm)

fmri_connectome = 'PPMI 74_15 (Horn 2017)_Patients';
connectome_type = '_func_seed_AvgR_Fz.nii';
seed_type = 'ECOG'; % create connectivity maps seeded from these channels
target_type = 'LFP'; % take values from connectivity maps at these channel regions

atlas = 'compound_atlas_HCPex_SUIT_ABGT.nii';
atlas_path = fullfile(repo_path, 'coherence\roi_atlases', atlas);

subjects = ["EL003", "EL004", "EL006", "EL007", "EL008", "EL009",...
    "EL010", "EL011", "EL012", "EL013", "EL014", "EL016", "EL017", ...
    "EL019", "EL020", "EL021", "EL022", "EL023"];


%% Create ROIs for each ECoG & LFP contact

coords = readtable(fullfile(folderpath_preprocessing, coords_fpath));
for a = 1:size(coords, 1)
    roi_fpath  = fullfile(folderpath_preprocessing, sprintf('sub-%s', string(coords.sub(a))), 'contact_ROIs',...
        strcat('sub-', string(coords.sub(a)), '_ROI_', string(coords.ch{a}), '.nii'));
    % take abs. to project to right hemisphere
    mni = [abs(coords.x(a)), coords.y(a), coords.z(a)];
    if startsWith(string(coords.ch{a}), "ECOG")
        roi_radius = ecog_roi_radius;
    elseif startsWith(string(coords.ch{a}), "LFP")
        roi_radius = lfp_roi_radius;
    else
        error("Channel type not recognised")
    end
    wjn_spherical_roi(roi_fpath, mni, roi_radius);
end
% ROIs consist of nifti files with ones at the contact and surrounding radius,
% and NaNs elsewhere


%% Generate the fMRI connectivity maps

lead_mapper
% 1: Seed and Output definition - "Manually choose seeds" (use ECoG & STN-LFP contacts)
% 2: Connectome definition - "PPMI 74_15 (Horn 2017) > Patients"
% 3: Command - "Connectivity map from seed"


%% Extract target ROI values from seed connectivity maps

entry_i = 1;
seed_names = {};
target_names = {};
con_values = [];
sub_ids = {};
for sub_i = 1:length(subjects)
    sub_id = sprintf('sub-%s', subjects(sub_i));

    % find seed connectome files
    connectome_path = fullfile(folderpath_processing, sub_id, 'connectomes', fmri_connectome);
    fnames = dir(connectome_path);
    connectome_fnames = {};
    use_i = 1;
    for file_i = 1:length(fnames)
        name = fnames(file_i).name;
        if contains(name, seed_type) && endsWith(name, connectome_type)
            connectome_fnames{use_i} = name;
            use_i = use_i + 1;
        end
    end
    connectome_fnames = string(connectome_fnames);
    
    % find target ROI files
    roi_path = fullfile(folderpath_preprocessing, sub_id, 'contact_ROIs');
    fnames = dir(roi_path);
    roi_fnames = {};
    use_i = 1;
    for file_i = 1:length(fnames)
        name = fnames(file_i).name;
        if contains(name, target_type)
            roi_fnames{use_i} = name;
            use_i = use_i + 1;
        end
    end
    roi_fnames = string(roi_fnames);

    % get ROI values from connectomes
    for con_i = 1:length(connectome_fnames)
        con_fname = char(connectome_fnames(con_i));
        for roi_i = 1:length(roi_fnames)
            roi_fname = char(roi_fnames(roi_i));
            spm_imcalc({fullfile(connectome_path, con_fname), fullfile(roi_path, roi_fname)},...
                'temp_corrected_roi_file.nii', 'i2'); % ensure files have same dimensions
            roi_nii = ea_load_nii('temp_corrected_roi_file.nii');
            con_nii = ea_load_nii(fullfile(connectome_path, con_fname));

            con_values(entry_i) = nanmean(con_nii.img(roi_nii.img > 0)); % average connectivity at ROI
            seed_names{entry_i} = extractBetween(con_fname, 'ROI_', connectome_type);
            target_names{entry_i} = extractBetween(roi_fname, 'ROI_', '.nii');
            sub_ids{entry_i} = extractAfter(sub_id, '-');

            entry_i = entry_i + 1;
        end
    end
end

seed_target_connectivity_table = table(con_values', string(seed_names)', string(target_names)', string(sub_ids)', ...
    'VariableNames', {'con_values', 'seed_names', 'target_names', 'sub'});
writetable(seed_target_connectivity_table, fullfile(folderpath_analysis, 'fmri_connectivity_maps_seed_target.csv'));


%% Extract atlas ROI values from connectivity maps

atlas_table = readtable([atlas_path(1:end-4), '.txt']);
atlas_table.Properties.VariableNames = {'Index', 'Name'};
atlas_nii = ea_load_nii(atlas_path);

entry_i = 1;
seed_names = {};
target_names = {};
con_values = [];
sub_ids = {};
for sub_i = 1:length(subjects)
    sub_id = sprintf('sub-%s', subjects(sub_i));

    % find seed connectome files
    connectome_path = fullfile(folderpath_processing, sub_id, 'connectomes', fmri_connectome);
    fnames = dir(connectome_path);
    connectome_fnames = {};
    use_i = 1;
    for file_i = 1:length(fnames)
        name = fnames(file_i).name;
        if contains(name, seed_type) && endsWith(name, connectome_type)
            connectome_fnames{use_i} = name;
            use_i = use_i + 1;
        end
        if contains(name, target_type) && endsWith(name, connectome_type)
            connectome_fnames{use_i} = name;
            use_i = use_i + 1;
        end
    end
    connectome_fnames = string(connectome_fnames);
    
    % get ROI values from connectomes
    for con_i = 1:length(connectome_fnames)
        con_fname = char(connectome_fnames(con_i));
        spm_imcalc({atlas_path, fullfile(connectome_path, con_fname)},...
            'temp_corrected_roi_file.nii', 'i2'); % ensure files have same dimensions
        con_nii = ea_load_nii('temp_corrected_roi_file.nii');
        for roi_i = 1:length(atlas_table.Index)
            con_values(entry_i) = nanmean(con_nii.img(atlas_nii.img == atlas_table.Index(roi_i))); % average connectivity at ROI
            seed_names{entry_i} = extractBetween(con_fname, 'ROI_', connectome_type);
            target_names{entry_i} = atlas_table.Name{roi_i};
            sub_ids{entry_i} = extractAfter(sub_id, '-');

            entry_i = entry_i + 1;
        end
    end
end

seed_atlas_table = table(con_values', string(seed_names)', string(target_names)', string(sub_ids)', ...
    'VariableNames', {'con_values', 'seed_names', 'target_names', 'sub'});
writetable(seed_atlas_table, fullfile(folderpath_analysis, 'fmri_connectivity_maps_seed_atlas.csv'));