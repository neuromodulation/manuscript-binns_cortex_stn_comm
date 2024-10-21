%% Create interpolated SSD patterns and plot raw points (Figures 2b, S3, & S4)

repo_path = 'Path_to\Manuscript_repository';
addpath('Path_to\wjn_toolbox_tsbinns');
addpath(genpath('Path_to\spm12'));
addpath(genpath('Path_to\leaddbs'));

folderpath_analysis = 'Path_to\Project\Analysis\Results\BIDS_01_Berlin_Neurophys\sub-multi\ses-multi';

%% MED OFF and ON averaged (Figures 2b & S4)

fbands = ["alpha", "low_beta", "high_beta"];
regions = ["cortex", "STN"];
patterns_table = readtable(fullfile(folderpath_analysis, 'ssd_topographies_cortex_STN_MedOffOn.csv'));

for region_i = 1:length(regions)
    region = regions(region_i);
    region_table = patterns_table(ci(region, patterns_table.ch_regions), :);

    % define meshes to plot and interpolate to
    if strcmp(region, 'cortex')
        interp_mesh = fullfile(meshes_path, 'Automated Anatomical Labeling 3 (Rolls 2020).nii');
        cortex_mesh = load(fullfile(meshes_path, 'CortexHiRes.mat'));
        plot_mesh.vertices = cortex_mesh.Vertices_rh;
        plot_mesh.faces = cortex_mesh.Faces_rh;
    elseif strcmp(region, 'STN')
        interp_mesh = fullfile(meshes_path, 'STN_rh.nii');
        plot_mesh = export(gifti(fullfile(meshes_path, 'STN_rh.surf.gii')));
    else
        error('Regions to interpolate to must be cortex or STN.')
    end

    for fband_i = 1:length(fbands)
        fband = fbands(fband_i);
        fband_table = region_table(ci(fband, region_table.band_names), :);
        coords = [];
        patterns = [];

        % extract values from table
        for idx = 1:size(fband_table, 1)
            coords(idx,:) = eval(fband_table.ch_coords{idx});
            patterns(idx,1) = fband_table.ssd_topographies(idx);
        end

        % pin coordinates to surface (if cortex)
        if strcmp(region, 'cortex')
            cortex_surf = export(gifti(fullfile(meshes_path, 'Automated Anatomical Labeling 3 (Rolls 2020).surf.gii')));
            for idx = 1 : size(coords, 1)
                [mind(1, idx), i(1, idx)] = min(wjn_distance(cortex_surf.vertices, [abs(coords(idx, 1)), coords(idx, 2:3)]));
                coords(idx, :) = cortex_surf.vertices(i(idx), :) * 0.975;
            end
        end

        % interpolate patterns to mesh
        heatmap_fname = sprintf('SSD_%s_%s_Med.nii', region, fband);
        wjn_heatmap(heatmap_fname, coords, patterns - min(patterns), interp_mesh, [4, 4, 4])
        % (above saves file in current directory, so move to figures directory)
        movefile(heatmap_fname, sprintf('..\\figures\\%s', heatmap_fname));
        movefile("s"+heatmap_fname, sprintf('..\\figures\\s%s', heatmap_fname));
        movefile("z_s"+heatmap_fname, sprintf('..\\figures\\z_s%s', heatmap_fname));

        % plot locations of raw points
        figure
        hold on
        if strcmp(region, 'cortex')
            mesh = wjn_plot_surface(plot_mesh);
            mesh.FaceAlpha = 0.4;
            hold on
            wjn_plot_colored_spheres(coords, patterns, 2)
        else
            mesh = wjn_plot_surface(plot_mesh, [], 10);
            mesh.FaceAlpha = 0.2;
            mesh.EdgeColor = 'w';
            mesh.EdgeAlpha = 0.5;
            hold on
            wjn_plot_colored_spheres(coords, patterns, 0.25)
            view(5, 23)
        end
        camlight
        exportgraphics(gcf, fullfile(repo_path, 'figures', sprintf('SSD_%s_%s_Med.png', region, fband)), 'Resolution', 1000);
    end
end

%% MED OFF and ON separately (Figure S3a)

fbands = ["high_beta"];
regions = ["cortex"];
meds = ["off", "on"];
patterns_table = readtable(fullfile(folderpath_analysis, 'ssd_topographies_cortex_STN_MedOffOn.csv'));

for med_i = 1:length(meds)
    med = meds(med_i);
    med_table = patterns_table(ci(med, patterns_table.med), :);

    for region_i = 1:length(regions)
        region = regions(region_i);
        region_table = med_table(ci(region, med_table.ch_regions), :);
    
        % define meshes to plot and interpolate to
        if strcmp(region, 'cortex')
            interp_mesh = fullfile(meshes_path, 'Automated Anatomical Labeling 3 (Rolls 2020).nii');
            cortex_mesh = load(fullfile(meshes_path, 'CortexHiRes.mat'));
            plot_mesh.vertices = cortex_mesh.Vertices_rh;
            plot_mesh.faces = cortex_mesh.Faces_rh;
        elseif strcmp(region, 'STN')
            interp_mesh = fullfile(meshes_path, 'STN_rh.nii');
            plot_mesh = export(gifti(fullfile(meshes_path, 'STN_rh.surf.gii')));
        else
            error('Regions to interpolate to must be cortex or STN.')
        end
    
        for fband_i = 1:length(fbands)
            fband = fbands(fband_i);
            fband_table = region_table(ci(fband, region_table.band_names), :);
            coords = [];
            patterns = [];
    
            % extract values from table
            for idx = 1:size(fband_table, 1)
                coords(idx,:) = eval(fband_table.ch_coords{idx});
                patterns(idx,1) = fband_table.ssd_topographies(idx);
            end
    
            % pin coordinates to surface (if cortex)
            if strcmp(region, 'cortex')
                cortex_surf = export(gifti(fullfile(meshes_path, 'Automated Anatomical Labeling 3 (Rolls 2020).surf.gii')));
                for idx = 1 : size(coords, 1)
                    [mind(1, idx), i(1, idx)] = min(wjn_distance(cortex_surf.vertices, [abs(coords(idx, 1)), coords(idx, 2:3)]));
                    coords(idx, :) = cortex_surf.vertices(i(idx), :) * 0.975;
                end
            end
    
            % interpolate patterns to mesh (and save as nifti file)
            heatmap_fname = sprintf('SSD_%s_%s_Med%s.nii', region, fband, med);
            wjn_heatmap(heatmap_fname, coords, patterns - min(patterns), interp_mesh, [4, 4, 4])
            % (above saves file in current directory, so move to figures directory)
            movefile(heatmap_fname, sprintf('..\\figures\\%s', heatmap_fname));
            movefile("s"+heatmap_fname, sprintf('..\\figures\\s%s', heatmap_fname));
            movefile("z_s"+heatmap_fname, sprintf('..\\figures\\z_s%s', heatmap_fname));
            
            % plot locations of raw points
            figure
            hold on
            if strcmp(region, 'cortex')
                mesh = wjn_plot_surface(plot_mesh);
                mesh.FaceAlpha = 0.4;
                hold on
                wjn_plot_colored_spheres(coords, patterns, 2)
            else
                mesh = wjn_plot_surface(plot_mesh, [], 10);
                mesh.FaceAlpha = 0.2;
                mesh.EdgeColor = 'w';
                mesh.EdgeAlpha = 0.5;
                hold on
                wjn_plot_colored_spheres(coords, patterns, 0.25)
                view(5, 23)
            end
            camlight
            exportgraphics(gcf, fullfile(coherence_path, 'figures', sprintf('SSD_%s_%s_Med%s.png', region, fband, med)), 'Resolution', 1000);
        end
    end
end

%% STIM OFF and ON separately (Figure S3b)

fbands = ["high_beta"];
regions = ["cortex"];
stims = ["off", "on"];
patterns_table = readtable(fullfile(folderpath_analysis, 'ssd_topographies_cortex_STN_StimOffOn.csv'));

for stim_i = 1:length(stims)
    stim = stims(stim_i);
    stim_table = patterns_table(ci(stim, patterns_table.stim), :);

    for region_i = 1:length(regions)
        region = regions(region_i);
        region_table = stim_table(ci(region, stim_table.ch_regions), :);
    
        % define meshes to plot and interpolate to
        if strcmp(region, 'cortex')
            interp_mesh = fullfile(meshes_path, 'Automated Anatomical Labeling 3 (Rolls 2020).nii');
            cortex_mesh = load(fullfile(meshes_path, 'CortexHiRes.mat'));
            plot_mesh.vertices = cortex_mesh.Vertices_rh;
            plot_mesh.faces = cortex_mesh.Faces_rh;
        elseif strcmp(region, 'STN')
            interp_mesh = fullfile(meshes_path, 'STN_rh.nii');
            plot_mesh = export(gifti(fullfile(meshes_path, 'STN_rh.surf.gii')));
        else
            error('Regions to interpolate to must be cortex or STN.')
        end
    
        for fband_i = 1:length(fbands)
            fband = fbands(fband_i);
            fband_table = region_table(ci(fband, region_table.band_names), :);
            coords = [];
            patterns = [];
    
            % extract values from table
            for idx = 1:size(fband_table, 1)
                coords(idx,:) = eval(fband_table.ch_coords{idx});
                patterns(idx,1) = fband_table.ssd_topographies(idx);
            end
    
            % pin coordinates to surface (if cortex)
            if strcmp(region, 'cortex')
                cortex_surf = export(gifti(fullfile(meshes_path, 'Automated Anatomical Labeling 3 (Rolls 2020).surf.gii')));
                for idx = 1 : size(coords, 1)
                    [mind(1, idx), i(1, idx)] = min(wjn_distance(cortex_surf.vertices, [abs(coords(idx, 1)), coords(idx, 2:3)]));
                    coords(idx, :) = cortex_surf.vertices(i(idx), :) * 0.975;
                end
            end
    
            % interpolate patterns to mesh (and save as nifti file)
            heatmap_fname = sprintf('SSD_%s_%s_Stim%s.nii', region, fband, stim);
            wjn_heatmap(heatmap_fname, coords, patterns - min(patterns), interp_mesh, [4, 4, 4])
            % (above saves file in current directory, so move to figures directory)
            movefile(heatmap_fname, sprintf('..\\figures\\%s', heatmap_fname));
            movefile("s"+heatmap_fname, sprintf('..\\figures\\s%s', heatmap_fname));
            movefile("z_s"+heatmap_fname, sprintf('..\\figures\\z_s%s', heatmap_fname));

            % plot locations of raw points
            figure
            hold on
            if strcmp(region, 'cortex')
                mesh = wjn_plot_surface(plot_mesh);
                mesh.FaceAlpha = 0.4;
                hold on
                wjn_plot_colored_spheres(coords, patterns, 2)
            else
                mesh = wjn_plot_surface(plot_mesh, [], 10);
                mesh.FaceAlpha = 0.2;
                mesh.EdgeColor = 'w';
                mesh.EdgeAlpha = 0.5;
                hold on
                wjn_plot_colored_spheres(coords, patterns, 0.25)
                view(5, 23)
            end
            camlight
            exportgraphics(gcf, fullfile(coherence_path, 'figures', sprintf('SSD_%s_%s_Stim%s.png', region, fband, stim)), 'Resolution', 1000);
        end
    end
end

%% Plot interpolated patterns
% To plot interpolated patterns, use SurfIce