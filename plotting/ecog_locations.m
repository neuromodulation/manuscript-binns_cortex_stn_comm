%% Plot ECoG locations on cortical surface (Figure 2a)

repo_path = 'Path_to\Manuscript_repository';
meshes_path = fullfile(coherence_path, 'plotting', 'meshes');
addpath('Path_to\wjn_toolbox_tsbinns');
addpath(genpath('Path_to\spm12'));
addpath(genpath('Path_to\leaddbs'));

folderpath_preprocessing = 'Path_to\Project\Analysis\Preprocessing\Data\BIDS_01_Berlin_Neurophys';

%% Plot locations

% load coordinates
coords_table = readtable(fullfile(folderpath_preprocessing, 'ECoG_LFP_Coords.csv'));

% find and select ECoG channels
ecog_coords = [];
for idx = 1:height(coords_table)
    if contains(coords_table.ch(idx), 'ECOG')
        ecog_coords(end+1, :) = [coords_table.x(idx), coords_table.y(idx), coords_table.z(idx)];
    end
end

% project ECoG channels to surface
cortex_surf = export(gifti(fullfile(meshes_path, 'Automated Anatomical Labeling 3 (Rolls 2020).surf.gii')));
for idx = 1 : size(ecog_coords, 1)
    [mind(1, idx), i(1, idx)] = min(wjn_distance(cortex_surf.vertices, [abs(ecog_coords(idx, 1)), ecog_coords(idx, 2:3)]));
    ecog_coords(idx, :) = cortex_surf.vertices(i(idx), :) * 0.975;
end

% plot channels on mesh
cortex_mesh = load(fullfile(meshes_path, 'CortexHiRes.mat'));
plot_mesh.vertices = cortex_mesh.Vertices_rh;
plot_mesh.faces = cortex_mesh.Faces_rh;
figure
hold on
mesh = wjn_plot_surface(plot_mesh);
mesh.FaceAlpha = 0.4;
camlight
for idx = 1:length(ecog_coords)
    wjn_plot_colored_spheres(ecog_coords(idx, :), 1, 2, "red");
end

%% Save figure

exportgraphics(gcf, fullfile(coherence_path, 'figures', 'ecog_locations.pdf'), 'Resolution', 300);
