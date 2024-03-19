function [data_colours, colours] = map_values_to_cmap(data, name)

if nargin == 1
    name = 'red_blue';
end

if strcmp(name, 'red_blue')
    colours_rgb_red = [ones(256, 1)*255, flip((0:255)'), flip((0:255)')] .* linspace(0.75, 1, 256)';
    colours_rgb_blue = [(0:255)', (0:255)', ones(256, 1)*255] .* flip(linspace(0.2, 0.7, 256))';
    colours_rgb = floor((colours_rgb_blue + colours_rgb_red * 2) / 3);
    colours_hex = rgb2hex(colours_rgb);
elseif strcmp(name, 'red_alpha')
    cmap = readtable('cmap_red_alpha.csv');
    colours_rgb = table2array(cmap) .* 255;
    colours_hex = rgb2hex(colours_rgb);
elseif strcmp(name, 'viridis')
    cmap = readtable('cmap_viridis.csv');
    colours_rgb = table2array(cmap) .* 255;
    colours_hex = rgb2hex(colours_rgb);
else
    error('Colourmap name is not recognised')
end

range_data = range(data);
min_data = min(data);
data_colours = colours_hex(floor(((data - min_data) / range_data) * 255) + 1, :);

colours = colours_rgb ./ 256;

end