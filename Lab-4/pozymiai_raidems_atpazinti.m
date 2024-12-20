function pozymiai = pozymiai_raidems_atpazinti(pavadinimas, pvz_eiluciu_sk)
%%  pozymiai = pozymiai_raidems_atpazinti(pavadinimas, pvz_eiluciu_sk)
% Features = pozymiai_raidems_atpazinti(image_file_name, Number_of_symbols_lines)
% taikymo pavyzdys:
% pozymiai = pozymiai_raidems_atpazinti('test_data.png', 8); 
% example of function use:
% Features = pozymiai_raidems_atpazinti('test_data.png', 8);
%%
% Vaizdo su pavyzdþiais nuskaitymas | Read image with written symbols
V = imread(pavadinimas);
figure(12), imshow(V)
%% Perform segmentation of the symbols and write into cell variable 
% Convert RGB image to grayscale
V_pustonis = rgb2gray(V);
% Calculate a threshold value for binary conversion
slenkstis = graythresh(V_pustonis);
% Convert grayscale image to binary
V_dvejetainis = imbinarize(V_pustonis, slenkstis);
% Display the result
figure(1), imshow(V_dvejetainis)
% Find contours of objects in the binary image
V_konturais = edge(uint8(V_dvejetainis));
% Display the result
figure(2), imshow(V_konturais)
% Fill contours
se = strel('square', 7); % Structural element for filling
V_uzpildyti = imdilate(V_konturais, se); 
% Display the result
figure(3), imshow(V_uzpildyti)
% Fill holes
V_vientisi = imfill(V_uzpildyti, 'holes');
% Display the result
figure(4), imshow(V_vientisi)
% Label connected components and calculate properties
[O_suzymeti, Skaicius] = bwlabel(V_vientisi);
O_pozymiai = regionprops(O_suzymeti);
% Extract bounding box properties
O_ribos = reshape([O_pozymiai.BoundingBox], [4, Skaicius]);
% Extract centroid properties
O_centras = reshape([O_pozymiai.Centroid], [2, Skaicius])';
% Add object indices
O_centras(:, 3) = 1:Skaicius;
% Sort objects by x-coordinate
O_centras = sortrows(O_centras, 2);

% Calculate the number of symbols per row
raidziu_sk = round(Skaicius / pvz_eiluciu_sk);
for k = 1:pvz_eiluciu_sk
    startIdx = (k - 1) * raidziu_sk + 1;
    endIdx = min(k * raidziu_sk, Skaicius); % Prevent out-of-bounds error
    O_centras(startIdx:endIdx, :) = sortrows(O_centras(startIdx:endIdx, :), 3);
end

% Crop symbols based on bounding boxes
for k = 1:Skaicius
    objektai{k} = imcrop(V_dvejetainis, O_ribos(:, O_centras(k, 3)));
end
% Display cropped symbols
figure(5),
for k = 1:Skaicius
   subplot(pvz_eiluciu_sk, raidziu_sk, k), imshow(objektai{k})
end

% Remove extra spaces around symbols
for k = 1:Skaicius
    V_fragmentas = objektai{k};
    % Eliminate white columns
    stulpeliu_sumos = sum(V_fragmentas, 1);
    V_fragmentas(:, stulpeliu_sumos == size(V_fragmentas, 1)) = [];
    % Eliminate white rows
    eiluciu_sumos = sum(V_fragmentas, 2);
    V_fragmentas(eiluciu_sumos == size(V_fragmentas, 2), :) = [];
    objektai{k} = V_fragmentas;
end

% Display cleaned symbols
figure(6),
for k = 1:Skaicius
   subplot(pvz_eiluciu_sk, raidziu_sk, k), imshow(objektai{k})
end

%% Normalize all symbols to 70x50
for k = 1:Skaicius
    V_fragmentas = objektai{k};
    V_fragmentas_7050 = imresize(V_fragmentas, [70, 50]);
    % Divide each image into 10x10 segments
    for m = 1:7
        for n = 1:5
            % Calculate average intensity for each 10x10 segment
            startRow = (m - 1) * 10 + 1;
            endRow = m * 10;
            startCol = (n - 1) * 10 + 1;
            endCol = n * 10;
            Vid_sviesumas_eilutese = sum(V_fragmentas_7050(startRow:endRow, startCol:endCol), 1);
            Vid_sviesumas((m - 1) * 5 + n) = sum(Vid_sviesumas_eilutese);
        end
    end
    % Normalize brightness values to [0, 1]
    Vid_sviesumas = ((100 - Vid_sviesumas) / 100);
    % Transform features into a column vector
    pozymiai{k} = Vid_sviesumas(:);
end
end
