
function edges(fn)
    % open the sent image
    img = im2double(imread(fn));
    % manipulate the image
    img = edge(rgb2gray(img),'Canny');
    % save image to send it back
    [pathstr,name,ext] = fileparts(fn);
    new_fn = fullfile(pathstr, [name '_ok' ext]);
    imwrite(img, new_fn);
end