% world2img.m
function imagePoints = world2img(intrinsics, R, t, worldPoints)
    % Convert world points to camera coordinates
    camPoints = (R * worldPoints' + t')';
    
    % Normalize camera coordinates
    camPoints = camPoints ./ camPoints(:, 3);
    
    % Convert to pixel coordinates using intrinsics
    imagePoints = [intrinsics.FocalLength(1) * camPoints(:, 1) + intrinsics.PrincipalPoint(1), ...
                   intrinsics.FocalLength(2) * camPoints(:, 2) + intrinsics.PrincipalPoint(2)];
end
