% Load image points from the .mat file
image_points_data = load('adjusted_coordinates.mat');  % Load the specified .mat file
if isfield(image_points_data, 'adjusted_coordinates')
    imagePoints = double(image_points_data.adjusted_coordinates); % Convert to double
else
    error('The .mat file does not contain the adjusted_coordinates field.');
end

% Load world points from the .mat file
world_points_data = load('matched_world_coordinates.mat');  % Load the specified .mat file
if isfield(world_points_data, 'matchedWorldCoordinates')
    worldPoints = double(world_points_data.matchedWorldCoordinates); % Convert to double
else
    error('The .mat file does not contain the matchedWorldCoordinates field.');
end

% Camera intrinsic parameters
focalLength = [599.8703, 599.8703];  % Focal length in pixels
principalPoint = [380, 214];  % Principal point in pixels
imageSize = [428, 760];  % Image size in pixels

% Create camera intrinsics object
intrinsics = cameraIntrinsics(focalLength, principalPoint, imageSize);

% Perform PnP camera pose estimation
[R, t, inliers] = estimateWorldCameraPose(imagePoints, worldPoints, intrinsics);

% Calculate the total number of correspondences
totalCorrespondences = size(imagePoints, 1);

% Calculate the total number of outliers
totalOutliers = totalCorrespondences - length(inliers);

% Display the results
disp('Estimated Rotation Matrix:');
disp(R);

disp('Estimated Translation Vector:');
disp(t);

disp(['Total Inliers: ', num2str(length(inliers))]);
disp(['Total Outliers: ', num2str(totalOutliers)]);
