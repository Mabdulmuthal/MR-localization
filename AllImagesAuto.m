% Load the images
clear; clc;
Fake_imageNames = dir(fullfile('C:\Localization\Scaled Image (All images)\Fake\',"*.png"));
Fake_imageNames = {Fake_imageNames.name}';

Synth_imageNames = dir(fullfile('C:\Localization\Scaled Image (All images)\Real\',"*.png"));
Synth_imageNames = {Synth_imageNames.name}';

CSVFileNames = dir(fullfile('C:\Localization\PixelCoordinates\',"*.csv"));
CSVFileNames = {CSVFileNames.name}';
CSVFileNames = natsort(CSVFileNames);

load('Camera_intrinsics.mat')

HololensPoses_text = readmatrix('Output_Edited_poses.txt');

for j = 1:100
%% Process Images
for i = 1:length(Fake_imageNames)
    Fake_image = rgb2gray(imread(fullfile('C:\Localization\Scaled Image (All images)\Fake\', Fake_imageNames{i})));
    Synth_image = rgb2gray(imread(fullfile('C:\Localization\Scaled Image (All images)\Real\', Synth_imageNames{i})));

    %Fake_image_colour = imread(fullfile('C:\Localization\Scaled Image (All images)\Fake\', Fake_imageNames{i}));
    %Synth_image_colour = imread(fullfile('C:\Localization\Scaled Image (All images)\Real\', Synth_imageNames{i}));

    % Detect and match features
    points_fake = detectKAZEFeatures(Fake_image);
    points_synth = detectKAZEFeatures(Synth_image);
    [features_fake, valid_points_fake] = extractFeatures(Fake_image, points_fake);
    [features_synth, valid_points_synth] = extractFeatures(Synth_image, points_synth);
    
    indexPairs = matchFeatures(features_fake, features_synth, "MatchThreshold", 20, "MaxRatio", 0.8);

    % **Skip image if too few matches**
    if size(indexPairs, 1) < 10
        warning(['Skipping image ', num2str(i), ' due to insufficient matches (', num2str(size(indexPairs,1)), ').']);
        continue;
    end

    % Extract the location of the keypoints on both real and synthetic images
    Unity_Coordinates = ceil(points_synth.Location(indexPairs(:,2),:));
    CycleGAN_Coordinates = ceil(points_fake.Location(indexPairs(:,1),:));

    % Read CSV file for 3D scene coordinates
    CSVFile = readmatrix(fullfile('C:\Localization\PixelCoordinates\', CSVFileNames{i}));
    SceneCoordinate = reshape(CSVFile, [428,760,3]);

    Synth_3D_Coordinates = [];
    valid_2D_points = [];
    
%% 

    % Extract 3D coordinates for matched 2D points
    for n = 1:length(Unity_Coordinates)
        x = Unity_Coordinates(n,1);
        y = Unity_Coordinates(n,2);

        if x > 0 && x <= size(SceneCoordinate, 2) && y > 0 && y <= size(SceneCoordinate, 1)
            % Extract 3D points and check if valid
            point3D = SceneCoordinate(y, x, :);
            if all(~isnan(point3D(:))) && all(point3D(:) ~= 0)  % Avoid NaN or zero values
                Synth_3D_Coordinates(end+1, :) = reshape(point3D, [1, 3]);
                valid_2D_points(end+1, :) = CycleGAN_Coordinates(n, :);
            end
        end
    end
%% 

    % **Skip image if too few valid 3D points**
    %if size(Synth_3D_Coordinates, 1) < 6
      %  warning(['Skipping image ', num2str(i), ' due to insufficient valid 3D points (', num2str(size(Synth_3D_Coordinates,1)), ').']);
      %  continue;
   % end

    % **Try to estimate camera pose and handle failures**
    try
        [R_cam, t_cam, inlierIdx] = estimateWorldCameraPose(double(valid_2D_points), Synth_3D_Coordinates, intrinsics, ...
            'MaxReprojectionError', 2, ...  
            'Confidence', 99, ...           
            'MaxNumTrials', 2000);           
         
        

        % **Skip image if too few inliers**
        numInliers = sum(inlierIdx);
        if numInliers < 10
            warning(['Skipping image ', num2str(i), ' due to insufficient inliers (', num2str(numInliers), ').']);
            continue;
        end

    catch
        warning(['Skipping image ', num2str(i), ' due to failure in estimateWorldCameraPose.']);
        continue;
    end

    % Display inlier count
    %disp(['Image ', num2str(i), ': Found ', num2str(numInliers), ' inliers']);

    % Retrieve HoloLens pose safely
    pose_index = i;
    if pose_index > size(HololensPoses_text, 1) || pose_index <= 0
        warning(['Skipping image ', num2str(i), ' due to invalid HoloLens pose index.']);
        continue;
    end

    HoloLenseImagePose = reshape(HololensPoses_text(pose_index, 2:end), [4,4])';
    R_transformed = HoloLenseImagePose(1:3, 1:3);
    T_transformed = HoloLenseImagePose(1:3, 4);

    % Adjust coordinate system
    R_transformed(1, 1) = -R_transformed(1, 1);
    R_transformed(1, 2) = -R_transformed(1, 2);
    R_transformed(2, 3) = -R_transformed(2, 3);
    R_transformed(3, 3) = -R_transformed(3, 3);
    T_transformed(1) = -T_transformed(1);
    
    % Compute pose differences
    angle_difference = (180/pi) * (rotm2eul(R_cam) - rotm2eul(R_transformed'));
    t_diff = sqrt(sum((t_cam - T_transformed').^2));

    % Compute reprojection error
    Inlier3DPoints = Synth_3D_Coordinates(inlierIdx, :);
    InlierImagePoints = valid_2D_points(inlierIdx,:);
    
    InitialProjectedPoints = world_to_pixel(intrinsics, Inlier3DPoints, R_transformed', T_transformed');
    FinalProjectPoints = world_to_pixel(intrinsics, Inlier3DPoints, R_cam, t_cam);


    CheckRMSE3 = sqrt(mean(sum((FinalProjectPoints - InitialProjectedPoints).^2, 2)));
    RMSE = sqrt(mean(sum((FinalProjectPoints-InitialProjectedPoints).^2,2)))
    RMSE_before = sqrt(mean(sum((InlierImagePoints-InitialProjectedPoints).^2,2)))
    RMSE_after = sqrt(mean(sum((InlierImagePoints-FinalProjectPoints).^2,2)))
    figure(3); imshow (Fake_image); hold on;
    scatter (InitialProjectedPoints (:,1), InitialProjectedPoints (:,2), 'red', 'filled');
    scatter (FinalProjectPoints(:,1), FinalProjectPoints(:,2), 'green', 'filled'); 
    hold off
    RMSE__before_stack{j,i} = RMSE_before;
    RMSE__after_stack{j,i} = RMSE_after;
    CheckRMSE2_stack{j,i} = CheckRMSE3;

%% % Visualization

    
 %    figure(3);
 %    imshow(Synth_image_colour);
 %    hold on;
 %    scatter(InitialProjectedPoints(:,1), InitialProjectedPoints(:,2), 'red', 'filled');
 %    scatter(FinalProjectPoints(:,1), FinalProjectPoints(:,2), 'green', 'filled'); 
 %    hold off;
 % 
 %    figure(1);
 %    imshow(Fake_image_colour);
 %    hold on;
 %    scatter(InitialProjectedPoints(:,1), InitialProjectedPoints(:,2), 'red', 'filled');
 %    scatter(FinalProjectPoints(:,1), FinalProjectPoints(:,2), 'green', 'filled'); 
 %    hold off;
 % 
 % 
 %    % Plot the inliers used for the estimation
 %    figure (2); showMatchedFeatures(Fake_image_colour, Synth_image_colour, valid_points_fake(indexPairs(inlierIdx,1)), valid_points_synth(indexPairs(inlierIdx,2)), 'montage');
 % 
 % 
 %  % Define save filename using image index
 %    save_folder = ('C:\Matlab files\PnP\Figures3\');
 %    save_filename = fullfile(save_folder, sprintf('Figure_%04d.jpg', i));
 %    saveas(figure(3), save_filename); 
 %    close(figure(3)); 
 % 
 % % Define save filename using image index
 %    save_folder = ('C:\Matlab files\PnP\Figures1\');
 %    save_filename = fullfile(save_folder, sprintf('Figure_%04d.jpg', i));
 %    saveas(figure(1), save_filename); 
 %    close(figure(1)); 
 %  % Define save filename using image index
 %     save_folder2 = ('C:\Matlab files\PnP\Figures2\');
 %     save_filename2 = fullfile(save_folder2, sprintf('Figure_%04d.jpg', i));
 %     saveas(figure(2), save_filename2); 
 %     close(figure(2)); 
%% 

    % Store results
   
    t_diff_stack{j,i} = t_diff;
    angle_stack{j,i} = angle_difference;

    disp(['Finished processing image ', num2str(i),'    Running     ', num2str(j)]);
end
 % **Stop after 100 loops**
    if j == 100
        disp('Completed 100 iterations. Stopping.');
        break;
    end
end