%Read in images and do some processing

I1 = imread('parliament-left-small.jpg'); %644x639
I2 = imread('parliament-right-small.jpg');%600x567

imshow(I1);
pause;
imshow(I2);
pause;

img1_width = size(I1,2);
img2_width = size(I2,2);
img1_height = size(I1,1);
img2_height = size(I2,1);


larger_w = max(img1_width,img2_width);
larger_h = max(img1_height,img2_height);
smaller_w = min(img1_width,img2_width);
smaller_h = min(img1_height,img2_height);

%Sift needs images to be single precision and flat.
I1s = single(rgb2gray(I1));
I2s = single(rgb2gray(I2));

%Find keypoints and descriptors
[f1,d1] = vl_sift(I1s);
[f2,d2] = vl_sift(I2s);

%Keypoints are of shape 4xnumKeypoints. For each keypoint, rows 1 and 2 are the center of the 'disc', row 3 is scale, row 4 is orientation


%Get closest matches of descriptors
[matches,distances] = vl_ubcmatch(d1,d2);

%Throw out half of descriptors, take everything below median error
pruned_matches = [];
pruned_distances = [];
median_dist = median(distances);
for i=1:size(distances,2),
    if distances(i) <= median_dist,
        pruned_distances = [pruned_distances distances(i)];
        pruned_matches = [pruned_matches matches(:,i)];
    end
end
matches = pruned_matches;
distances = pruned_distances;
num_matches = size(matches,2);

best_acc = 0;

%Contains index of a point in image to be transformed
best_inliers = [];
best_inliers2 = [];
best_match = [];
best_error = 1e6;
for i=1:50,

    %Sample 3 points.
    n1 = randi(num_matches);
    n2 = randi(num_matches);
    n3 = randi(num_matches);

    %Make sure points are not the same
    if n1 == n2 | n2 == n3,
        continue;
    end
    cor1 = matches(:,n1);
    cor2 = matches(:,n2);
    cor3 = matches(:,n3);

    %Retrieve location of each feature, 1 for each image.
    p1a = [ ceil(f1(1:2,cor1(1)))];
    p1b = [ ceil(f2(1:2,cor1(2)))];
    p2a = [ ceil(f1(1:2,cor2(1)))];
    p2b = [ ceil(f2(1:2,cor2(2)))];
    p3a = [ ceil(f1(1:2,cor3(1)))];
    p3b = [ ceil(f2(1:2,cor3(2)))];

    Pa = [p1a p2a p3a];

    %Collinearity check
    det_Pa = [p1a(1) - p3a(2) p1a(2)-p3a(2); p2a(1) - p3a(1) p2a(2)-p3a(2)];
    det_Pb = [p1b(1) - p3b(2) p1b(2)-p3b(2); p2b(1) - p3b(1) p2b(2)-p3b(2)];
    if det(det_Pa) == 0 | det(det_Pb) == 0,
        continue;
    end
    Pa = [Pa; 1 1 1];
    Pb = [p1b p2b p3b];


    %Get pseudo-inverse of X, use it to estimate A
    Pa_pinv = pinv(Pa);
    A = Pb * Pa_pinv;
    A = [A; 0 0 1];

    %Apply the transformation to image 1
    T = maketform('affine',A');
    J = imtransform(I1s,T,'XYScale',1);

    acc = 0;
    inliers = [];
    inliers2 = [];
    match = [];

    n=100;
    t=0.5;

    %Randomly sample n features
    sampled_correlation_indices = randsample(size(matches,2),n);
    sampled_correlations = matches(:,sampled_correlation_indices);

    I1_keypoints = f1(1:2,sampled_correlations(1,:));
    I2_keypoints = f2(1:2,sampled_correlations(2,:));

    % I1_keypoints = [I1_keypoints ; ones(1,n)];
    T1_keypoints = A * [I1_keypoints ; ones(1,n)];
    T1_keypoints = T1_keypoints(1:2,:);

    % err = sqrt(((T1_keypoints(1,:) - I2_keypoints(1,:)) .^ 2) + ((T1_keypoints(2,:) - I2_keypoints(2,:)) .^ 2));

    err = sqrt(sum((T1_keypoints-I2_keypoints).^2));

    I1_keypoints = I1_keypoints(:,find(err < t));
    I2_keypoints = I2_keypoints(:,find(err < t));
    inliers = I1_keypoints;
    inliers2 = I2_keypoints;
    acc = size(inliers,2);

    if acc > best_acc,
        best_acc = acc;
        best_inliers = inliers;
        best_inliers2 = inliers2;
    end
end

best_inliers = [best_inliers ; ones(1,size(best_inliers,2))];
best_inliers_P = pinv(best_inliers);
A = (best_inliers2 * (pinv(best_inliers' * best_inliers) * best_inliers'));



ref = imref2d(size(I1s));
T = affine2d(A');
[IT,ST] = imwarp(I1,ref,T,'nearest');

large_img1 = zeros(size(IT,1),size(IT,2)+img2_width,3);
large_img2 = zeros(size(IT,1),size(IT,2)+img2_width,3);
large_img3 = zeros(size(IT,1),size(IT,2)+img2_width,3);

off_x = ST.XWorldLimits(1);
off_y = ST.YWorldLimits(1);

large_img1(1-off_y:size(I2,1)-off_y,1-off_x:size(I2,2)-off_x,:) = I2(:,:,:);
large_img2(1:size(IT,1) ,1:size(IT,2),:) = IT(:,:,:);

large_img3(:,:,1) = max(large_img1(:,:,1),large_img2(:,:,1));
large_img3(:,:,2) = max(large_img1(:,:,2),large_img2(:,:,2));
large_img3(:,:,3) = max(large_img1(:,:,3),large_img2(:,:,3));

imshow(uint8(large_img3),[]);

imwrite(uint8(large_img3),'affine.jpg');



function findInliers(Im1,Im2,n,t)
    img1_width = size(Im1,2);
    img2_width = size(Im2,2);
    img1_height = size(Im1,1);
    img2_height = size(Im2,1);
    
    
    larger_w = max(img1_width,img2_width);
    larger_h = max(img1_height,img2_height);
    smaller_w = min(img1_width,img2_width);
    smaller_h = min(img1_height,img2_height);

    sampled_points_x = randsample(smaller_w,n);
    sampled_points_y = randsample(smaller_h,n);

    acc = 0;
    inliers = [];

    pixel = [sampled_points_y(1) sampled_points_x(1)];

    Im2_patch = Im2(pixel(1)-3:pixel(1)+3,pixel(2)-3:pixel(2)+3);

end
