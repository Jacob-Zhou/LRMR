function featureimg = ExtractFeature(img)

%path(path, './matlabPyrTools');
%path(path, './matlabPyrTools/MEX');
%path(path, './edison_matlab_interface');

[height,width,channel] = size(img);
dim = 53;
featureimg = zeros(height,width,dim);
img = img.*255;
pos = 1;
%% Color
img1 = double(img);
img2 = rgb2hsv(double(img));
% 1~3 是该像素源图像
featureimg(:,:,pos:pos+2) = img1;
% 4 是色调
featureimg(:,:,pos+3) = (img2(:,:,1)-0.5).*255;
% 5 是饱和度
featureimg(:,:,pos+4) = img2(:,:,2).*255;

pos = pos+5;
%% Steerable Pyramid
% 4个方向, 3种比例; 提取12个特征
grayimg = double(rgb2gray(uint8(img)));
[pyr,pind] = buildSpyr(grayimg,3,'sp3Filters');
pyramids = getSpyr(pyr,pind);
pyrNum = size(pyramids,2);
for n = 1:pyrNum-1
    pyrImg = imresize(pyramids{n},[height, width], 'bicubic');
    for i = 1:height
        for j = 1:width   
            featureimg(i,j,pos) = pyrImg(i,j);
        end
    end
    pos = pos+1;
end
%% gabor filter
% 12个方向, 3种比例; 提取36个特征
scales = 3;
directions = 12;
[EO, BP] = gaborconvolve(grayimg, scales, directions, 6,2,0.65);

for wvlength = 1:scales
    for angle = 1:directions
        Aim = abs(EO{wvlength,angle});
        maxres = max(Aim(:));
        for i = 1:height
            for j = 1:width
                featureimg(i,j,pos) = Aim(i,j)/maxres*255;
            end
        end
        pos = pos+1;
    end
end

