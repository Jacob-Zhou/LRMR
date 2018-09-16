function [featMap] =  featMapRGBShow(feat,labelMap) 
[height, width] = size(labelMap);
featMap = zeros(height, width, 3);
if min(min(labelMap)) == 0
    labelMap = labelMap + 1;
end
feat = mapminmax(feat', 0, 0.7)';
%feat = feat .* 0.7;
for i=1:height
    for j=1:width
        label = labelMap(i,j);
        featMap(i,j, 1) = 0.7 - feat(label);
        featMap(i,j, 2) = 0.9;
        featMap(i,j, 3) = 0.9;
    end
end
featMap = hsv2rgb(featMap);
% imshow(featMap,[]);


