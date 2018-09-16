clc;clear;
close all;
models = {'TSMD_AUTO', 'BSMD_AUTO', 'BFMN_SMD_AUTO', 'DNN_SMD_AUTO', 'FNN_SMD_AUTO'};
datas = {'DUTOMRON', 'ECSSD' ,'ICOSEG', 'MSRA10K', 'SOD'};
addpath(genpath('Dependencies'));
gtSuffix = '.png';

for j = 1:length(datas)
    inputImgPath = strcat(['INPUT_IMG/', datas{j}]);
    imgFiles = imdir(inputImgPath);
    imgCount = length(imgFiles);
    for i = 1:length(models)
        modelname = models{i};
        resSalPath = strcat(['SAL_MAP/', modelname]);
        resPath = strcat(['SAL_MAP/', modelname, '/seg']);
        
        if ~exist(resPath, 'file')
            mkdir(resPath);
        end
        
        resSalSegDataPath = strcat([resPath, '/', datas{j}]);
            
        if ~exist(resSalSegDataPath, 'file')
                mkdir(resSalSegDataPath);
        end
       
        parfor indImg = 1:imgCount
            imgPath = fullfile(inputImgPath, imgFiles(indImg).name);
            imgName = imgPath((strfind(imgPath,'\')+1):end);
            resSegPath = fullfile(resPath, strcat(imgName(1:end-4), '.png'));
            salPath = fullfile(resSalPath, strcat(imgName(1:end-4), '.png'));
            inputImg = imread(imgPath);
            salImg = imread(salPath);
            salMap = im2double(inputImg) .* im2double(salImg);
            % save saliency map
            imwrite(salMap,resSegPath);
            close all;
        end
    end
end