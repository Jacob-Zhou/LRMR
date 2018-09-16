clc;clear;
close all;
r = [-1];
% lambdas = [0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9 10];
% a = [0.1, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2];
% a = [1.1];
% b = [0.35];
% b = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1];
% r = [1, 5, 10, 15, 20, 30, 40, 53];
% models = {'TRSMD_5'};
% pixs = [50, 100, 200, 300, 350, 400, 1000];
% pixs = [1000, 1250, 1500];
% models = {'SMD', 'BFMN_SMD', 'DNN_SMD', 'FNN_SMD'};
models = {'DNN_SMD'};
% datas = {'ECSSD'};
% datas = {'DUTOMRON', 'ECSSD' ,'ICOSEG', 'MSRA10K', 'SOD'};
datas = {'dataset1'};
addpath(genpath('Dependencies'));
gtSuffix = '.png';

for j = 1:length(datas)
    inputImgPath = strcat(['INPUT_IMG/', datas{j}]);
    imgFiles = imdir(inputImgPath);
    imgCount = length(imgFiles);
    % imgCount = 100;
    % randInd  = randperm(imgCount);
    
    gtPath = strcat(['GROUND_TRUTH/', datas{j}]);
    for i = 1:length(models)
        for k = 1:length(r)
            %for indl = 1:length(lambdas)
            %             for indA = 1:length(a)
            %                 for indB = 1:length(b)
            % for l = 1:length(pixs)
            % compile mex file
            %disp 'compile mex files ... ...'
            %compile;
            %disp 'compile done'
            
            %% Path settings
            if strcmp(models{i}, 'SMD') == 1
                if r(k) == -1
                    modelname = models{i};
                    % modelname = strcat([models{i}, '_', num2str(pixs(l))]);
                else
                    continue;
                end
            else
                if r(k) ~= -1
                % modelname = strcat([models{i}, '_', num2str(r(k)), '_', num2str(pixs(l))]);
                    modelname = strcat([models{i}, '_', num2str(r(k))]);
                % modelname = strcat([models{i}, '_', num2str(r(k)), '_a', num2str(a(indA) * 100), '_b', num2str(b(indB) * 100)]);
                % modelname = strcat([models{i}, '_', num2str(r(k)), '_test']);
                % modelname = models{i};
                else
                    modelname = strcat([models{i}, '_AUTO']);
                end
            end
            fprintf('%s %s\n', modelname, datas{j});                % input image path
            resSalPath = strcat(['SAL_MAP/', modelname]);                   % result path
            if ~exist(resSalPath, 'file')
                mkdir(resSalPath);
            end
            
            resSalEDataPath = strcat([resSalPath, '/', datas{j}]);
            
            if ~exist(resSalEDataPath, 'file')
                mkdir(resSalEDataPath);
            end
            
%             resSalSegDataPath = strcat([resSalPath, '/seg/', datas{j}]);
%             
%             if ~exist(resSalSegDataPath, 'file')
%                 mkdir(resSalSegDataPath);
%             end
            % addpath
            
            FeaturePath = strcat([ 'Feature/', datas{j}]);
            if ~exist(FeaturePath, 'file')
                mkdir(FeaturePath);
            end
            
            %% Parameter settings
            paras.alpha = 1.1;
            % paras.alpha = a(indA);
            paras.beta = 0.35;
            % paras.beta = b(indB);
            paras.delta = 0.05;
            setting.postProc = true;
            % paras.lambda = lambdas(indl);
            paras.lambda = 1;
            paras.r = r(k);
            % paras.pix = pixs(l);
            % paras.pix = 250;
            paras.pix = 250;
            paras.comp = 20;
            
            switch models{i}
                case 'SMD'
                    setting.model = 'smd';
                    setting.lapNorm = true;
                case 'LRR_SR'
                    setting.model = 'lrrsr';
                case 'LRR_SRB'
                    setting.model = 'lrrsr';
                case 'BSMD'
                    setting.model = 'bsmd';
                    setting.lapNorm = true;
                case 'TSMD'
                    setting.model = 'tsmd';
                    setting.lapNorm = true;
                case 'BFMN_SMD'
                    setting.model = 'bfmn_smd';
                case 'DNN_SMD'
                    setting.model = 'dnn_smd';
                case 'FNN_SMD'
                    setting.model = 'fnn_smd';    
                case 'prior'
                    setting.model = 'prior';
                case 'THEORY'
                    setting.lapNorm = true;
                    setting.model = 'theory';
            end
            % perform the context-based propagation technique metioned in Sec. 4.1 (Step 4).
            
            %% Calculate saliency using Structured Matrix Fractorization (SMF)
            
            
            times = zeros(imgCount, 1);
            pure_times = zeros(imgCount, 1);
            iters = zeros(imgCount, 1);
            parfor indImg = 1:imgCount
                % parfor indImg = 1:imgCount
                % read image
                imgPath = fullfile(inputImgPath, imgFiles(indImg).name);
                % imgPath = fullfile(inputImgPath, imgFiles(randInd(indImg)).name);
                img = {};
                img.RGB = imread(imgPath);
                img.name = imgPath((strfind(imgPath,'\')+1):end);
                salPath = fullfile(resSalPath, strcat(img.name(1:end-4), '.png'));
                salSegPath = fullfile(resSalPath, strcat(['seg/', img.name(1:end-4), '.png']));
                if (exist(salPath, 'file'))
                    continue;
                end
                
                % calculate saliency map via structured matrix decomposition
                % ComputeSaliency为计算显著性
                % img：为描述图片信息的数据结果{RBG, name}
                % paras：为模型中两个个tradeoff的α和β，还有affinity matrix中的δ
                % setting：是否开启context-based propagation
                indT = tic;
                [salMap, iter, pt] = ComputeSaliency(img, paras, setting);
                times(indImg) = toc(indT);
                % times(indImg) = time;
                iters(indImg) = iter;
                pure_times(indImg) = pt;
                
                % save saliency map
                imwrite(salMap,salPath);
                % imwrite(imbinarize(salMap, 0.4) .* im2double(img.RGB), salSegPath);
                
                %                     if mod(indImg, 100) == 0
                %                         fprintf('%s\n', num2str(indImg));
                %                     end
                close all;
            end
            averageTime = mean(times);
            averageIter = mean(iters);
            averagePureTime = mean(pure_times);
            
            resPath = strcat(['results/', datas{j}]);
            if ~exist(resPath,'file')
                mkdir(resPath);
            end
            
            TimePath = fullfile(resPath, ['Time', '_', modelname, '.mat']);
            save(TimePath, 'averageTime');
            fprintf('average time: %s\n', num2str(averageTime));
            fprintf('average pure time: %s\n', num2str(averagePureTime));
            %
            IterPath = fullfile(resPath, ['Iter', '_', modelname, '.mat']);
            save(IterPath, 'averageIter');
            fprintf('average iter: %s\n', num2str(averageIter));
            
%             % compute Precison-recall curve
%             [REC, PRE] = DrawPRCurve(resSalEDataPath, '.png', gtPath, gtSuffix, true, true, 'r');
%             PRPath = fullfile(resPath, ['PR', '_', modelname, '.mat']);
%             save(PRPath, 'REC', 'PRE');
%             fprintf('The precison-recall curve is saved in the file: %s \n', resPath);
%             
%             % compute ROC curve
%             thresholds = [0:1:255]./255;
%             [TPR, FPR] = CalROCCurve(resSalEDataPath, '.png', gtPath, gtSuffix, thresholds, 'r');
%             ROCPath = fullfile(resPath, ['ROC', '_', modelname, '.mat']);
%             save(ROCPath, 'TPR', 'FPR');
%             fprintf('The ROC curve is saved in the file: %s \n', resPath);
%             
%             % compute F-measure curve
%             setCurve = true;
%             [AvePreCurve, AveRecCurve, FmeasureCurve] = CalMeanFmeasure(resSalEDataPath, '.png', gtPath, gtSuffix, setCurve, 'r');
%             FmeasurePath = fullfile(resPath, ['AvgPRFCurve', '_', modelname, '.mat']);
%             save(FmeasurePath, 'AvePreCurve', 'AveRecCurve', 'FmeasureCurve');
%             fprintf('The F-measure curve is saved in the file: %s \n', resPath);

            % compute MAE
            MAE = CalMeanMAE(resSalEDataPath, '.png', gtPath, gtSuffix);
            MAEPath = fullfile(resPath, ['MAE', '_', modelname, '.mat']);
            save(MAEPath, 'MAE');
            fprintf('MAE: %s\n', num2str(MAE'));
            
            % compute WF
            Betas = [1];
            WF = CalMeanWF(resSalEDataPath, '.png', gtPath, gtSuffix, Betas);
            WFPath = fullfile(resPath, ['WF', '_', modelname, '.mat']);
            save(WFPath, 'WF');
            fprintf('WF: %s\n', num2str(WF'));
            
            % compute AUC
            AUC = CalAUCScore(resSalEDataPath, '.png', gtPath, gtSuffix);
            AUCPath = fullfile(resPath, ['AUC', '_', modelname, '.mat']);
            save(AUCPath, 'AUC');
            fprintf('AUC: %s\n', num2str(AUC'));
            
            % compute Overlap ratio fixed
            setCurve = false;
            overlapRatio = CalOverlap_Batch(resSalEDataPath, '.png', gtPath, gtSuffix, setCurve, '0');
            overlapFixedPath = fullfile(resPath, ['ORFixed', '_', modelname, '.mat']);
            save(overlapFixedPath, 'overlapRatio');
            fprintf('OR: %s\n', num2str(overlapRatio'));
            
%             % compute Overlap ratio curve
%             setCurve = true;
%             overlapRatioCurve = CalOverlap_Batch(resSalEDataPath, '.png', gtPath, gtSuffix, setCurve, '0');
%             overlapCurvePath = fullfile(resPath, ['ORCurve', '_', modelname, '.mat']);
%             save(overlapCurvePath, 'overlapRatioCurve');
%             fprintf('The Overlap ratio curve is saved in the file: %s \n', resPath);
%             fprintf('\n');
            %                 end
            %             end
            % end
        end
    end
end