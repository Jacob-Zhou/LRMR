datas = {'DUTOMRON', 'ECSSD' ,'ICOSEG', 'MSRA10K', 'SOD'};

models = {'DNN_SMD_AUTO', 'FNN_SMD_AUTO'};
for i = 1:length(models)
    for j = 1:length(datas)
        % for k = 1:length(r)
        % modelname = strcat([models{i}, '_', num2str(r(k))]);
        modelname = models{i};
        gtPath = strcat(['GROUND_TRUTH/', datas{j}]);
        resSalPath = strcat(['SAL_MAP/', modelname, '/', datas{j}]);
        gtSuffix = '.png';
        resPath = strcat(['results/', datas{j}]);
        if ~exist(resPath,'file')
            mkdir(resPath);
        end
        fprintf([modelname, ' ']);
        fprintf([datas{j}, '\n']);
        
        % compute Precison-recall curve
        [REC, PRE] = DrawPRCurve(resSalPath, '.png', gtPath, gtSuffix, true, true, 'r');
        PRPath = fullfile(resPath, ['PR', '_', modelname, '.mat']);
        save(PRPath, 'REC', 'PRE');
        % plot(rec, pre, 'r', 'linewidth', 2);
        % saveas(gcf, fullfile(resPath, ['PR', '_', models{i}, '_', datas{j}, '.png']));
        fprintf('The precison-recall curve is saved in the file: %s \n', resPath);
        
        % compute ROC curve
        thresholds = [0:1:255]./255;
        [TPR, FPR] = CalROCCurve(resSalPath, '.png', gtPath, gtSuffix, thresholds, 'r');
        ROCPath = fullfile(resPath, ['ROC', '_', models{i}, '.mat']);
        save(ROCPath, 'TPR', 'FPR');
        % plot(FPR, TPR, 'r', 'linewidth', 2);
        % saveas(gcf, fullfile(resPath, ['ROC', '_', models{i}, '_', datas{j}, '.png']));
        fprintf('The ROC curve is saved in the file: %s \n', resPath);
        
        % compute F-measure curve
        setCurve = true;
        [AvePreCurve, AveRecCurve, FmeasureCurve] = CalMeanFmeasure(resSalPath, '.png', gtPath, gtSuffix, setCurve, 'r');
        FmeasurePath = fullfile(resPath, ['AvgPRFCurve', '_', modelname, '.mat']);
        save(FmeasurePath, 'AvePreCurve', 'AveRecCurve', 'FmeasureCurve');
        % plot(meanF, 'r', rand(1,3), 'linewidth', 2);
        % saveas(gcf, fullfile(resPath, ['FmeasureCurve', '_', models{i}, '_', datas{j}, '.png']));
        fprintf('The F-measure curve is saved in the file: %s \n', resPath);
        %
        % compute MAE
        MAE = CalMeanMAE(resSalPath, '.png', gtPath, gtSuffix);
        MAEPath = fullfile(resPath, ['MAE', '_', modelname, '.mat']);
        save(MAEPath, 'MAE');
        fprintf('MAE: %s\n', num2str(MAE'));
        
        % compute WF
        Betas = [1];
        WF = CalMeanWF(resSalPath, '.png', gtPath, gtSuffix, Betas);
        WFPath = fullfile(resPath, ['WF', '_', modelname, '.mat']);
        save(WFPath, 'WF');
        fprintf('WF: %s\n', num2str(WF'));
        
        % compute AUC
        AUC = CalAUCScore(resSalPath, '.png', gtPath, gtSuffix);
        AUCPath = fullfile(resPath, ['AUC', '_', modelname, '.mat']);
        save(AUCPath, 'AUC');
        fprintf('AUC: %s\n', num2str(AUC'));
        
        % compute Overlap ratio fixed
        setCurve = false;
        overlapRatio = CalOverlap_Batch(resSalPath, '.png', gtPath, gtSuffix, setCurve, '0');
        overlapFixedPath = fullfile(resPath, ['ORFixed', '_', modelname, '.mat']);
        save(overlapFixedPath, 'overlapRatio');
        fprintf('overlapRatio: %s\n', num2str(overlapRatio'));
        
        % compute Overlap ratio curve
        setCurve = true;
        overlapRatioCurve = CalOverlap_Batch(resSalPath, '.png', gtPath, gtSuffix, setCurve, '0');
        overlapCurvePath = fullfile(resPath, ['ORCurve', '_', models{i}, '.mat']);
        save(overlapCurvePath, 'overlapRatioCurve');
        fprintf('The Overlap ratio curve is saved in the file: %s \n', resPath);
        fprintf('\n');
        % end
    end
end