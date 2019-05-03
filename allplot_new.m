% clc;clear;close all;

% models = {'BSMD_AUTO', 'TSMD_AUTO', 'BFMN_SMD_AUTO'};
% models = {'BSMD_AUTO', 'TSMD_AUTO'};
% models_name = {'BSMD', 'TSMD', 'BFMN-SMD'};
models = {'DNN_EMR_AUTO'};
models_name = {'DNN-EMR'};
for indM = 1:length(models)
    %% plot Overlap Ratio
    compareList22 = { models{indM}, 'SMD', 'WLRR', 'DRFI','RBD', 'HCT', 'MR',...
        'SVO',  'TD', 'GS', 'CB', 'GC', 'SEG',  ...
        'SS', };
    compareList22_name = { models_name{indM}, 'SMD', 'WLRR', 'DRFI', 'RBD', 'HCT', 'MR',...
        'SVO',  'TD', 'GS', 'CB', 'GC', 'SEG',  ...
        'SS', };
    compareList11 = { models{indM}, 'SMD', 'DSR','MC','HS', ...
        'PCA', 'LR','SLR','LRR','RC', 'SF',...
        'CA', 'FT','SR'};
    compareList11_name =    { models_name{indM}, 'SMD', 'DSR','MC','HS', ...
        'PCA', 'ULR','SLR','LRR','RC',  'SF',...
        'CA', 'FT','SR'};
    
    % compareList22 = {'SMD', 'TSMD_1', 'TSMD_5', 'TSMD_10', 'TSMD_15', 'TSMD_20', 'TSMD_30', 'TSMD_40', 'TSMD_53'};
    % compareList22_name = {'SMD', 'TSMD_1', 'TSMD_5', 'TSMD_10', 'TSMD_15', 'TSMD_20', 'TSMD_30', 'TSMD_40', 'TSMD_53'};
    %
    % compareList11 = {'SMD', 'BSMD_1', 'BSMD_5', 'BSMD_10', 'BSMD_15', 'BSMD_20', 'BSMD_30', 'BSMD_40', 'BSMD_53'};
    % compareList11_name = {'SMD', 'BSMD_1', 'BSMD_5', 'BSMD_10', 'BSMD_15', 'BSMD_20', 'BSMD_30', 'BSMD_40', 'BSMD_53'};
    
    for k=1:length(compareList11_name)
        tag11{k}=sprintf(compareList11_name{k});
    end
    for k=1:length(compareList22_name)
        tag22{k}=sprintf(compareList22_name{k});
    end
    datasetList = {'MSRA10K', 'DUTOMRON', 'ICOSEG', 'SOD', 'ECSSD'};
    % datasetList = {'DUTOMRON'};
    for j = 1:length(datasetList)
        resPath = strcat(['./Results/', datasetList{j}]);
        if ~exist(resPath, 'file')
            mkdir(resPath);
        end
        savePath = strcat(['./figures/', models_name{indM}, '/', datasetList{j}]);
        if ~exist(savePath,'file')
            mkdir(savePath);
        end
        colorList = [0.9333 0 0; 0 0 1; 0 0.75 1; 0 0.392 0;
            1 0.64 0; 0 0 0; 0.5804 0 0.8275; 0.5 0 0;
            0 1 0; 1 0.0784 0.5765; 0.4157 0.3529 0.8039; 0.35 0.35 0.35;
            0.5 0 0; 0 1 0; 1 0.0784 0.5765; 0.4157 0.3529 0.8039; 0.35 0.35 0.35;
            ];
        
        % shapeList ={'-';'-';'-';'-';'-.';'-.';'-.'; '-.';'-.'; '-.';'-.'; ':';':'; ':';':';':'};
        shapeList ={'-';'-';'-';'-.';'-.'; '-.';'-.'; '-.';'-.';'-.';'-.';'-.'; '-.';'-.'; '-.';'-.'};
        %% plot PR Curve
        
        for i = 1:length(compareList11)
            fileName =['PR_', compareList11{i},'.mat'];
            load(fullfile(resPath, fileName));
            figure(indM*100 + j*10 + 1);
            if i == 1
                hdl7 = plot(REC,PRE,shapeList{i},'color',colorList(i,:),'linewidth',2);
            else
                hdl7 = [hdl7 plot(REC,PRE,shapeList{i},'color',colorList(i,:),'linewidth',2)];
            end
            
            hold on;
        end
        fig = figure(indM*100 + j*10 + 1);
        grid on;
        switch j
            case 1
                axis([0.2,0.95, 0.4,1]);
            case 2
                axis([0.2,0.95, 0.15,0.75]);
            case 3
                axis([0.2,0.95, 0.45,0.92]);
            case 4
                axis([0.2,0.95, 0.25,0.83]);
            case 5
                axis([0.2,0.95, 0.3,0.92]);
        end
        set(gcf,'position',[100 100 1024 768]);
        set(gca,'Position',[.09 .12 .88 0.86]);
        set(gca,'xtick',[0:0.1:1],'FontSize', 22,'FontWeight','bold');
        set(gca,'ytick',[0.1:0.1:1],'FontSize', 22,'FontWeight','bold');
        
        xlabel('Recall','FontWeight','bold','FontSize',24);
        ylabel('Precision','FontWeight','bold','FontSize',24);
        
        tag = tag11; % ±Í«©

        legendflex(hdl7, tag, 'anchor', {'sw','sw'}, ...
            'buffer', [10 10], ...
            'ncol', 3, ...
            'fontsize', 20, ...
            'xscale', 1, ...
            'padding',[0,0,50], ...
            'box', 'on');
        
        for i = 1:length(compareList22)
            fileName =['PR_', compareList22{i},'.mat'];
            load(fullfile(resPath, fileName));
            figure(indM*100 + j*10 + 2);
            if i == 1
                hdl8 = plot(REC,PRE,shapeList{i},'color',colorList(i,:),'linewidth',2);
            else
                hdl8 = [hdl8 plot(REC,PRE,shapeList{i},'color',colorList(i,:),'linewidth',2)];
            end
            
            hold on;
        end
        fig = figure(indM*100 + j*10 + 2);
        grid on;
        switch j
            case 1
                axis([0.2,0.95, 0.4,1]);
            case 2
                axis([0.2,0.95, 0.15,0.75]);
            case 3
                axis([0.2,0.95, 0.45,0.92]);
            case 4
                axis([0.2,0.95, 0.25,0.83]);
            case 5
                axis([0.2,0.95, 0.3,0.92]);
        end % x÷·y÷·µƒ∑∂Œß
        
        set(gcf,'position',[100 100 1024 768]);
        set(gca,'Position',[.09 .12 .88 0.86]);
        set(gca,'xtick',[0:0.1:1],'FontSize', 22,'FontWeight','bold');
        set(gca,'ytick',[0.1:0.1:1],'FontSize', 22,'FontWeight','bold');
        
        xlabel('Recall','FontWeight','bold','FontSize',24);
        ylabel('Precision','FontWeight','bold','FontSize',24);
        tag = tag22; % ±Í«©
        legendflex(hdl8, tag, 'anchor', {'sw','sw'}, ...
            'buffer', [10 10], ...
            'ncol', 3, ...
            'fontsize', 20, ...
            'xscale', 1, ...
            'padding',[0,0,50], ...
            'box', 'on');
        
        fig = figure(indM*100 + j*10 + 1);
        str1 = strcat([savePath,'PRI_', datasetList{j},'.fig']);
        %    savefig([savePath,'PRI_', datasetList{j},'.fig']);
        
        print('-dpng', fullfile(savePath, strcat(['PRI_', datasetList{j},'.png'])));
        print('-dmeta', fullfile(savePath, strcat(['PRI_', datasetList{j},'.emf'])));
        savefig(fig, fullfile(savePath, strcat(['PRI_', datasetList{j},'.fig'])), 'compact');
        
        
        fig = figure(indM*100 + j*10 + 2);
        str2 =strcat([savePath,'PRII_', datasetList{j},'.fig']);
        %    savefig([savePath,'PRII_', datasetList{j},'.fig']);
        print('-dpng', fullfile(savePath, strcat(['PRII_', datasetList{j},'.png'])));
        print('-dmeta', fullfile(savePath, strcat(['PRII_', datasetList{j},'.emf'])));
        savefig(fig, fullfile(savePath, strcat(['PRII_', datasetList{j},'.fig'])), 'compact');
        
        %% plot Fmeasure Curve
        
        for i = 1:length(compareList11)
            fileName =['AvgPRFCurve_', compareList11{i},'.mat'];
            load(fullfile(resPath, fileName));
            figure(indM*100 + j*10 + 3);
            if i == 1
                hdl3 = plot([0:255],FmeasureCurve,shapeList{i},'color',colorList(i,:),'linewidth',2);
            else
                hdl3 = [hdl3 plot([0:255],FmeasureCurve,shapeList{i},'color',colorList(i,:),'linewidth',2)];
            end
            hold on;
        end
        fig = figure(indM*100 + j*10 + 3);
        grid on;
        switch j
            case 1
                axis([0,250,0.2,0.97]);
            case 2
                axis([0,250, 0.1,0.72]);
            case 3
                axis([0,250, 0.2,0.87]);
            case 4
                axis([0,250, 0.15,0.75]);
            case 5
                axis([0,250, 0.2,0.82]);
        end % x÷·y÷·µƒ∑∂Œß
        set(gcf,'position',[100 100 1024 768]);
        set(gca,'Position',[.09 .12 .88 0.86]);
        set(gca,'xtick',[0:50:255],'FontSize', 22,'FontWeight','bold');
        set(gca,'ytick',[0:0.1:1],'FontSize', 22,'FontWeight','bold');
        
        xlabel('Threshold','FontWeight','bold','FontSize',24);
        ylabel('F-measure','FontWeight','bold','FontSize',24);
        
        tag = tag11; % ±Í«©
        
        legendflex(hdl3, tag, 'anchor', {'n','n'}, ...
            'buffer', [0 -5], ...
            'ncol', 5, ...
            'fontsize', 20, ...
            'xscale', 1, ...
            'padding',[0,0,50], ...
            'rowHeight',22, ...
            'box', 'on');
        
        for i = 1:length(compareList22)
            fileName =['AvgPRFCurve_', compareList22{i},'.mat'];
            load(fullfile(resPath, fileName));
            figure(indM*100 + j*10 + 4);
            if i == 1
                hdl4 = plot([0:255],FmeasureCurve,shapeList{i},'color',colorList(i,:),'linewidth',2);
            else
                hdl4 = [hdl4 plot([0:255],FmeasureCurve,shapeList{i},'color',colorList(i,:),'linewidth',2)];
            end
            hold on;
        end
        fig = figure(indM*100 + j*10 + 4);
        grid on;
        switch j
            case 1
                axis([0,250,0.2,0.97]);
            case 2
                axis([0,250, 0.1,0.72]);
            case 3
                axis([0,250, 0.2,0.87]);
            case 4
                axis([0,250, 0.15,0.75]);
            case 5
                axis([0,250, 0.2,0.82]);
        end % x÷·y÷·µƒ∑∂Œß
        set(gcf,'position',[100 100 1024 768]);
        set(gca,'Position',[.09 .12 .88 0.86]);
        set(gca,'xtick',[0:50:255],'FontSize', 22,'FontWeight','bold');
        set(gca,'ytick',[0:0.1:1],'FontSize', 22,'FontWeight','bold');
        
        xlabel('Threshold','FontWeight','bold','FontSize',24);
        ylabel('F-measure','FontWeight','bold','FontSize',24);
        
        tag = tag22; % ±Í«©
        
        legendflex(hdl4, tag, 'anchor', {'n','n'}, ...
            'buffer', [0 -5], ...
            'ncol', 5, ...
            'fontsize', 20, ...
            'xscale', 1, ...
            'padding',[0,0,50], ...
            'rowHeight',22, ...
            'box', 'on');
        
        fig = figure(indM*100 + j*10 + 3);
        %        savefig( strcat([savePath,'AvgPRFCurveI_', datasetList{j},'.fig']));
        print('-dpng', fullfile(savePath, strcat(['AvgPRFCurveI_', datasetList{j},'.png'])));
        print('-dmeta', fullfile(savePath, strcat(['AvgPRFCurveI_', datasetList{j},'.emf'])));
        savefig(fig, fullfile(savePath, strcat(['AvgPRFCurveI_', datasetList{j},'.fig'])), 'compact');
        
        fig = figure(indM*100 + j*10 + 4);
        %        savefig( strcat([savePath,'AvgPRFCurveII_', datasetList{j},'.fig']));
        print('-dpng', fullfile(savePath, strcat(['AvgPRFCurveII_', datasetList{j},'.png'])));
        print('-dmeta', fullfile(savePath, strcat(['AvgPRFCurveII_', datasetList{j},'.emf'])));
        savefig(fig, fullfile(savePath, strcat(['AvgPRFCurveII_', datasetList{j},'.fig'])), 'compact');
        
        %         %% plot ROC Curve
        %         for i = 1:length(compareList11)
        %             fileName =['ROC_', compareList11{i},'.mat'];
        %             load(fullfile(resPath, fileName));
        %             figure(indM*100 + j*10 + 5);
        %             if i == 1
        %                 hdl5 = plot(FPR,TPR,shapeList{i},'color',colorList(i,:),'linewidth',2);
        %             else
        %                 hdl5 = [hdl5 plot(FPR,TPR,shapeList{i},'color',colorList(i,:),'linewidth',2)];
        %             end
        %             hold on;
        %         end
        %         figure(indM*100 + j*10 + 5);
        %         grid on;
        %         axis([0,1,0.5,1]);
        %         set(gcf,'position',[100 100 1024 768]);
        %         set(gca,'Position',[.09 .12 .88 0.86]);
        %         set(gca,'xtick',[0:0.1:1],'FontSize', 22,'FontWeight','bold');
        %         set(gca,'ytick',[0.1:0.1:1],'FontSize', 22,'FontWeight','bold');
        %
        %         xlabel('False Positive Rate','FontWeight','bold','FontSize',24);
        %         ylabel('True Positive Rate','FontWeight','bold','FontSize',24);
        %         tag = tag11;
        %         tagSet = gridLegend(hdl5, 3, tag,'FontSize', 20,'Orientation','Horizontal','location','north');
        %         set(tagSet, 'Position',[0.44 0.13 0.52 0.2]);
        %
        %         for i = 1:length(compareList22)
        %             fileName =['ROC_', compareList22{i},'.mat'];
        %             load(fullfile(resPath, fileName));
        %             figure(indM*100 + j*10 + 6);
        %             if i == 1
        %                 hdl6 = plot(FPR,TPR,shapeList{i},'color',colorList(i,:),'linewidth',2);
        %             else
        %                 hdl6 = [hdl6 plot(FPR,TPR,shapeList{i},'color',colorList(i,:),'linewidth',2)];
        %             end
        %             hold on;
        %         end
        %         figure(indM*100 + j*10 + 6);
        %         grid on;
        %         axis([0,1,0.5,1]);
        %         set(gcf,'position',[100 100 1024 768]);
        %         set(gca,'Position',[.09 .12 .88 0.86]);
        %         set(gca,'xtick',[0:0.1:1],'FontSize', 22,'FontWeight','bold');
        %         set(gca,'ytick',[0.1:0.1:1],'FontSize', 22,'FontWeight','bold');
        %
        %         xlabel('False Positive Rate','FontWeight','bold','FontSize',24);
        %         ylabel('True Positive Rate','FontWeight','bold','FontSize',24);
        %         tag = tag22;
        %         tagSet = gridLegend(hdl6,3, tag,'FontSize', 20,'Orientation','Horizontal','location','north');
        %         set(tagSet, 'Position',[0.44 0.13 0.52 0.2]);
        %
        %
        %         figure(indM*100 + j*10 + 5);
        %         print('-dpng', fullfile(savePath, strcat(['ROCI_', datasetList{j},'.png'])));
        %         print('-dmeta', fullfile(savePath, strcat(['ROCI_', datasetList{j},'.emf'])));
        %         figure(indM*100 + j*10 + 6);
        %         print('-dpng', fullfile(savePath, strcat(['ROCII_', datasetList{j},'.png'])));
        %         print('-dmeta', fullfile(savePath, strcat(['ROCII_', datasetList{j},'.emf'])));
        %
        %
        %         %% plot OR Curve
        %         for i = 1:length(compareList11)
        %             fileName =['ORCurve_', compareList11{i},'.mat'];
        %             load(fullfile(resPath, fileName));
        %             figure(indM*100 + j*10 + 7);
        %             if i == 1
        %                 hdl1 = plot([0:255],overlapRatioCurve,shapeList{i},'color',colorList(i,:),'linewidth',2);
        %             else
        %                 hdl1 = [hdl1 plot([0:255],overlapRatioCurve,shapeList{i},'color',colorList(i,:),'linewidth',2)];
        %             end
        %             hold on;
        %         end
        %         figure(indM*100 + j*10 + 7);
        %         grid on;
        %         switch j
        %               case 1
        %                 axis([0,250, 0.1,0.87]);
        %             case 2
        %                 axis([0,250, 0.1,0.57]);
        %             case 3
        %                 axis([0,250, 0.1,0.76]);
        %             case 4
        %                 axis([0,250, 0.1,0.58]);
        %             case 5
        %                 axis([0,250, 0.1,0.68]);
        %         end % x÷·y÷·µƒ∑∂Œß
        %         set(gcf,'position',[100 100 1024 768]);
        %         set(gca,'Position',[.09 .12 .88 0.86]);
        %         set(gca,'xtick',[0:50:255],'FontSize', 22,'FontWeight','bold');
        %         set(gca,'ytick',[0:0.1:1],'FontSize', 22,'FontWeight','bold');
        %
        %         xlabel('Threshold','FontWeight','bold','FontSize',24);
        %         ylabel('Overlap Ratio','FontWeight','bold','FontSize',24);
        %         tag = tag11;
        %         tagSet = gridLegend(hdl1, 5, tag,'FontSize', 20,'Orientation','Horizontal','location','north');
        %         set(tagSet, 'Position',[0.11 0.87 0.84 0.1]);
        %
        %
        %
        %         for i = 1:length(compareList22)
        %             fileName =['ORCurve_', compareList22{i},'.mat'];
        %             load(fullfile(resPath, fileName));
        %             figure(indM*100 + j*10 + 8);
        %             if i == 1
        %                 hdl2 = plot([0:255],overlapRatioCurve,shapeList{i},'color',colorList(i,:),'linewidth',2);
        %             else
        %                 hdl2 = [hdl2 plot([0:255],overlapRatioCurve,shapeList{i},'color',colorList(i,:),'linewidth',2)];
        %             end
        %             hold on;
        %         end
        %         figure(indM*100 + j*10 + 8);
        %         grid on;
        %         switch j
        %             case 1
        %                 axis([0,250, 0.1,0.87]);
        %             case 2
        %                 axis([0,250, 0.1,0.57]);
        %             case 3
        %                 axis([0,250, 0.1,0.76]);
        %             case 4
        %                 axis([0,250, 0.1,0.58]);
        %             case 5
        %                 axis([0,250, 0.1,0.68]);
        %         end % x÷·y÷·µƒ∑∂Œß
        %         set(gcf,'position',[100 100 1024 768]);
        %         set(gca,'Position',[.09 .12 .88 0.86]);
        %         set(gca,'xtick',[0:50:255],'FontSize', 22,'FontWeight','bold');
        %         set(gca,'ytick',[0:0.1:1],'FontSize', 22,'FontWeight','bold');
        %
        %         xlabel('Threshold','FontWeight','bold','FontSize',24);
        %         ylabel('Overlap Ratio','FontWeight','bold','FontSize',24);
        %         tag = tag22;
        %         tagSet = gridLegend(hdl2, 5, tag,'FontSize', 20,'Orientation','Horizontal','location','north');
        %         set(tagSet, 'Position',[0.11 0.87 0.84 0.1]);
        %
        %         figure(indM*100 + j*10 + 7);
        %         print('-dpng', fullfile(savePath, strcat(['OverlapRatioI_', datasetList{j},'.png'])));
        %         print('-dmeta', fullfile(savePath, strcat(['OverlapRatioI_', datasetList{j},'.emf'])));
        %         figure(indM*100 + j*10 + 8);
        %         print('-dpng', fullfile(savePath, strcat(['OverlapRatioII_', datasetList{j},'.png'])));
        %         print('-dmeta', fullfile(savePath, strcat(['OverlapRatioII_', datasetList{j},'.emf'])));
        close all;
    end
end