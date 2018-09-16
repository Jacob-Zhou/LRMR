clc;clear;close all;

%% plot Overlap Ratio
% compareList11 = {'SMD', 'SMDnoLap', 'TriSMD', 'ESMD', 'ESMD_5','DRFI','RBD', 'HCT', 'MR',...
%     'SVO',  'TD', 'GS', 'CB', 'GC', 'SEG',  ...
%     'SS', 'LC', };
compareList11 = {'SMD', 'ESMD_5', 'TRSMD_5_1K'};
compareList11_name = {'SMD', 'TRSMD_5', 'TRSMD_51K'};
% compareList11_name = {'SMD', 'RSMD', 'TSMD', 'TRSMD_1', 'TRSMD','DRFI','RBD', 'HCT', 'MR',...
%     'SVO',  'TD', 'GS', 'CB', 'GC', 'SEG',  ...
%     'SS', 'LC', };
% compareList22 = {'SMD', 'SMDnoLap', 'TriSMD', 'ESMD', 'ESMD_5','DSR','MC','HS', ...
%     'PCA', 'LR','SLR','LRR','RC',  'SF',...
%     'CA', 'FT','SR'};
compareList22 = {'SMD'};
compareList22_name =    {'SMD', 'RSMD', 'TSMD', 'TRSMD_1', 'TRSMD', 'DSR','MC','HS', ...
    'PCA', 'ULR','SLR','LRR','RC',  'SF',...
    'CA', 'FT','SR'};

for k=1:length(compareList11)
    tag11{k}=sprintf(compareList11_name{k});
end
for k=1:length(compareList22_name)
    tag22{k}=sprintf(compareList22_name{k});
end
datasetList = {'DUTOMRON', 'ECSSD' ,'ICOSEG', 'MSRA10K', 'SOD'};
for j = 1:length(datasetList)
    resPath = strcat(['./Results/', datasetList{j}]);
    colorList = [0.9333 0 0; 0 0 1; 0 0.75 1; 0 0.392 0;
        1 0.64 0; 0 0 0; 0.5804 0 0.8275; 0.5 0 0;
        0 1 0; 1 0.0784 0.5765; 0.4157 0.3529 0.8039; 0.35 0.35 0.35;
        0.5 0 0; 0 1 0; 1 0.0784 0.5765; 0.4157 0.3529 0.8039; 0.35 0.35 0.35;
        ];
    
    shapeList ={'-';'-';'-';'-';'-';'-.';'-.';'-.'; '-.';'-.'; '-.';'-.'; ':';':'; ':';':';':'};
    
%     for i = 1:length(compareList11)
%         fileName =['ORCurve_', compareList11{i},'.mat'];
%         load(fullfile(resPath, fileName));
%         figure(j*10 + 1);
%         if i == 1
%             hdl1 = plot([0:255],overlapRatioCurve,shapeList{i},'color',colorList(i,:),'linewidth',2);
%         else
%             hdl1 = [hdl1 plot([0:255],overlapRatioCurve,shapeList{i},'color',colorList(i,:),'linewidth',2)];
%         end
%         hold on;
%     end
%     figure(j*10 + 1);
%     grid on;
%     axis([0,255,0,1]); % x��y��ķ��?
%     set(gca,'xtick',[0:50:255],'FontSize', 22,'FontWeight','bold');
%     set(gca,'ytick',[0:0.1:1],'FontSize', 22,'FontWeight','bold');
%     
%     xlabel('Threshold','FontWeight','bold','FontSize',22)%,'FontSize',15);
%     ylabel('Overlap Ratio','FontWeight','bold','FontSize',22)%,'FontSize',15);
%     title('Overlap Ratio Curve','FontWeight','bold','FontSize',24)%,'FontSize',15);
%     
%     tag = tag11; % ��ǩ
%     % tagSet = legend(tag,3);  % 3 = Lower left-hand corner
%     tagSet = gridLegend(hdl1, 5, tag,'location','eastoutside','Fontsize',14,'Box','on');
%     set(tagSet, 'Position',[0.05 0.65 0.95 0.4]);
%     % tagSet = columnlegend(2, tag);
%     %some variables
%     % numlines = length(str);
%     % numpercolumn = ceil(numlines/numcolumns);
%     % set( tagSet, 'FontSize', 14,'FontWeight','bold', 'Location','SouthWest');
%     
%     
%     
%     for i = 1:length(compareList22)
%         fileName =['ORCurve_', compareList22{i},'.mat'];
%         load(fullfile(resPath, fileName));
%         figure(j*10 + 2);
%         if i == 1
%             hdl2 = plot([0:255],overlapRatioCurve,shapeList{i},'color',colorList(i,:),'linewidth',2);
%         else
%             hdl2 = [hdl2 plot([0:255],overlapRatioCurve,shapeList{i},'color',colorList(i,:),'linewidth',2)];
%         end
%         hold on;
%     end
%     figure(j*10 + 2);
%     grid on;
%     axis([0,255,0,1]); % x��y��ķ��?
%     set(gca,'xtick',[0:50:255],'FontSize', 22,'FontWeight','bold');
%     set(gca,'ytick',[0:0.1:1],'FontSize', 22,'FontWeight','bold');
%     
%     xlabel('Threshold','FontWeight','bold','FontSize',22)%,'FontSize',15);
%     ylabel('Overlap Ratio','FontWeight','bold','FontSize',22)%,'FontSize',15);
%     title('Overlap Ratio Curve','FontWeight','bold','FontSize',24)%,'FontSize',15);
%     
%     tag = tag22; % ��ǩ
%     %tagSet = legend(tag,3);  % 3 = Lower left-hand corner
%     tagSet = gridLegend(hdl2, 5, tag,'FontSize', 14,'Orientation','Horizontal','location','north');
%     set(tagSet, 'Position',[0.05 0.65 0.95 0.4]);
%     %tagSet = columnlegend(2, tag);
%     %some variables
%     % numlines = length(str);
%     % numpercolumn = ceil(numlines/numcolumns);
%     % set( tagSet, 'FontSize', 14,'FontWeight','bold', 'Location','SouthWest');
%     
%     
%     savePath = './figures';
%     if ~exist(savePath,'file')
%         mkdir(savePath);
%     end
%     figure(j*10 + 1);
%     print('-depsc', fullfile(savePath, strcat(['OverlapRatioI_', datasetList{j},'.eps'])));
%     figure(j*10 + 2);
%     print('-depsc', fullfile(savePath, strcat(['OverlapRatioII_', datasetList{j},'.eps'])));
%     pause;
    
    %% plot Fmeasure Curve
    
    for i = 1:length(compareList11)
        fileName =['AvgPRFCurve_', compareList11{i},'.mat'];
        load(fullfile(resPath, fileName));
        figure(j*10 + 3);
        if i == 1
            hdl3 = plot([0:255],FmeasureCurve,shapeList{i},'color',colorList(i,:),'linewidth',2);
        else
            hdl3 = [hdl3 plot([0:255],FmeasureCurve,shapeList{i},'color',colorList(i,:),'linewidth',2)];
        end
        hold on;
    end
    figure(j*10 + 3);
    grid on;
    axis([0,255,0,1]); % x��y��ķ��?
    set(gca,'xtick',[0:50:255],'FontSize', 22,'FontWeight','bold');
    set(gca,'ytick',[0:0.1:1],'FontSize', 22,'FontWeight','bold');
    
    xlabel('Threshold','FontWeight','bold','FontSize',22)%,'FontSize',15);
    ylabel('F-measure','FontWeight','bold','FontSize',22)%,'FontSize',15);
    title('F-measure Curve','FontWeight','bold','FontSize',24)%,'FontSize',15);
    
    tag = tag11; % ��ǩ
    %tagSet = legend(tag,3);  % 3 = Lower left-hand corner
    tagSet = gridLegend(hdl3, 5, tag, 'FontSize', 14,'Orientation','Horizontal','location','north');
    set(tagSet, 'Position',[0.05 0.65 0.95 0.4]);
    %tagSet = columnlegend(2, tag);
    %some variables
    % numlines = length(str);
    % numpercolumn = ceil(numlines/numcolumns);
    % set( tagSet, 'FontSize', 14,'FontWeight','bold', 'Location','SouthWest');
    
    
    for i = 1:length(compareList22)
        fileName =['AvgPRFCurve_', compareList22{i},'.mat'];
        load(fullfile(resPath, fileName));
        figure(j*10 + 4);
        if i == 1
            hdl4 = plot([0:255],FmeasureCurve,shapeList{i},'color',colorList(i,:),'linewidth',2);
        else
            hdl4 = [hdl4 plot([0:255],FmeasureCurve,shapeList{i},'color',colorList(i,:),'linewidth',2)];
        end
        hold on;
    end
    figure(j*10 + 4);
    grid on;
    axis([0,255,0,1]); % x��y��ķ��?
    set(gca,'xtick',[0:50:255],'FontSize', 22,'FontWeight','bold');
    set(gca,'ytick',[0:0.1:1],'FontSize', 22,'FontWeight','bold');
    
    xlabel('Threshold','FontWeight','bold','FontSize',22)%,'FontSize',15);
    ylabel('F-measure','FontWeight','bold','FontSize',22)%,'FontSize',15);
    title('F-measure Curve','FontWeight','bold','FontSize',24)%,'FontSize',15);
    
    tag = tag22; % ��ǩ
    %tagSet = legend(tag,3);  % 3 = Lower left-hand corner
    tagSet = gridLegend(hdl4, 5, tag, 'FontSize', 14,'Orientation','Horizontal','location','north');
    set(tagSet, 'Position',[0.05 0.65 0.95 0.4]);
    %tagSet = columnlegend(2, tag);
    %some variables
    % numlines = length(str);
    % numpercolumn = ceil(numlines/numcolumns);
    % set( tagSet, 'FontSize', 14,'FontWeight','bold', 'Location','SouthWest');
    
    
    savePath = './figures';
    if ~exist(savePath,'file')
        mkdir(savePath);
    end
    figure(j*10 + 3);
    print('-depsc', fullfile(savePath, strcat(['AvgPRFCurveI_', datasetList{j},'.eps'])));
    figure(j*10 + 4);
    print('-depsc', fullfile(savePath, strcat(['AvgPRFCurveII_', datasetList{j},'.eps'])));
    
    
    %% plot ROC Curve
    
%     for i = 1:length(compareList11)
%         fileName =['ROC_', compareList11{i},'.mat'];
%         load(fullfile(resPath, fileName));
%         figure(j*10 + 5);
%         if i == 1
%             hdl5 = plot(FPR,TPR,shapeList{i},'color',colorList(i,:),'linewidth',2);
%         else
%             hdl5 = [hdl5 plot(FPR,TPR,shapeList{i},'color',colorList(i,:),'linewidth',2)];
%         end
%         hold on;
%     end
%     figure(j*10 + 5);
%     grid on;
%     axis([0,1,0,1]); % x��y��ķ��?
%     set(gca,'xtick',[0:0.1:1],'FontSize', 22,'FontWeight','bold');
%     set(gca,'ytick',[0.1:0.1:1],'FontSize', 22,'FontWeight','bold');
%     
%     xlabel('False Positive Rate','FontWeight','bold','FontSize',22)%,'FontSize',15);
%     ylabel('True Positive Rate','FontWeight','bold','FontSize',22)%,'FontSize',15);
%     title('Receiver Operating Characteristic (ROC)','FontWeight','bold','FontSize',24)%,'FontSize',15);
%     
%     tag = tag11; % ��ǩ
%     %tagSet = legend(tag,3);  % 3 = Lower left-hand corner
%     tagSet = gridLegend(hdl5, 2,tag,'FontSize', 14,'Orientation','Horizontal');
%     set(tagSet, 'Position',[0.575 0.03 0.36 0.4]);
%     %tagSet = columnlegend(2, tag);
%     %some variables
%     % numlines = length(str);
%     % numpercolumn = ceil(numlines/numcolumns);
%     % set( tagSet, 'FontSize', 14,'FontWeight','bold', 'Location','SouthWest');
%     
%     
%     for i = 1:length(compareList22)
%         fileName =['ROC_', compareList22{i},'.mat'];
%         load(fullfile(resPath, fileName));
%         figure(j*10 + 6);
%         if i == 1
%             hdl6 = plot(FPR,TPR,shapeList{i},'color',colorList(i,:),'linewidth',2);
%         else
%             hdl6 = [hdl6 plot(FPR,TPR,shapeList{i},'color',colorList(i,:),'linewidth',2)];
%         end
%         hold on;
%     end
%     figure(j*10 + 6);
%     grid on;
%     axis([0,1,0,1]); % x��y��ķ��?
%     set(gca,'xtick',[0:0.1:1],'FontSize', 22,'FontWeight','bold');
%     set(gca,'ytick',[0.1:0.1:1],'FontSize', 22,'FontWeight','bold');
%     
%     xlabel('False Positive Rate','FontWeight','bold','FontSize',22)%,'FontSize',15);
%     ylabel('True Positive Rate','FontWeight','bold','FontSize',22)%,'FontSize',15);
%     title('Receiver Operating Characteristic (ROC)','FontWeight','bold','FontSize',24)%,'FontSize',15);
%     
%     
%     tag = tag22; % ��ǩ
%     %tagSet = legend(tag,3);  % 3 = Lower left-hand corner
%     %set( tagSet, 'FontSize', 14,'FontWeight','bold');
%     tagSet = gridLegend(hdl6,2, tag,'FontSize', 14,'Orientation','Horizontal');
%     set(tagSet, 'Position',[0.575 0.03 0.36 0.4]);
%     
%     
%     savePath = './figures';
%     if ~exist(savePath,'file')
%         mkdir(savePath);
%     end
%     figure(j*10 + 5);
%     print('-depsc', fullfile(savePath, strcat(['ROCI_', datasetList{j},'.eps'])));
%     figure(j*10 + 6);
%     print('-depsc', fullfile(savePath, strcat(['ROCII_', datasetList{j},'.eps'])));
    
    
    
    %% plot PR Curve
    
    for i = 1:length(compareList11)
        fileName =['PR_', compareList11{i},'.mat'];
        load(fullfile(resPath, fileName));
        figure(j*10 + 7);
        if i == 1
            hdl7 = plot(REC,PRE,shapeList{i},'color',colorList(i,:),'linewidth',2);
        else
            hdl7 = [hdl7 plot(REC,PRE,shapeList{i},'color',colorList(i,:),'linewidth',2)];
        end
        
        hold on;
    end
    figure(j*10 + 7);
    grid on;
    axis([0.2,0.95, 0,1]); % x��y��ķ��?
    set(gca,'xtick',[0:0.1:1],'FontSize', 22,'FontWeight','bold');
    set(gca,'ytick',[0.1:0.1:1],'FontSize', 22,'FontWeight','bold');
    
    xlabel('Recall','FontWeight','bold','FontSize',22)%,'FontSize',15);
    ylabel('Precision','FontWeight','bold','FontSize',22)%,'FontSize',15);
    title('Precision and Recall (PR) Curve','FontWeight','bold','FontSize',24)%,'FontSize',15);
    
    tag = tag11; % ��ǩ
    %tagSet = legend(tag,3);  % 3 = Lower left-hand corner
    tagSet = gridLegend(hdl7, 2,tag,'FontSize', 14,'Orientation','Horizontal');
    set(tagSet, 'Position',[0.1 0.03 0.36 0.4]);
    %tagSet = columnlegend(2, tag);
    %some variables
    % numlines = length(str);
    % numpercolumn = ceil(numlines/numcolumns);
    % set( tagSet, 'FontSize', 14,'FontWeight','bold', 'Location','SouthWest');
    
    
    for i = 1:length(compareList22)
        fileName =['PR_', compareList22{i},'.mat'];
        load(fullfile(resPath, fileName));
        figure(j*10 + 8);
        if i == 1
            hdl8 = plot(REC,PRE,shapeList{i},'color',colorList(i,:),'linewidth',2);
        else
            hdl8 = [hdl8 plot(REC,PRE,shapeList{i},'color',colorList(i,:),'linewidth',2)];
        end
        
        hold on;
    end
    figure(j*10 + 8);
    grid on;
    axis([0.2,0.95, 0,1]); % x��y��ķ��?
    set(gca,'xtick',[0:0.1:1],'FontSize', 22,'FontWeight','bold');
    set(gca,'ytick',[0.1:0.1:1],'FontSize', 22,'FontWeight','bold');
    
    xlabel('Recall','FontWeight','bold','FontSize',22)%,'FontSize',15);
    ylabel('Precision','FontWeight','bold','FontSize',22)%,'FontSize',15);
    title('Precision and Recall (PR) Curve','FontWeight','bold','FontSize',24)%,'FontSize',15);
    
    
    tag = tag22; % ��ǩ
    %tagSet = legend(tag,3);  % 3 = Lower left-hand corner
    %set( tagSet, 'FontSize', 14,'FontWeight','bold');
    tagSet = gridLegend(hdl8, 2,tag,'FontSize', 14,'Orientation','Horizontal');
    set(tagSet, 'Position',[0.1 0.03 0.36 0.4]);
    
    
    figure(j*10 + 7);
    print('-depsc', fullfile(savePath, strcat(['PRI_', datasetList{j},'.eps'])));
    figure(j*10 + 8);
    print('-depsc', fullfile(savePath, strcat(['PRII_', datasetList{j},'.eps'])));
    pause;
end