function [prior,colorPrior, centerPrior] = GetHighLevelPriors(supPos, supNum, colorPriorMat, colorFeatures, bgPrior)
    % center prior
    center = [0.5 0.5];
    centerPrior = zeros(supNum,1);
    sigma = 0.25;
    for c = 1:supNum
        tmpDist = norm( supPos(c,:) - center );
        centerPrior(c) = exp(-tmpDist^2/(2*sigma^2));
    end
    % color prior
    colorPrior = zeros(supNum,1);
    for index = 1:supNum
        % nR = R / (R + G + B)
        % nG = G / (R + G + B)
        % ���� 1e-6 ����������
        nR = colorFeatures(index,1)/(sum(colorFeatures(index,:))+1e-6);
        nG = colorFeatures(index,2)/(sum(colorFeatures(index,:))+1e-6);
        % /0.05 �� [0, 1] ���ŵ� [0, 20]
        x = min(floor(nR/0.05)+1,20);
        y = min(floor(nG/0.05)+1,20);
        % ������ [X. Shen and Y. Wu, CVPR'12] �е�������һ��
        colorPrior(index,1) = (colorPriorMat(x,y)+0.5)/1.5;
    end
    % integrate center, color and background priors
    % ��[X. Shen and Y. Wu, CVPR'12]��ͬ������һ��Ϊ: ʹ��bgPrior�滻��semantic prior
    prior = centerPrior .* colorPrior .* bgPrior;
    prior = mapminmax(prior',0,1)';