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
        % 加上 1e-6 避免除零错误
        nR = colorFeatures(index,1)/(sum(colorFeatures(index,:))+1e-6);
        nG = colorFeatures(index,2)/(sum(colorFeatures(index,:))+1e-6);
        % /0.05 将 [0, 1] 缩放到 [0, 20]
        x = min(floor(nR/0.05)+1,20);
        y = min(floor(nG/0.05)+1,20);
        % 这里与 [X. Shen and Y. Wu, CVPR'12] 中的做法不一样
        colorPrior(index,1) = (colorPriorMat(x,y)+0.5)/1.5;
    end
    % integrate center, color and background priors
    % 与[X. Shen and Y. Wu, CVPR'12]不同的另外一点为: 使用bgPrior替换了semantic prior
    prior = centerPrior .* colorPrior .* bgPrior;
    prior = mapminmax(prior',0,1)';