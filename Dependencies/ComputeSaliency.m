function [salMap, iter, time] = ComputeSaliency(img, paras, setting)
% FUNCTION: Calculate saliency using Structured Matrix Fractorization (SMF)
% INPUT:    img - input image
%           paras - parameter setting
% OUTPUT:   saliency map

% resPath = strcat(['visual\', img.name(1:end-4)]);
% SMD_Sal = im2double(imread(fullfile('SAL_MAP\SMD', [img.name(1:end-4), '.png'])));
% resPath = strcat(['visual\', img.name(1:end-4)]);
% if ~exist(resPath, 'file')
%     mkdir(resPath);
% end
FeatPath = fullfile('Feature', [img.name(1:end-4), '.mat']);
if (~exist(FeatPath, 'file'))
    %% STEP-1. Read an input images and perform preprocessing
    % ȥ��ͼƬ�ı߿�
    [img.height, img.width] = size(img.RGB(:,:,1));
    [noFrameImg.RGB, noFrameImg.frame] = RemoveFrame(img.RGB, 'sobel');
    [noFrameImg.height, noFrameImg.width, noFrameImg.channel] = size(noFrameImg.RGB);
    %% STEP-2. Generate superpixels using SLIC
    % ʹ��SLIC�����ɳ�����
    % label��ÿ�����ص��Ӧ�ĳ�����
    % num�������ص�����
    % Lab���������е�Labƽ��ֵ
    % pixIdx��ÿ���������а��������ص㼯��(һά��)
    % pixNum��ÿ���������а��������ص����
    % pos�������ص�ƽ��λ��
    
    [sup.label, sup.num, sup.Lab] = PerformSLIC(noFrameImg, paras.comp, paras.pix);
    % get superpixel statistics
    sup.pixIdx = cell(sup.num, 1);
    sup.pixNum = zeros(sup.num,1);
    for i = 1:sup.num
        temp = find(sup.label==i);
        sup.pixIdx{i} = temp;
        sup.pixNum(i) = length(temp);
    end
    sup.pos = GetNormedMeanPos(sup.pixIdx, noFrameImg.height, noFrameImg.width);
    
    % RGBLabel = superpixels2RGB(sup.label);
%     
%     mask = boundarymask(sup.label);
%     imwrite(imoverlay(noFrameImg.RGB,mask,'white'),strcat([resPath, '\OS.png']),'png');
    %% STEP-3. Extract features
    % ��ȡ���� ���Ϊ(h, w, 53)ά����
%     imdata = drfiGetImageData( noFrameImg.RGB );
%     pbgdata = drfiGetPbgFeat( imdata );
%     
%     imsegs.imsize = size(sup.label);
%     imsegs.segimage = uint16(sup.label);
%     imsegs.nseg = sup.num;
%     imsegs = APPgetSpStats(imsegs);
%     
%     % data of each superpixel
%     spdata = drfiGetSuperpixelData( imdata, imsegs );
%     
%     % saliency feature of each segment (region)
%     featMat = drfiGetRegionSaliencyFeature( imsegs, spdata, imdata, pbgdata );

    featImg = ExtractFeature(im2single(img.RGB));
    % ��ǰ3��ά�ȴ�[0,1]��չΪ[0,255]
    for i = 1:3
        featImg(:,i) = mat2gray(featImg(:,i));
    end
%     for i = 1:3
%         featMat(:,i) = mat2gray(featMat(:,i));
%     end
        % ��ÿ���������е��������ص���ƽ��ֵ���Ϊ(num, 53)
        % ���� (h, w, 53) -> (num, 53)
        featMat = GetMeanFeat(featImg, sup.pixIdx);
        % ��ֵ��[0, 255]����Ϊ[0,1]
        featMat = featMat./255;
    colorFeatures = featMat(:,1:3);
        % ��ȡRBG�е���λֵ
    medianR = median(colorFeatures(:,1));
    medianG = median(colorFeatures(:,2));
    medianB = median(colorFeatures(:,3));
    % normalized by subtracting its median value over entire image
    % ͨ����ȥͼƬ�е���λֵ����һ��
    % featC = (C - (1.2 * midC)) * 1.5
    featMat(:,1:3) = (featMat(:,1:3)-1.2*repmat([medianR, medianG, medianB],size(featMat,1),1))*1.5;
%     % ���������Ŀ��ӻ�Ч��ͼ
%         [h, ~]= size(featMat);
%         FMI = abs(reshape(featMat(:, 1:51), [h, 17, 3]));
%         r = FMI(:, :, 1);
%         g = FMI(:, :, 2);
%         b = FMI(:, :, 3);
%         all = (r + b + g)/3;
%         r = (r - all)*1.5 + 0.53;
%         g = (g - all)*5 + 0.8;
%         b = (b - all)*4 + 0.97;
%         FMI(:, :, 1) = r;
%         FMI(:, :, 2) = g;
%         FMI(:, :, 3) = b;
%         FMI = imrotate(FMI, -90);
%         FMI = flip(FMI, 2);
%         FMI = imresize(FMI, 30, 'nearest');
%         imwrite(FMI ,strcat([resPath, '\FM.png']),'png');
%         imshow(abs(featMat(:,1:51)'));
    %% STEP-4. Create index tree
    % ֱ�Ӵӳ����ؿ�ʼ����
    % get the first and second order reachable matrix (a.k.a adjacent matrix)
    % ��ȡһ���ɵ��ĳ�������Ϻ������ɵ��ĳ��������
    % fstSndReachMat_lowTri �������Խ����Լ��Խ������ϵ�Ԫ��
    [fstReachMat, fstSndReachMat, fstSndReachMat_lowTri] = GetFstSndReachMat(sup.label, sup.num);
    % get linked superpixel pairs
    [tmpRow, tmpCol] = find(fstSndReachMat_lowTri>0);
    edge = [tmpRow, tmpCol];
    % compute color distance between adjacent superpixels
    % ����LAB���� ||V_lab_i - V_lab_j||_2
    colorDistMat = ComputeFeatDistMat(sup.Lab);
    % get the weights on edges
    weightOnEdge = ComputeWeightOnEdge(colorDistMat, fstSndReachMat_lowTri, paras.delta);
    % hierachical segmenation
    % k = [300 400 600 800 1200 1600 2000];
    % k = [100 300 600 2000 8000];
    k = [300 600 2000];
    % for hierachical segmentation
    multiSegments = mexMergeAdjRegs_Felzenszwalb(edge, weightOnEdge, sup.num, k, sup.pixNum);
    % index tree
    % ���ɵ���ֻ������
    Tree = CreateIndexTree(sup.num, multiSegments);
    % �������Ŀ��ӻ�Ч��ͼ
    % showIndexTree(img, sup, multiSegments);
    %% STEP-5. Get high-level priors
    % load color prior matrix as used in [X. Shen and Y. Wu, CVPR'12]
    if ~exist('colorPriorMat','var')
        fileID = fopen('ColorPrior','rb');
        data = fread(fileID,'double');
        fclose(fileID);
        colorPriorMat = reshape(data(end-399:end), 20, 20);
    end
    % get the indexes of boundary superpixels
    % ��ȡ�߽�ĳ�����
    bndIdx = GetBndSupIdx(sup.label);
    % get banckground prior by robust background detection [W. Zhu et al., CVPR'14]
    % ʹ��RBD���������鱳��֪ʶ
    % bgPrior��ÿ���������Ǳ����ĸ���[0, 1]; 1ΪС, 0Ϊ��
    % bdCon����߽����ͨ��
    [bgPrior, bdCon] = GetPriorByRobustBackgroundDetection(colorDistMat, fstReachMat, bndIdx);
    % ����bgPrior�Ŀ��ӻ�Ч��ͼ

    % prior��(num, 1)
    [prior, ~, ~] = GetHighLevelPriors(sup.pos, sup.num, colorPriorMat, colorFeatures, bgPrior);
    % [prior, colorPrior, centerPrior] = GetHighLevelPriors(sup.pos, sup.num, colorPriorMat, colorFeatures, bgPrior);
    % pre-processing
    % ʹ��
%     imwrite(featMapRGBShow(centerPrior, sup.label) ,strcat([resPath, '\LocaltionP.png']),'png');
%     imwrite(featMapRGBShow(colorPrior, sup.label) ,strcat([resPath, '\ColorP.png']),'png');
%     imwrite(featMapRGBShow(bgPrior, sup.label) ,strcat([resPath, '\BackgroundP.png']),'png');
%     imwrite(featMapRGBShow(prior, sup.label) ,strcat([resPath, '\High-levelP.png']),'png');
    
    featMat = repmat(prior,1,53) .* featMat;
    % featMat = repmat(prior,1,93) .* featMat;
    
    % convert high-level priors as the weight in the structured sparsity
    % regularization term
    % ���߲�����ת��Ϊ������״�ṹʱ���õ�Ȩ�� v_ij
    weight = cell(1,length(Tree));
    for m = 1:length(Tree)
        for n = 1:length(Tree{m})
            ind = cell2mat(Tree{m}(n));
            tmpPri = prior(ind);
            weight{m}(n) = max(1-max(tmpPri),0);
        end
    end
    
    %% STEP-6. Compute affinity and laplacian matrix
    % link boundary superpixels
    % ���߽�ĳ�������������
    fstSndReachMat(bndIdx, bndIdx) = 1;
    % ȥ���Խ����ϵ�Ԫ��
    fstSndReachMat_lowTri_Bnd = tril(fstSndReachMat, -1);
    [tmpRow, tmpCol] = find(fstSndReachMat_lowTri_Bnd>0);
    edge = [tmpRow, tmpCol];
    % get the weights on edges
    weightOnEdge = ComputeWeightOnEdge(colorDistMat, fstSndReachMat_lowTri_Bnd, paras.delta);
    % compute affinity matrix
    % ����ǶԽ����ϵ�Ԫ��
    W = sparse([edge(:,1);edge(:,2)], [edge(:,2);edge(:,1)], [weightOnEdge; weightOnEdge], sup.num, sup.num);
    % ����Խ����ϵ�Ԫ�� �ǶԽ����ϵ�Ԫ����Ϊ0
    DCol = full(sum(W,2));
    D = spdiags(DCol,0,speye(sup.num));
    % ���������laplacian matrix
    M = D - W;
    HLP = 1 ./ (1 + exp(repmat(prior,1,53)'));
    save(FeatPath, 'featMat', 'M', 'Tree', 'weight', 'sup', 'noFrameImg', 'bdCon', 'fstReachMat', 'W', 'bgPrior', 'HLP');
else
    load(FeatPath, 'featMat', 'M', 'Tree', 'weight', 'sup', 'noFrameImg', 'bdCon', 'fstReachMat', 'W', 'bgPrior', 'HLP');
end

%% STEP-7. Structured matrix decomposition
% ��feature������е��ȷֽ�
% [L, S] = SMD_M(featMat', M, Tree, weight, paras.alpha, paras.beta, paras.r);
switch setting.model
    case 'lrrsr'
        tic;
        [ZZ, LL, S, iter] = LRRSR(featMat', M, Tree, weight, paras.alpha, paras.beta);
        time = toc;
        L = featMat' * ZZ + LL * featMat';
        %     FMI = abs(ZZ);
        %     FMI = imresize(FMI, 30, 'nearest');
        %     imwrite(FMI ,strcat(['visual\', img.name(1:end-4), '_ZZP.png']),'png');
        %
        %     FMI = abs(LL);
        %     FMI = imresize(FMI, 30, 'nearest');
        %     imwrite(FMI ,strcat(['visual\', img.name(1:end-4), '_LLP.png']),'png');
    case 'smd'
        tic;
        [L, S, iter] = SMD(featMat', M, Tree, weight, paras.alpha, paras.beta, setting.lapNorm);
        time = toc;
    case 'bsmd'
        tic;
        [L, S, iter] = SMD_Bi(featMat', M, Tree, weight, paras.alpha, paras.beta, paras.r, setting.lapNorm);
        time = toc;
    case 'tsmd'
        tic;
        [L, S, iter] = SMD_Tri(featMat', M, Tree, weight, paras.alpha, paras.beta, paras.r, setting.lapNorm);
        time = toc;
    case 'bfmn_smd'
        tic;
        % [U, V, S, iter] = EBFMN_SMD(featMat', M, Tree, weight, paras.alpha, paras.beta, paras.r, paras.lambda);
        [U, V, S, iter] = BFMN_SMD(featMat', M, Tree, weight, paras.alpha, paras.beta, paras.r);
        L = U * V';
        time = toc;
    case 'dnn_smd'
        tic;
        [U, V, S, iter] = DNN_SMD(featMat', M, Tree, weight, paras.alpha, paras.beta, paras.r);
        L = U * V';
        time = toc;
    case 'fnn_smd'
        tic;
        [U, V, S, iter] = FNN_SMD(featMat', M, Tree, weight, paras.alpha, paras.beta, paras.r);
        L = U * V';
        time = toc;
    case 'prior'
        tic;
        [h, w] = size(featMat');
        L = zeros(h, w);
        S = featMat';
        iter = 0;
        time = toc;
    case 'theory'
        [L, S, ~] = BFMN_SMD(featMat', M, Tree, weight, paras.alpha, paras.beta, paras.r);
        tic;
        [L, S, iter] = SMD_T(featMat', M, Tree, weight, paras.alpha, paras.beta, setting.lapNorm, L, S);
        time = toc;
        %         L = Q * M *R;
        %         frame = abs(Q);
        %         [h,w] = size(frame);
        %         FMI = zeros(h, w, 3);
        %         r = frame;
        %         g = frame;
        %         b = ones(h, w) * 0.97;
        %         meanV = mean(mean(frame)) * 3;
        %         r(frame < 0.05) = meanV;
        %         g(frame < 0.05) = meanV;
        %         b(frame > 0.05) = 0.1;
        %         r = (meanV - r)*2 + 0.53;
        %         g = (g - meanV)*2 + 0.8;
        %         FMI(:, :, 1) = r;
        %         FMI(:, :, 2) = g;
        %         FMI(:, :, 3) = b;
        %         FMI = imresize(FMI, 30, 'nearest');
        %         imwrite(FMI ,strcat([resPath, '\Q.png']),'png');
        %
        %         frame = abs(M);
        %         [h,w] = size(frame);
        %         FMI = zeros(h, w, 3);
        %         r = frame;
        %         g = frame;
        %         b = ones(h, w) * 0.97;
        %         meanV = mean(mean(frame));
        %         r(frame < 0.05) = meanV;
        %         g(frame < 0.05) = meanV;
        %         r = (frame - meanV) + 0.53;
        %         g = (frame - meanV) + 0.8;
        %         FMI(:, :, 1) = r;
        %         FMI(:, :, 2) = g;
        %         FMI(:, :, 3) = b;
        %         FMI = imresize(FMI, 30, 'nearest');
        %         imwrite(FMI ,strcat([resPath, '\M.png']),'png');
        %
        %         frame = abs(R);
        %         [h,w] = size(frame);
        %         FMI = zeros(h, w, 3);
        %         r = frame;
        %         g = frame;
        %         b = ones(h, w) * 0.97;
        %         meanV = mean(mean(frame)) * 5;
        %         r(frame < 0.05) = meanV;
        %         g(frame < 0.05) = meanV;
        %         b(frame > 0.05) = 0.1;
        %         r = (meanV - r)*2 + 0.53;
        %         g = (g - meanV)*2 + 0.8;
        %         FMI(:, :, 1) = r;
        %         FMI(:, :, 2) = g;
        %         FMI(:, :, 3) = b;
        % %         FMI = rgb2hsv(FMI);
        % %         FMI(:, :, 2) = FMI(:, :, 2) + 0.8;
        % %         FMI = hsv2rgb(FMI);
        %         FMI = imresize(FMI, 30, 'nearest');
        %         imwrite(FMI ,strcat([resPath, '\R.png']),'png');
end

% ��ϡ������||A||_1; ����ȡΪ[0, 1]
S_sal = mapminmax(sum(abs(S),1),0,1);
% [h, ~]= size(featMat);
% FMI = abs(reshape(featMat(:, 1:51), [h, 17, 3]));
% r = FMI(:, :, 1);
% g = FMI(:, :, 2);
% b = FMI(:, :, 3);
% all = (r + b + g)/3;
% r = (r - all)*1.5 + 0.53;
% g = (g - all)*5 + 0.8;
% b = (b - all)*4 + 0.97;
% FMI(:, :, 1) = r;
% FMI(:, :, 2) = g;
% FMI(:, :, 3) = b;
% % FMI = rgb2hsv(FMI);
% % FMI(:, :, 2) = FMI(:, :, 2) + 0.1;
% % FMI = hsv2rgb(FMI);
% FMI = imrotate(FMI, -90);
% FMI = flip(FMI, 2);
% % FMI = abs(featMat');
% FMI = imresize(FMI, 30, 'nearest');
% imwrite(FMI ,strcat([resPath, '\FM_2.png']),'png');
% 
% FMI = abs(reshape(L(1:51, :)', [h, 17, 3]));
% r = FMI(:, :, 1);
% g = FMI(:, :, 2);
% b = FMI(:, :, 3);
% all = (r + b + g)/3;
% r = (r - all)*1.5 + 0.53;
% g = (g - all)*5 + 0.8;
% b = (b - all)*4 + 0.97;
% FMI(:, :, 1) = r;
% FMI(:, :, 2) = g;
% FMI(:, :, 3) = b;
% % FMI = rgb2hsv(FMI);
% % FMI(:, :, 2) = FMI(:, :, 2) + 0.8;
% % FMI = hsv2rgb(FMI);
% FMI = imrotate(FMI, -90);
% FMI = flip(FMI, 2);
% % FMI = abs(L);
% FMI = imresize(FMI, 30, 'nearest');
% imwrite(FMI, strcat([resPath, '\', setting.model, '_L.png']),'png');
% 
% FMI = abs(reshape(S(1:51, :)', [h, 17, 3]));
% r = FMI(:, :, 1);
% g = FMI(:, :, 2);
% b = FMI(:, :, 3);
% all = (r + b + g)/3;
% r = (r - all)*1.5 + 0.53;
% g = (g - all)*5 + 0.8;
% b = (b - all)*4 + 0.97;
% FMI(:, :, 1) = r;
% FMI(:, :, 2) = g;
% FMI(:, :, 3) = b;
% % FMI = rgb2hsv(FMI);
% % FMI(:, :, 2) = FMI(:, :, 2) + 0.8;
% % FMI = hsv2rgb(FMI);
% FMI = imrotate(FMI, -90);
% FMI = flip(FMI, 2);
% % FMI = abs(S);
% FMI = imresize(FMI, 30, 'nearest');
% imwrite(FMI, strcat([resPath, '\', setting.model, '_S.png']),'png');
% 
% % D = featMat - L' - S';
% % O = 0;
% % for i = 1:length(Tree)
% %     % ���ڲ�ѭ�������ÿ�������ص�norm_1
% %     Slinf = max(abs(S));
% %     % ��ѭ��Ϊ����ǰ��ȵĽ��
% %     for j = 1:length(Tree{i})
% %         % ��ȡ��ǰ�������ĳ�����
% %         gij = Tree{i}{j};
% %         % ���㵱ǰ�������ĳ����������ɵľ���ĵ�norm_1
% %         gij_linf = max(Slinf(gij)); %%%%
% %         % weightij Ϊ�����е� v_ij
% %         O = O + gij_linf * weight{i}(j);
% %     end
% % end
% % loss = paras.lambda * rank(L) + paras.alpha * trace(S * M * S') + paras.beta * O;
% fprintf('L: %s \n', num2str(rank(L)));
% fprintf('S: %s \n', num2str(sum(S(:))));
% % fprintf('Loss: %s \n', num2str(loss));
%% STEP-8: Post-processing to get improvements
% parameters for postprocessing
% context-based propagation
if(setting.postProc)
    lambada = 0.1;  % 0.1 is good
    L_sal = lambada * (1-mapminmax(sum(abs(L),1),0,1)') + (1-lambada) * (1 - mapminmax(bgPrior',0,1)');
    S_sal = postProcessing(L_sal, S_sal, bdCon, fstReachMat, W, sup);
end
% save saliency map
% ��ϡ����ӳ�䵽ͼƬ��(����ͼ����)


salMap = GetSaliencyMap(S_sal, sup.pixIdx, noFrameImg.frame, true);
% fprintf('%s %s\n', img.name, num2str(mean(max(salMap - SMD_Sal))));
