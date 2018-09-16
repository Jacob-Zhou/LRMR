function [Z, L, S, iter] = LRRSR( F, laplacian, tree, weight, alpha, beta)
% function [L, S] = SMD_M( F, laplacian, tree, weight, alpha, beta, r)
% This routine solves the following structured matrix fractorization optimization problem,
%   min |L|_* + alpha*tr(SMS^T) + beta*\sum\sum|w_g*S_g|_p
%   s.t., F = L + S
% inputs:
%       F -- D*N data matrix, D is the data dimension, and N is the number of data vectors.           
%       M -- N*N laplacian matrix
%       tree -- index tree
%       weight -- weight of each node in the tree
%       alpha -- the penalty for tr(SMS^T)
%       beta -- the penalty for |w_g*S_g|_p
% outputs:
%       L -- D*N low-rank matrix
%       S -- D*N structured-sparse matrix
%
%% References
%   [1] Houwen Peng et al. 
%       Salient Object Detection via Structured Matrix Decomposition.
%   [2] Houwen Peng, Bing Li, Rongrong Ji, Weiming Hu, Weihua Xiong, Congyan Lang: 
%       Salient Object Detection via Low-Rank and Structured Sparse Matrix 
%       Decomposition. AAAI 2013: 796-802.
addpath(genpath('PROPACK'));
%% intialize
tol1 = 1e-5; %threshold for the error in constraint
tol2 = 1e-8; %threshold for the change in the solutions
maxIter = 1000; % 最大迭代次数
rho = 1.1; % 用于更新惩罚系数
mu = 1e-1; % 惩罚系数
max_mu = 1e10; % 最大的惩罚系数
[d, n] = size(F);

% to save time
% 主要用于后面的收敛检测
normfF = norm(F,'fro');

% initialize optimization variables
L = zeros(d,n);
% 其实这个初始化没什么用
H = zeros(d,n);
Z = zeros(n,n);
L = zeros(d,d);
J = zeros(n,n);
E = zeros(d,d);
S = zeros(d,n);
Y1 = zeros(d,n);
Y2 = zeros(n,n);
Y3 = zeros(d,d);
Y4 = zeros(d,n);
I_Z = eye(n, n);
I_L = eye(d, d);
% 其实这个初始化也没什么用
sv_Z = 3; % for svd
sv_L = 3;
%% start main loop
iter = 0;
%disp(['initial rank=' num2str(rank(Z))]);
while iter < maxIter
    iter = iter + 1;
    
    % copy Z and S to compute the change in the solutions
    % 保留当前值，以便后续计算变化率
    Lk = L;
    Sk = S;
    Zk = Z;
    % to save time
    Y1_mu = Y1./mu;
    Y2_mu = Y2./mu;
    Y3_mu = Y3./mu;
    Y4_mu = Y4./mu;
    
    % update M 
    % 更新 M
    temp_Z = Z + Y2_mu;
    [U_Z,sigma_Z,V_Z] = lansvd(temp_Z,sv_Z,'L');
    % 将对角矩阵转换为列矩阵(默认为降序排列)
    sigma_Z = diag(sigma_Z);
    sigma_Z = sigma_Z(1:sv_Z);
    % 只保留值大于1/mu的特征值
    svn_Z = length(find(sigma_Z>1/mu));
    svp_Z = svn_Z;
    ratio = sigma_Z(1:end-1)./sigma_Z(2:end);
    [max_ratio_Z, max_idx_Z] = max(ratio);
    if max_ratio_Z > 2
        svp_Z = min(svn_Z, max_idx_Z);
    end
    if svp_Z < sv_Z %|| iter < 10
        sv_Z = min(svp_Z + 1, d);
    else
        sv_Z = min(svp_Z + 10, d);
    end
    J = U_Z(:,1:svp_Z) * diag(sigma_Z(1:svp_Z) - 1 / mu) * V_Z(:,1:svp_Z)';
    
        temp_L = L + Y3_mu;
    [U_L,sigma_L,V_L] = lansvd(temp_L,sv_L,'L');
    % 将对角矩阵转换为列矩阵(默认为降序排列)
    sigma_L = diag(sigma_L);
    sigma_L = sigma_L(1:sv_L);
    % 只保留值大于1/mu的特征值
    svn_L = length(find(sigma_L>1/mu));
    svp_L = svn_L;
    ratio = sigma_L(1:end-1)./sigma_L(2:end);
    [max_ratio_L, max_idx_L] = max(ratio);
    if max_ratio_L > 2
        svp_L = min(svn_L, max_idx_L);
    end
    if svp_L < sv_L %|| iter < 10
        sv_L = min(svp_L + 1, n);
    else
        sv_L = min(svp_L + 10, n);
    end
    E = U_L(:,1:svp_L) * diag(sigma_L(1:svp_L) - 1 / mu) * V_L(:,1:svp_L)';

    Z = (F' * F + I_Z) \ (F' * (F - L * F - S) + J + F' * Y1_mu - Y2_mu);
    L = ((F - F * Z - S) * F' + E + Y1_mu * F' - Y3_mu) / (F * F' + I_L);
    
    % udpate Z
    % 更新 H
    % 这里的alpha是论文中的beta
    % 使用MATLAB中矩阵"除法"来代替inv求逆
	temp = (2*alpha).*laplacian + mu.*eye(n);
	H = (mu.*S + Y4) / temp; %faster and more accurate than inv(temp)

%     H = (alpha/mu).*F + Y4_mu + S;

    % update S 
    % 更新 S
    % T为论文中的 X_S
    T = 0.5.* ( F - F * Z - L * F + H + Y1_mu - Y4_mu);
    % 这里的beta是论文中的alpha
    lambada = 0.5*beta/mu;
   
    S = T;
    % 外循环为树的深度(自顶向下)
    for i = 1:length(tree)
        % 在内层循环外计算每个超像素的norm_1
        Sl1 = sum(abs(S));
        % 内循环为树当前深度的结点
        for j = 1:length(tree{i})
            % 获取当前结点包含的超像素
            gij = tree{i}{j};
            % 计算当前结点包含的超像素所构成的矩阵的的norm_1
            gij_l1 = sum(Sl1(gij)); %%%%
            % weightij 为论文中的 v_ij
            weightij = lambada * weight{i}(j);
            if gij_l1 > weightij
                temp_Z = (gij_l1 - weightij) / gij_l1;
                S(:,gij) = temp_Z .* S(:,gij);
            else
                S(:,gij) = 0;
            end
        end
    end
    
    % check convergence condition
    % 计算两个约束的偏差
    leq1 = F - F * Z - L * F - S;
    leq2 = Z - J;
    leq3 = L - E;
    leq4 = S - H;
    % 计算L和S的变化量, 取两者的最大值
    relChgL = norm(L - Lk,'fro')/normfF;
    relChgZ = norm(Z - Zk,'fro')/normfF;
    relChgS = norm(S - Sk,'fro')/normfF;
    relChg = max([relChgL, relChgS, relChgZ]);
    % 计算稀疏矩阵与低秩矩阵合成的误差
    recErr = norm(leq1,'fro')/normfF; 
    
    % 如果变化率和误差小于设定值, 则认为已经收敛, 循环结束
    convergenced = recErr <tol1 && relChg < tol2;
    
    % 用于打印训练轨迹
%     if iter==1 || mod(iter,1)==0 || convergenced
%         disp(['iter ' num2str(iter) ',mu=' num2str(mu,'%2.1e') ...
%           ',rank=' num2str(rank(L,1e-3*norm(L,2))) ',stopADM=' num2str(recErr,'%2.3e')]);
%     end
    
    if convergenced
        % 用于提示训练结束
        % disp('SMF done.');
        break;
    else
        % 更新拉格朗日乘子Y1和Y2
        Y1 = Y1 + mu*leq1;
        Y2 = Y2 + mu*leq2;
        Y3 = Y3 + mu*leq3;
        Y4 = Y4 + mu*leq4;
        % 更新罚函数系数 加速收敛
        mu = min(max_mu,mu*rho);        
    end
end
