function [L, S, iter] = SMD_Bi( F, laplacian, tree, weight, alpha, beta, r, lapNorm)
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

%% intialize
tol1 = 1e-5; %threshold for the error in constraint
tol2 = 1e-5; %threshold for the change in the solutions
maxIter = 1000; % 最大迭代次数
rho = 1.1; % 用于更新惩罚系数
mu = 0.1; % 惩罚系数
max_mu = 1e10; % 最大的惩罚系数
[d, n] = size(F);

% to save time
% 主要用于后面的收敛检测
normfF = norm(F,'fro');
if r <= 0
    r = round(1.2*rank_estimation(F));
end
% initialize optimization variables
L = zeros(d,n);
% 其实这个初始化没什么用
Z = zeros(d,n); % 论文中的 H
S = zeros(d,n);
Y1 = zeros(d,n);
Y2 = zeros(d,n);
Q = zeros(d, r);
R = eye(r, n);
% 其实这个初始化也没什么用
svp = 5; % for svd
alphalap = (2*alpha).*laplacian;
I_n = eye(n);
%% start main loop
iter = 0;
%disp(['initial rank=' num2str(rank(Z))]);
while iter < maxIter
    iter = iter + 1;
    
    % copy Z and S to compute the change in the solutions
    % 保留当前值，以便后续计算变化率
    Lk = L;
    Sk = S;
    % to save time
    Y1_mu = Y1./mu;
    Y2_mu = Y2./mu;

    P = F - S + Y1_mu;
    [Q, ~] = qr(P * R', 0);
    
    % update M 
    % 更新 M
    t1 = Q' * P;
    [U,sigma,V] = svd(t1,'econ');
    % 将对角矩阵转换为列矩阵(默认为降序排列)
    sigma = diag(sigma);
    % 只保留值大于1/mu的特征值
    svp = length(find(sigma>1/mu));
    if svp>=1
        sigma = sigma(1:svp)-1/mu;
    else
        svp = 1;
        sigma = 0; % 不明白为什么这样处理
    end
    R = U(:,1:svp) * diag(sigma) * V(:,1:svp)';
    L = Q * R;
    % save time
    
    % udpate Z
    % 更新 H
    % 这里的alpha是论文中的beta
    % 使用MATLAB中矩阵"除法"来代替inv求逆
    % temp = (2*alpha).*laplacian + mu.*eye(n);
    % Z = (mu.*S + Y2) / temp; %faster and more accurate than inv(temp)
    % Z = (alpha/mu).*F + Y2_mu + S;
    
    if lapNorm
        t2 = alphalap + mu.*I_n;
        Z = (mu.*S + Y2) / t2; %faster and more accurate than inv(temp)
    else
        Z = (alpha/mu).*F + Y2_mu + S;
    end
    
    % update S 
    % 更新 S
    % T为论文中的 X_S
    T = 0.5.* ( F - L + Z + Y1_mu - Y2_mu);
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
                temp = (gij_l1 - weightij) / gij_l1;
                S(:,gij) = temp .* S(:,gij);
            else
                S(:,gij) = 0;
            end
        end
    end
    
    
    % check convergence condition
    % 计算两个约束的偏差
    leq1 = F - L - S;
    leq2 = S - Z;
    % leq3 = L - T;
    % 计算L和S的变化量, 取两者的最大值
    relChgL = norm(L - Lk,'fro')/normfF;
    relChgS = norm(S - Sk,'fro')/normfF;
    relChg = max(relChgL, relChgS);
    % 计算稀疏矩阵与低秩矩阵合成的误差
    recErr = norm(leq1,'fro')/normfF; 

    
    % 如果变化率和误差小于设定值, 则认为已经收敛, 循环结束
    convergenced = recErr <tol1 && relChg < tol2;
    
    % 用于打印训练轨迹
%     if iter==1 || mod(iter,1000)==0 || convergenced
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
        % 更新罚函数系数 加速收敛
        mu = min(max_mu,mu*rho);        
    end
end
