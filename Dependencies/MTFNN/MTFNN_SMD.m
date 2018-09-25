% function [U, V, S, iter] = BFMN_SMD( F, laplacian, tree, weight, alpha, beta, r)
function [U, V, S, iter] = MTFNN_SMD( F, laplacian, tree, weight, alpha, beta, r)
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
tol1 = 1e-4; %threshold for the error in constraint
% tol2 = 1e-4; %threshold for the change in the solutions
% tolProj = 1e-2;
maxIter = 500000; % 最大迭代次数
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
% L = zeros(d,n);
UVT = zeros(d,n);
% 其实这个初始化没什么用
Z = zeros(d,n); % 论文中的 H
U = {eye(5, r), eye(12, r), eye(36, r)};
V = {eye(n, r), eye(n, r), eye(n, r)};
V_hat = {eye(n, r), eye(n, r), eye(n, r)};
S = zeros(d, n);
Y1 = {eye(n, r), eye(n, r), eye(n, r)};
Y1_mu = {eye(n, r), eye(n, r), eye(n, r)};
Y3 = zeros(d, n);
a1 = lansvd(F, 1, 'L');
a2 = norm(F, Inf);
Y2 = F/max(a1,a2);
M = zeros(d, r);
% 其实这个初始化也没什么用
alphalap = (2*alpha).*laplacian;
I_n = eye(n);
I_r = eye(r);
% leqy1y2 = 1000;
lambda = 1;
lambda_2 = lambda * 2;
% lambda = sqrt(max(d, n));
%% start main loop
iter = 0;
% primal_iter = 0;
%disp(['initial rank=' num2str(rank(Z))]);
while iter < maxIter
    iter = iter + 1;
    
    % copy Z and S to compute the change in the solutions
    % 保留当前值，以便后续计算变化率
    % to save time
    for i = 1:3
        Y1_mu{i} = Y1{i}./mu;
    end
    Y2_mu = Y2./mu;
    Y3_mu = Y3./mu;
    mu_3 = mu * 3;
    
    M = mu .* (F - S) - Y2;
    
    M_p = {M(1:5, :), M(6:17, :), M(18:53, :)};
    
    for i = 1:3
        % update U
        % 更新 U
        tu = ((2/(3 * mu)) .* I_r + V{i}' * V{i});
        U{i} = (M_p{i} ./ mu) * V{i} / tu;
        
        % update V
        % 更新 V
        tv = ((lambda + mu) .* I_r + mu .* U{i}' * U{i});
        V{i} = (V_hat{i} .* mu + Y1{i} + M_p{i}' * U{i}) / tv;
        
        % update V_hat
        % 更新 V_hat
        tv_hat = V{i} - Y1_mu{i};
        [VU,sigma,VV] = svd(tv_hat,'econ');
        sigma = diag(sigma);
        svp = length(find(sigma>lambda_2/mu_3));
        if svp>=1
            sigma = sigma(1:svp)-lambda_2/mu_3;
        else
            svp = 1;
            sigma = 0;
        end
        V_hat{i} = VU(:,1:svp) * diag(sigma) * VV(:,1:svp)';
        
    end
    UVT = [U{1} * V{1}'; U{2} * V{2}'; U{3} * V{3}'];
    % udpate Z
    % 更新 H
    % 这里的alpha是论文中的beta
    % 使用MATLAB中矩阵"除法"来代替inv求逆
    
    t2 = alphalap + mu.*I_n;
    Z = (mu.*S + Y3) / t2; %faster and more accurate than inv(temp)
    
    % update S
    % 更新 S
    % T为论文中的 X_S
    T = 0.5.* ( F - UVT + Z - Y2_mu - Y3_mu);
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
    leq1 = [V_hat{1}; V_hat{2}; V_hat{3}] - [V{1}; V{2}; V{3}];
    leq2 = UVT + S - F;
    leq3 = S - Z;
    % 计算L和S的变化量, 取两者的最大值
    
    % 计算稀疏矩阵与低秩矩阵合成的误差
    recErrV = norm(leq1,'fro')/normfF;
    recErrLSF = norm(leq2,'fro')/normfF;
    recErr = max(recErrLSF, recErrV);
    % 如果变化率和误差小于设定值, 则认为已经收敛, 循环结束
    convergenced = recErr <tol1;
    
    % 用于打印训练轨迹
%         if iter==1 || mod(iter,1)==0 || convergenced
%             disp(['iter ' num2str(iter) ',mu=' num2str(mu,'%2.1e') ...
%                 ',rank=' num2str(rank(UVT)) ',stopADM=' num2str(recErr,'%2.3e')]);
%         end
    
    if convergenced
        % 用于提示训练结束
        % disp('SMF done.');
        break;
    else
        % 更新拉格朗日乘子Y1和Y2
        leq1_p = {leq1(1:n, :), leq1(n + 1:2 * n, :), leq1(2 * n + 1:3 * n, :)};
        for i = 1:3
            Y1{i} = Y1{i} + mu*leq1_p{i};
        end
        Y2 = Y2 + mu*leq2;
        Y3 = Y3 + mu*leq3;
        % 更新罚函数系数 加速收敛
        mu = min(max_mu,mu*rho);
    end
end
