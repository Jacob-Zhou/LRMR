function [U, V, S, iter] = BFMN_SMD( F, laplacian, tree, weight, alpha, beta, r)
% function [L, S, iter] = EBFMN_SMD( F, laplacian, tree, weight, alpha, beta, r)
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

% [~,sig,~] = lansvd(F,25);
% sig = diag(sig);
% dsig = (sig(1:24) - sig(2:25)) ./ sig(2:25);
% d2sig = (dsig(1:23) - dsig(2:24)) ./ dsig(2:24);
% rp = find(d2sig < 0);
% if ~isempty(rp)
%     r = rp(1) + 2;
% else
%     r = 25;
% end

%% intialize
tol1 = 1e-4; %threshold for the error in constraint
% tol2 = 1e-4; %threshold for the change in the solutions
% tolProj = 1e-1;
maxIter = 5000; % 最大迭代次数
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
L = F;
% 其实这个初始化没什么用
Z = zeros(d,n); % 论文中的 H
U = eye(d, r);
V = eye(n, r);
U_hat = eye(d, r);
V_hat = zeros(n, r);
S = zeros(d, n);
Y1 = zeros(d, r);
Y2 = zeros(n, r);
a1 = lansvd(F, 1, 'L'); 
a2 = norm(F, Inf); 
Y3 = F/max(a1,a2);
Y4 = Y3;
% Y3 = zeros(d, n);
% Y4 = F;
Y5 = zeros(d, n);
M = zeros(d, r);
% 其实这个初始化也没什么用
alphalap = (2*alpha).*laplacian;
I_n = eye(n);
I_r = eye(r);
lambda = 1;

% lambda = sqrt(max(d, n));

%% start main loop
iter = 0;
% primal_iter = 0;
%disp(['initial rank=' num2str(rank(Z))]);
while iter < maxIter
    iter = iter + 1;
    
    % copy Z and S to compute the change in the solutions
    % 保留当前值，以便后续计算变化率
%     Lk = L;
%     Sk = S;
    % to save time
    Y1_mu = Y1./mu;
    Y2_mu = Y2./mu;
    Y3_mu = Y3./mu;
    Y4_mu = Y4./mu;
    Y5_mu = Y5./mu;
    mu_2 = mu * 2;
    

    %     primal_converged = false;
    %     primal_iter = 0;
    %     while primal_converged == false
    M = mu .* L - Y3;
%     M = L - Y3_mu;
    
    % update U
    % 更新 U
    tu = ((lambda + mu) .* I_r + mu .* V' * V);
    U = (U_hat .* mu + Y1 + M * V) / tu;
%     tu = (I_r + V' * V);
%     U = (U_hat + Y1_mu + M * V) / tu;    

    % update V
    % 更新 V
    tv = ((lambda + mu) .* I_r + mu .* U' * U);
    V = (V_hat .* mu + Y2 + M' * U) / tv;    
%     tv = (I_r + U' * U);
%     V = (V_hat + Y2_mu + M' * U) / tv;  
    
    % update U_hat
    % 更新 U_hat
    tu_hat = U - Y1_mu;
    [UU,sigma,UV] = svd(tu_hat,'econ');
    sigma = diag(sigma);
    svp = length(find(sigma>lambda/mu_2));
    if svp>=1
        sigma = sigma(1:svp)-lambda/mu_2;
    else
        svp = 1;
        sigma = 0;
    end
    U_hat = UU(:,1:svp) * diag(sigma) * UV(:,1:svp)';
    
    % update V_hat
    % 更新 V_hat
    tv_hat = V - Y2_mu;
    [VU,sigma,VV] = svd(tv_hat,'econ');
    sigma = diag(sigma);
    svp = length(find(sigma>lambda/mu_2));
    if svp>=1
        sigma = sigma(1:svp)-lambda/mu_2;
    else
        svp = 1;
        sigma = 0;
    end
    V_hat = VU(:,1:svp) * diag(sigma) * VV(:,1:svp)';
    
    UVT = U * V';
    
    L = 0.5 * (UVT + F - S + Y3_mu + Y4_mu);
    
    % udpate Z
    % 更新 H
    % 这里的alpha是论文中的beta
    % 使用MATLAB中矩阵"除法"来代替inv求逆
    
    t2 = alphalap + mu.*I_n;
    Z = (mu.*S + Y5) / t2; %faster and more accurate than inv(temp)
    
    % update S
    % 更新 S
    % T为论文中的 X_S
    % T = 0.5.* ( F - L + Z - Y3_mu - Y4_mu);
    T = 0.5.* ( F - L + Z + Y4_mu - Y5_mu);
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
%     relChgL = norm(L - Lk,'fro')/normfF;
%     relChgS = norm(S - Sk,'fro')/normfF;
%     relChg = max(relChgL, relChgS);
    %         if relChg < tolProj || primal_iter > 1000
    %             primal_converged = true;
    %         end
    %         primal_iter = primal_iter + 1;
    %     end
    
    % check convergence condition
    % 计算两个约束的偏差
    leq1 = U_hat - U;
    leq2 = V_hat - V;
    leq3 = UVT - L;
    % leq3 = L + S - F;
    leq4 = F - L - S;
    leq5 = S - Z;
    % 计算L和S的变化量, 取两者的最大值
    
    % 计算稀疏矩阵与低秩矩阵合成的误差
    recErrUVL = norm(leq3,'fro')/normfF;
    recErrLSF = norm(leq4,'fro')/normfF;
    %leqy1y2k = leqy1y2;
    %leqy1y2 = Y1 * pinv(V_hat) - pinv(U_hat') * Y2';
    %recErrY1Y2 = norm(leqy1y2k - leqy1y2,'fro')/normfF;
    % recErrY1Y2 = norm(leqy1y2,'fro')/normfF;
    recErrY1Y2 = 0;
    a = max([recErrUVL, recErrLSF, recErrY1Y2]);
    recErrU = norm(leq1,'fro')/normfF;
    recErrV = norm(leq2,'fro')/normfF;
    b = max(recErrU, recErrV);
    recErr = max(a, b);
    % 如果变化率和误差小于设定值, 则认为已经收敛, 循环结束
    convergenced = recErr <tol1;% && relChg < tol2;
    
    % 用于打印训练轨迹
%             if iter==1 || mod(iter,1)==0 || convergenced
%                 disp(['iter ' num2str(iter) ',mu=' num2str(mu,'%2.1e') ...
%                     ',rank=' num2str(rank(U * V')) ',stopADM=' num2str(recErr,'%2.3e')]);
%             end
    
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
        Y5 = Y5 + mu*leq5;
        % 更新罚函数系数 加速收敛
        mu = min(max_mu,mu*rho);
    end
end
