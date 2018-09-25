% function [U, V, S, iter] = BFMN_SMD( F, laplacian, tree, weight, alpha, beta, r)
function [U, V, S, iter] = FNN_SMD( F, laplacian, tree, weight, alpha, beta, r)
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
maxIter = 500000; % ����������
rho = 1.1; % ���ڸ��³ͷ�ϵ��
mu = 0.1; % �ͷ�ϵ��
max_mu = 1e10; % ���ĳͷ�ϵ��
[d, n] = size(F);

% to save time
% ��Ҫ���ں�����������
normfF = norm(F,'fro');
if r <= 0
    r = round(1.2*rank_estimation(F));
end
% initialize optimization variables
% L = zeros(d,n);
UVT = zeros(d,n);
% ��ʵ�����ʼ��ûʲô��
Z = zeros(d,n); % �����е� H
U = eye(d, r);
V = eye(n, r);
% U_hat = zeros(d, r);
V_hat = zeros(n, r);
S = zeros(d, n);
Y1 = zeros(n, r);
% Y3 = zeros(d, n);
Y3 = zeros(d, n);
a1 = lansvd(F, 1, 'L'); 
a2 = norm(F, Inf); 
Y2 = F/max(a1,a2);
M = zeros(d, r);
% ��ʵ�����ʼ��Ҳûʲô��
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
    % ������ǰֵ���Ա��������仯��
    % to save time
    Y1_mu = Y1./mu;
    Y2_mu = Y2./mu;
    Y3_mu = Y3./mu;
    mu_3 = mu * 3;
    
    % M = F - S + Y3_mu;
    M = mu .* (F - S) - Y2;
    
%         primal_converged = false;
%         primal_iter = 0;
%         while primal_converged == false
%     % update U
%     % ���� U
%     tu = (I_r + V' * V);
%     U = (U_hat + Y1_mu + M * V) / tu;
%     
%     % update V
%     % ���� V
%     tv = (I_r + U' * U);
%     V = (V_hat + Y2_mu + M' * U) / tv;
%     L = U * V';

    % update U
    % ���� U
    tu = ((2/(3 * mu)) .* I_r + V' * V);
    U = (F - S - Y2_mu) * V / tu;
    
    % update V
    % ���� V
    tv = ((lambda + mu) .* I_r + mu .* U' * U);
    V = (V_hat .* mu + Y1 + M' * U) / tv;    
    UVT = U * V';
    
    % update V_hat
    % ���� V_hat
    tv_hat = V - Y1_mu;
    [VU,sigma,VV] = svd(tv_hat,'econ');
    sigma = diag(sigma);
    svp = length(find(sigma>lambda_2/mu_3));
    if svp>=1
        sigma = sigma(1:svp)-lambda_2/mu_3;
    else
        svp = 1;
        sigma = 0;
    end
    V_hat = VU(:,1:svp) * diag(sigma) * VV(:,1:svp)';
    
    % udpate Z
    % ���� H
    % �����alpha�������е�beta
    % ʹ��MATLAB�о���"����"������inv����
    
    t2 = alphalap + mu.*I_n;
    Z = (mu.*S + Y3) / t2; %faster and more accurate than inv(temp)
    
    % update S
    % ���� S
    % TΪ�����е� X_S
    T = 0.5.* ( F - UVT + Z - Y2_mu - Y3_mu);
    % �����beta�������е�alpha
    lambada = 0.5*beta/mu;
    
    S = T;
    % ��ѭ��Ϊ�������(�Զ�����)
    for i = 1:length(tree)
        % ���ڲ�ѭ�������ÿ�������ص�norm_1
        Sl1 = sum(abs(S));
        % ��ѭ��Ϊ����ǰ��ȵĽ��
        for j = 1:length(tree{i})
            % ��ȡ��ǰ�������ĳ�����
            gij = tree{i}{j};
            % ���㵱ǰ�������ĳ����������ɵľ���ĵ�norm_1
            gij_l1 = sum(Sl1(gij)); %%%%
            % weightij Ϊ�����е� v_ij
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
%             if relChg < tolProj || primal_iter > 1000
%                 primal_converged = true;
%             end
%             primal_iter = primal_iter + 1;
%         end
    
    % check convergence condition
    % ��������Լ����ƫ��
    leq1 = V_hat - V;
    leq2 = UVT + S - F;
    leq3 = S - Z;
    % ����L��S�ı仯��, ȡ���ߵ����ֵ
    
    % ����ϡ���������Ⱦ���ϳɵ����
    recErrV = norm(leq1,'fro')/normfF;
    recErrLSF = norm(leq2,'fro')/normfF;
    recErr = max(recErrLSF, recErrV);
    % ����仯�ʺ����С���趨ֵ, ����Ϊ�Ѿ�����, ѭ������
    convergenced = recErr <tol1;
    
    % ���ڴ�ӡѵ���켣
%     if iter==1 || mod(iter,1)==0 || convergenced
%         disp(['iter ' num2str(iter) ',mu=' num2str(mu,'%2.1e') ...
%             ',rank=' num2str(rank(U * V')) ',stopADM=' num2str(recErr,'%2.3e')]);
%     end
    
    if convergenced
        % ������ʾѵ������
        % disp('SMF done.');
        break;
    else
        % �����������ճ���Y1��Y2
        Y1 = Y1 + mu*leq1;
        Y2 = Y2 + mu*leq2;
        Y3 = Y3 + mu*leq3;
        % ���·�����ϵ�� ��������
        mu = min(max_mu,mu*rho);
    end
end
