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
maxIter = 1000; % ����������
rho = 1.1; % ���ڸ��³ͷ�ϵ��
mu = 1e-1; % �ͷ�ϵ��
max_mu = 1e10; % ���ĳͷ�ϵ��
[d, n] = size(F);

% to save time
% ��Ҫ���ں�����������
normfF = norm(F,'fro');

% initialize optimization variables
L = zeros(d,n);
% ��ʵ�����ʼ��ûʲô��
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
% ��ʵ�����ʼ��Ҳûʲô��
sv_Z = 3; % for svd
sv_L = 3;
%% start main loop
iter = 0;
%disp(['initial rank=' num2str(rank(Z))]);
while iter < maxIter
    iter = iter + 1;
    
    % copy Z and S to compute the change in the solutions
    % ������ǰֵ���Ա��������仯��
    Lk = L;
    Sk = S;
    Zk = Z;
    % to save time
    Y1_mu = Y1./mu;
    Y2_mu = Y2./mu;
    Y3_mu = Y3./mu;
    Y4_mu = Y4./mu;
    
    % update M 
    % ���� M
    temp_Z = Z + Y2_mu;
    [U_Z,sigma_Z,V_Z] = lansvd(temp_Z,sv_Z,'L');
    % ���ԽǾ���ת��Ϊ�о���(Ĭ��Ϊ��������)
    sigma_Z = diag(sigma_Z);
    sigma_Z = sigma_Z(1:sv_Z);
    % ֻ����ֵ����1/mu������ֵ
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
    % ���ԽǾ���ת��Ϊ�о���(Ĭ��Ϊ��������)
    sigma_L = diag(sigma_L);
    sigma_L = sigma_L(1:sv_L);
    % ֻ����ֵ����1/mu������ֵ
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
    % ���� H
    % �����alpha�������е�beta
    % ʹ��MATLAB�о���"����"������inv����
	temp = (2*alpha).*laplacian + mu.*eye(n);
	H = (mu.*S + Y4) / temp; %faster and more accurate than inv(temp)

%     H = (alpha/mu).*F + Y4_mu + S;

    % update S 
    % ���� S
    % TΪ�����е� X_S
    T = 0.5.* ( F - F * Z - L * F + H + Y1_mu - Y4_mu);
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
                temp_Z = (gij_l1 - weightij) / gij_l1;
                S(:,gij) = temp_Z .* S(:,gij);
            else
                S(:,gij) = 0;
            end
        end
    end
    
    % check convergence condition
    % ��������Լ����ƫ��
    leq1 = F - F * Z - L * F - S;
    leq2 = Z - J;
    leq3 = L - E;
    leq4 = S - H;
    % ����L��S�ı仯��, ȡ���ߵ����ֵ
    relChgL = norm(L - Lk,'fro')/normfF;
    relChgZ = norm(Z - Zk,'fro')/normfF;
    relChgS = norm(S - Sk,'fro')/normfF;
    relChg = max([relChgL, relChgS, relChgZ]);
    % ����ϡ���������Ⱦ���ϳɵ����
    recErr = norm(leq1,'fro')/normfF; 
    
    % ����仯�ʺ����С���趨ֵ, ����Ϊ�Ѿ�����, ѭ������
    convergenced = recErr <tol1 && relChg < tol2;
    
    % ���ڴ�ӡѵ���켣
%     if iter==1 || mod(iter,1)==0 || convergenced
%         disp(['iter ' num2str(iter) ',mu=' num2str(mu,'%2.1e') ...
%           ',rank=' num2str(rank(L,1e-3*norm(L,2))) ',stopADM=' num2str(recErr,'%2.3e')]);
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
        Y4 = Y4 + mu*leq4;
        % ���·�����ϵ�� ��������
        mu = min(max_mu,mu*rho);        
    end
end
