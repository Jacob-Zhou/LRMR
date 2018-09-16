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
maxIter = 1000; % ����������
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
L = zeros(d,n);
% ��ʵ�����ʼ��ûʲô��
Z = zeros(d,n); % �����е� H
S = zeros(d,n);
Y1 = zeros(d,n);
Y2 = zeros(d,n);
Q = zeros(d, r);
R = eye(r, n);
% ��ʵ�����ʼ��Ҳûʲô��
svp = 5; % for svd
alphalap = (2*alpha).*laplacian;
I_n = eye(n);
%% start main loop
iter = 0;
%disp(['initial rank=' num2str(rank(Z))]);
while iter < maxIter
    iter = iter + 1;
    
    % copy Z and S to compute the change in the solutions
    % ������ǰֵ���Ա��������仯��
    Lk = L;
    Sk = S;
    % to save time
    Y1_mu = Y1./mu;
    Y2_mu = Y2./mu;

    P = F - S + Y1_mu;
    [Q, ~] = qr(P * R', 0);
    
    % update M 
    % ���� M
    t1 = Q' * P;
    [U,sigma,V] = svd(t1,'econ');
    % ���ԽǾ���ת��Ϊ�о���(Ĭ��Ϊ��������)
    sigma = diag(sigma);
    % ֻ����ֵ����1/mu������ֵ
    svp = length(find(sigma>1/mu));
    if svp>=1
        sigma = sigma(1:svp)-1/mu;
    else
        svp = 1;
        sigma = 0; % ������Ϊʲô��������
    end
    R = U(:,1:svp) * diag(sigma) * V(:,1:svp)';
    L = Q * R;
    % save time
    
    % udpate Z
    % ���� H
    % �����alpha�������е�beta
    % ʹ��MATLAB�о���"����"������inv����
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
    % ���� S
    % TΪ�����е� X_S
    T = 0.5.* ( F - L + Z + Y1_mu - Y2_mu);
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
    
    
    % check convergence condition
    % ��������Լ����ƫ��
    leq1 = F - L - S;
    leq2 = S - Z;
    % leq3 = L - T;
    % ����L��S�ı仯��, ȡ���ߵ����ֵ
    relChgL = norm(L - Lk,'fro')/normfF;
    relChgS = norm(S - Sk,'fro')/normfF;
    relChg = max(relChgL, relChgS);
    % ����ϡ���������Ⱦ���ϳɵ����
    recErr = norm(leq1,'fro')/normfF; 

    
    % ����仯�ʺ����С���趨ֵ, ����Ϊ�Ѿ�����, ѭ������
    convergenced = recErr <tol1 && relChg < tol2;
    
    % ���ڴ�ӡѵ���켣
%     if iter==1 || mod(iter,1000)==0 || convergenced
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
        % ���·�����ϵ�� ��������
        mu = min(max_mu,mu*rho);        
    end
end
