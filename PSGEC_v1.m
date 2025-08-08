function [label, B] = PSGEC_v1(Hs, nCluster, eta, Hb,knn_size)

nBase = length(Hs);
nSmp = size(Hs{1}, 1);

fprintf('--- PSGEC_v1 started: eta=%.2f, knn_size=%d, nCluster=%d, nBase=%d ---\n', eta, knn_size, nCluster, nBase);

%*********************************************************************
% Step 1, get the graph
%*********************************************************************
A = zeros(nBase);
for iBase1 = 1:nBase
    Hi = Hs{iBase1};
    for iBase2 = iBase1:nBase
        Hj = Hs{iBase2};
        HiTHj = Hi' * Hj; % ki * kj * n
        A(iBase1, iBase2) = sum(sum( HiTHj.^2 )); % ki * kj
        A(iBase2, iBase1) = A(iBase1, iBase2);
    end
end
fprintf('Base graph affinity A computed.\n');

if ~exist('max_iter2', 'var')
    max_iter2 = 5;
end

if ~exist('knn_size', 'var')
    knn_size = 5;
end


w = ones(nBase, 1)/nBase;
for iter2 = 1:max_iter2
    %*********************************************************************
    % Step 1.1, update A_sparse based on sparsity constraints (e.g., row sparsity)
    %*********************************************************************
     %fprintf('  [Step 1.%d/%d] Optimizing weights...\n', iter2, max_iter2);

    Aw = zeros(nSmp);
    for iBase1 = 1:nBase
        Aw = w(iBase1) * Hs{iBase1} * Hs{iBase1}'; % n * n * ki
    end

    % Aw = Aw - 1e8 * eye(nSmp);
    % [Av, Idx] = sort(Aw, 2, 'descend');
    % Idx_k = Idx(:, 1:knn_size);
    % rIdx = repmat((1:nSmp)', knn_size, 1);
    % cIdx = Idx_k(:);
    % Av_k = Av(:, 1:knn_size);
    % vIdx = Av_k(:);
    % A_sparse = sparse(rIdx, cIdx, vIdx, nSmp, nSmp, nSmp * knn_size);


    %*********************************************************************
    % Step 1.2, update w by quadprog
    %*********************************************************************
    b = zeros(nBase, 1);
    for iBase1 = 1:nBase
        Hi = Hs{iBase1};
        HiTAs = (Hi' * Aw)';
        b(iBase1) = sum(sum( HiTAs .* Hi));
    end
    % Solve QP
    lb = zeros(nBase, 1);
    ub = ones(nBase, 1);
    Aeq = ones(1, nBase);
    beq = 1;
    options = optimoptions('quadprog', 'Display', 'off');
    w = quadprog(A, -b, [], [], Aeq, beq, lb, ub, [], options);
   % fprintf('    -> Updated weight vector: [%s]\n', num2str(w', '%.3f '));
end
fprintf('Initial weight vector w calculated.\n');
% A_jac = (Hb * Hb')/nBase;
% A_jac = A_jac - diag(diag(A_jac));
% [bcs, baseClsSegs] = getAllSegs(Hb');
% A_jac = full(simxjac(baseClsSegs));


%*********************************************************************
% Step 2: Graph filter via diffusion (Graph Enhancement)
%*********************************************************************
fprintf('  [Step 2] Building sparse graph with knn_size=%d\n', knn_size);

if ~exist('alpha', 'var')
    alpha = 0.85;
end
% Aw = Aw - 1e8 * eye(nSmp);
% G = Network_Enhancement(Aw);

Aw = Aw - 1e8 * eye(nSmp);
[Av, Idx] = sort(Aw, 2, 'descend');
Idx_k = Idx(:, 1:knn_size);
rIdx = repmat((1:nSmp)', knn_size, 1);
cIdx = Idx_k(:);
Av_k = Av(:, 1:knn_size);
vIdx = Av_k(:);
A_sparse = sparse(rIdx, cIdx, vIdx, nSmp, nSmp, nSmp * knn_size);

%G = Network_Enhancement(Aw);

%G = Network_Enhancement(A_sparse);

A_dsm = normalize_to_DSM(A_sparse);
% A_dsm = normalize_to_DSM(Aw);

% Step 8: Compute the Laplacian matrix L
L = eye(nSmp) - A_dsm;


% Step 9: Compute the exponential of the Laplacian matrix
% if ~exist('eta', 'var')
%     eta = 1;
% end
fprintf('  [Step 2] Performing diffusion with eta = %.2f...\n', eta);

G = expm(-eta * L);

%*********************************************************************
% Step 3: Smooth Kmeans Consensus Clustering
%*********************************************************************
fprintf('  [Step 3] Performing clustering...\n');
% ECA
% t = 20;
% [bcs, baseClsSegs] = getAllSegs(Hb');
% clsSim = full(simxjac(baseClsSegs));
% clsSimRW = computePTS_II(clsSim, t);
% Hc_new = Hb * clsSimRW;
% Hc_new = clsSimRW * Hb;

% Ha = cell2mat(Hs);
Hc_new = G * Hb; % n * n * d
Hc_new = NormalizeFea(Hc_new);
[label, B] = litekmeans(Hc_new, nCluster, 'MaxIter', 50, 'Replicates', 10);

fprintf('  Cluster label histogram: ');
disp(histcounts(label, nCluster));

fprintf('--- PSGEC_v1 completed ---\n\n');

 % [~, C_0] = kmeanspp(Hc_new', nCluster);
 % [label, ~] = litekmeans(Hc_new, nCluster, 'MaxIter', 50, 'Start', C_0');
end