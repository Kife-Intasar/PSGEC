function [Hns, Hb] = updatesPartition(label,B,Hns,Hb)

G_proj = ind2vec(label')'*B;
G_proj = NormalizeFea(G_proj);
entropy_type = 'Gini1';
entropies = zeros(1, length(Hns));
for k = 1:length(Hns)
    H_part = Hns{k};  % likely a matrix [nSamples × dim]

    % Compute entropy over each column, reduce to scalar
    entropy_per_col = compute_entropy(mean(abs(H_part), 1), entropy_type);  % use columnwise mean as vector

    entropies(k) = sum(entropy_per_col);  % now guaranteed to be scalar
end
% Find the worst part index by entropy (max entropy is worst)
[~,worst_idx] = max(entropies);
% Remove worst element from Hns
Hns(worst_idx) = [];

numColsPerPart = size(Hns{1},2);
start_col = (worst_idx - 1) * numColsPerPart + 1;
end_col = worst_idx * numColsPerPart;
Hb(:, start_col:end_col) = [];

Hns{end+1} = G_proj;
Hb = [Hb, G_proj];

end