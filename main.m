clear;clc;

exp_n = 'PSGEC'

data_name = 'COIL20_1440n_1024d_20c_CDKM200';
dir_name = [pwd, filesep, exp_n, filesep, data_name];

create_dir(dir_name);

clear BPs Y;

load(data_name);

assert(size(BPs,1)==size(Y,1));

nSmp = size(BPs,1);

nCluster = length(unique(Y));

nBase = 20;

    best_score = -inf;
    best_result = [];
    best_eta = 0;
    best_knn = 0;

    eta_list = [1, 3, 5, 7, 9];
    knn_size_list = [5,10,15,20,25];

    target_metric_idx = 1;
    %*********************************************************************
    % PSGEC
    %*********************************************************************
    fname2 = fullfile(dir_name, [data_name, '_', exp_n, '.mat']);
    if ~exist(fname2, 'file')
        nRepeat = 5;

        seed = 2025;
        rng(seed, 'twister')

        % Generate 50 random seeds
        random_seeds = randi([0, 1000000], 1, nRepeat * nRepeat );

        % Store the original state of the random number generator
        original_rng_state = rng;

        nMeasure = 15;
        % 
        for eta = eta_list
           for knn_size = knn_size_list

                
                PSGEC_result = zeros(nRepeat, nMeasure);
                PSGEC_time = zeros(nRepeat, 1);

                for iRepeat = 1:nRepeat

                    if (iRepeat ==1)
                        idx = (iRepeat - 1) * nBase + 1 : iRepeat * nBase;
                        BPi = BPs(:, idx);
                        Hns = compute_Hc_normalized(BPi);
                        Hb = compute_Hc(BPi);
                    else
                        [Hns, Hb] = updatesPartition(label,B,Hns,Hb);
                    end

                    t1_s = tic;

                    % Restore the original state of the random number generator
                    rng(original_rng_state);
                    % Set the seed for the current iteration
                    rng(random_seeds( (iRepeat-1) * nRepeat + 1 ));

                    [label, B] = PSGEC_v1(Hns, nCluster, eta, Hb, knn_size);

                    t1 = toc(t1_s);
                    result_10 = my_eval_y(label, Y);
                    PSGEC_result(iRepeat, :) = [result_10', t1];
                end
                PSGEC_result_summary = mean(PSGEC_wI_result, 1);

                score = PSGEC_result_summary(target_metric_idx);

                % Track best result and show progress
                if score > best_score
                    best_score = score;
                    best_result = PSGEC_result;
                    best_eta = eta;
                    best_knn = knn_size;

                    % Show progress
                    fprintf('[%s] New best score: %.4f with eta = %.2f, knn_size = %d\n', ...
                        data_name, best_score, eta, knn_size);

                end
            end
         end

        PSGEC_result_summary_best = mean(best_result, 1);
        PSGEC_result_summary_std_best = std(best_result);

        PSGEC_result_summary = mean(PSGEC_result, 1);
        PSGEC_result_summary_std = std(PSGEC_result);
        save(fname2, 'PSGEC_result', 'PSGEC_result_summary_best', 'PSGEC_result_summary_std_best');

        disp([data_name, ' has been completed!']);
    end
end
