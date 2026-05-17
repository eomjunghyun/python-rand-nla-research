classdef Common
    % Standalone MATLAB implementation of Reference 1 Section 7.2 utilities.

    methods (Static)
        function methods = methodNames()
            methods = {'Random Projection', 'Random Sampling', 'Non-random'};
        end

        function methods = summaryMethodNames()
            methods = {'Non-random', 'Random Projection', 'Random Sampling'};
        end

        function c = methodColor(methodName)
            switch char(methodName)
                case 'Random Projection'
                    c = [31, 119, 180] ./ 255;
                case 'Random Sampling'
                    c = [255, 127, 14] ./ 255;
                case 'Non-random'
                    c = [44, 160, 44] ./ 255;
                otherwise
                    c = [0, 0, 0];
            end
        end

        function nValues = parseNValues(value)
            if isnumeric(value)
                nValues = double(value(:))';
                return;
            end
            parts = split(string(value), ",");
            nValues = str2double(strtrim(parts));
            nValues = nValues(~isnan(nValues))';
        end

        function raw = runExperimentModels123(cfg, showProgress, thetaMode, detailedTiming)
            if nargin < 2
                showProgress = true;
            end
            if nargin < 3
                thetaMode = 'exact';
            end
            if nargin < 4
                detailedTiming = false;
            end

            master = RandStream('mt19937ar', 'Seed', cfg.seed);
            records = struct([]);
            totalSteps = numel(cfg.model_ids) * numel(cfg.n_values) * cfg.reps * 3;
            doneSteps = 0;
            tGlobal = tic;

            for modelId = cfg.model_ids
                for n = cfg.n_values
                    for rep = 1:cfg.reps
                        repSeed = randi(master, 2147483646, 1, 1);
                        rs = RandStream('mt19937ar', 'Seed', repSeed);

                        t0 = tic;
                        [A, P, BTrue, yTrue] = sec72.Common.generateModel123Instance(modelId, n, rs);
                        instanceSec = toc(t0);
                        eyeK = eye(cfg.K);
                        ThetaTrue = eyeK(yTrue, :);

                        if modelId == 3
                            KPrime = cfg.K_prime_rankdef;
                        else
                            KPrime = cfg.K_prime_fullrank;
                        end

                        baseRecord = struct('model', modelId, 'n', n, 'rep', rep);
                        jobs = sec72.Common.methodNames();
                        for j = 1:numel(jobs)
                            methodName = jobs{j};
                            record = sec72.Common.runOneMethod( ...
                                methodName, baseRecord, A, P, BTrue, ThetaTrue, yTrue, ...
                                cfg.K, KPrime, cfg.r, cfg.q, cfg.p, rs, false, thetaMode, ...
                                detailedTiming, instanceSec);
                            records = sec72.Common.appendRecord(records, record);
                            doneSteps = doneSteps + 1;
                            if showProgress
                                sec72.Common.printProgress(doneSteps, totalSteps, 'model/n', ...
                                    sprintf('%d/%d', modelId, n), rep, cfg.reps, methodName, tGlobal);
                            end
                        end
                    end
                end
            end

            if showProgress
                fprintf('\n');
            end
            raw = struct2table(records);
            raw = sec72.Common.orderRawColumns(raw, detailedTiming);
        end

        function raw = runExperimentModels456(cfg, showProgress, thetaMode, detailedTiming)
            if nargin < 2
                showProgress = true;
            end
            if nargin < 3
                thetaMode = 'exact';
            end
            if nargin < 4
                detailedTiming = false;
            end

            master = RandStream('mt19937ar', 'Seed', cfg.seed);
            records = struct([]);
            totalSteps = numel(cfg.model_ids) * numel(cfg.n_values) * cfg.reps * 3;
            doneSteps = 0;
            tGlobal = tic;

            for modelId = cfg.model_ids
                for n = cfg.n_values
                    for rep = 1:cfg.reps
                        repSeed = randi(master, 2147483646, 1, 1);
                        rs = RandStream('mt19937ar', 'Seed', repSeed);

                        t0 = tic;
                        [A, P, BTrue, yTrue] = sec72.Common.generateModel456Instance(modelId, n, rs);
                        instanceSec = toc(t0);
                        eyeK = eye(cfg.K);
                        ThetaTrue = eyeK(yTrue, :);

                        if modelId == 6
                            KPrime = cfg.K_prime_rankdef;
                        else
                            KPrime = cfg.K_prime_fullrank;
                        end

                        baseRecord = struct('model', modelId, 'n', n, 'rep', rep);
                        jobs = sec72.Common.methodNames();
                        for j = 1:numel(jobs)
                            methodName = jobs{j};
                            record = sec72.Common.runOneMethod( ...
                                methodName, baseRecord, A, P, BTrue, ThetaTrue, yTrue, ...
                                cfg.K, KPrime, cfg.r, cfg.q, cfg.p, rs, true, thetaMode, ...
                                detailedTiming, instanceSec);
                            records = sec72.Common.appendRecord(records, record);
                            doneSteps = doneSteps + 1;
                            if showProgress
                                sec72.Common.printProgress(doneSteps, totalSteps, 'model/n', ...
                                    sprintf('%d/%d', modelId, n), rep, cfg.reps, methodName, tGlobal);
                            end
                        end
                    end
                end
            end

            if showProgress
                fprintf('\n');
            end
            raw = struct2table(records);
            raw = sec72.Common.orderRawColumns(raw, detailedTiming);
        end

        function records = appendRecord(records, record)
            if isempty(records)
                records = record;
            else
                records(end + 1) = record; %#ok<AGROW>
            end
        end

        function raw = orderRawColumns(raw, detailedTiming)
            cols = {'model', 'n', 'rep', 'method', 'error_P', 'error_Theta', 'error_B', 'time_sec'};
            if detailedTiming
                extra = sec72.Common.timingFieldNames();
                cols = [cols, extra];
            end
            cols = cols(ismember(cols, raw.Properties.VariableNames));
            raw = raw(:, cols);
        end

        function record = runOneMethod(methodName, baseRecord, A, P, BTrue, ThetaTrue, yTrue, ...
                K, KPrime, r, q, p, rs, normalizeRows, thetaMode, detailedTiming, instanceSec)
            switch char(methodName)
                case 'Random Projection'
                    [AHat, yPred, timing] = sec72.Common.runRandomProjection(A, K, KPrime, r, q, rs, normalizeRows);
                case 'Random Sampling'
                    [AHat, yPred, timing] = sec72.Common.runRandomSampling(A, K, KPrime, p, rs, normalizeRows);
                case 'Non-random'
                    [AHat, yPred, timing] = sec72.Common.runNonRandom(A, K, KPrime, rs, normalizeRows);
                otherwise
                    error('Unknown method: %s', methodName);
            end

            t0 = tic;
            [errP, errTheta, errB] = sec72.Common.evaluateMetrics(AHat, yPred, P, BTrue, ThetaTrue, yTrue, K, thetaMode);
            metricSec = toc(t0);

            record = baseRecord;
            record.method = methodName;
            record.error_P = errP;
            record.error_Theta = errTheta;
            record.error_B = errB;
            record.time_sec = timing.algo_total_sec;

            if detailedTiming
                record = sec72.Common.attachTimingBreakdown(record, timing, instanceSec, metricSec);
            end
        end

        function [A, P, BTrue, yTrue] = generateModel123Instance(modelId, n, rs)
            if ~ismember(modelId, [1, 2, 3])
                error('modelId must be one of 1, 2, 3.');
            end

            if modelId == 2
                sizes = sec72.Common.sizesFromProportions(n, [1/6, 1/2, 1/3]);
            else
                sizes = sec72.Common.sizesFromProportions(n, ones(1, 3) / 3);
            end

            yTrue = sec72.Common.labelsFromSizes(sizes, rs);
            if ismember(modelId, [1, 2])
                BTrue = sec72.Common.buildBModel12(rs);
            else
                BTrue = sec72.Common.buildBModel3();
            end

            P = BTrue(yTrue, yTrue);
            P(1:size(P, 1) + 1:end) = 0;
            A = sec72.Common.sampleSymmetricAdjacency(P, rs);
        end

        function [A, P, BTrue, yTrue] = generateModel456Instance(modelId, n, rs)
            if ~ismember(modelId, [4, 5, 6])
                error('modelId must be one of 4, 5, 6.');
            end

            sizes = sec72.Common.sizesFromProportions(n, ones(1, 3) / 3);
            yTrue = sec72.Common.labelsFromSizes(sizes, rs);

            if ismember(modelId, [4, 5])
                BTrue = sec72.Common.buildBModel45(rs);
            else
                BTrue = sec72.Common.buildBModel3();
            end

            if ismember(modelId, [4, 6])
                theta = sec72.Common.sampleThetaModel4(yTrue, rs);
            else
                theta = sec72.Common.sampleThetaModel5(yTrue, rs);
            end

            P = (theta * theta') .* BTrue(yTrue, yTrue);
            P = min(max(P, 0), 1);
            P(1:size(P, 1) + 1:end) = 0;
            A = sec72.Common.sampleSymmetricAdjacency(P, rs);
        end

        function sizes = sizesFromProportions(n, proportions)
            raw = proportions(:)' * n;
            sizes = floor(raw);
            remain = n - sum(sizes);
            if remain > 0
                [~, order] = sort(-(raw - sizes));
                for i = 1:remain
                    sizes(order(i)) = sizes(order(i)) + 1;
                end
            end
            sizes(end) = sizes(end) + n - sum(sizes);
            sizes = double(sizes);
        end

        function labels = labelsFromSizes(sizes, rs)
            labels = [];
            for k = 1:numel(sizes)
                labels = [labels; repmat(k, sizes(k), 1)]; %#ok<AGROW>
            end
            labels = labels(randperm(rs, numel(labels)));
        end

        function A = sampleSymmetricAdjacency(P, rs)
            n = size(P, 1);
            tri = find(triu(true(n), 1));
            probs = min(max(P(tri), 0), 1);
            edges = double(rand(rs, numel(probs), 1) < probs(:));
            A = zeros(n, n);
            A(tri) = edges;
            A = A + A';
            A(1:n + 1:end) = 0;
        end

        function B = buildBModel12(rs)
            K = 3;
            B = zeros(K, K);
            for i = 1:K
                B(i, i) = 0.2 + 0.1 * rand(rs);
            end
            for i = 1:K
                for j = (i + 1):K
                    v = 0.01 + 0.09 * rand(rs);
                    B(i, j) = v;
                    B(j, i) = v;
                end
            end
        end

        function B = buildBModel45(rs)
            K = 3;
            B = zeros(K, K);
            for i = 1:K
                B(i, i) = 0.4 + 0.2 * rand(rs);
            end
            for i = 1:K
                for j = (i + 1):K
                    v = 0.01 + 0.19 * rand(rs);
                    B(i, j) = v;
                    B(j, i) = v;
                end
            end
        end

        function B = buildBModel3()
            C = [
                2 * sin(pi / 3), 2 * cos(pi / 3);
                sin(pi / 5), 2 * cos(pi / 5);
                (2 / 5) * sin(2 * pi / 5), (6 / 5) * cos(2 * pi / 5)
            ];
            B = C * C';
            B = B / max(1, max(B, [], 'all'));
        end

        function theta = sampleThetaModel4(labels, rs)
            theta = zeros(numel(labels), 1);
            classes = unique(labels(:))';
            for k = classes
                idx = find(labels == k);
                u = rand(rs, numel(idx), 1);
                thetaK = ones(numel(idx), 1);
                thetaK(u <= 0.8) = 0.2;
                thetaK = thetaK ./ max(1e-12, max(thetaK));
                theta(idx) = thetaK;
            end
        end

        function theta = sampleThetaModel5(labels, rs)
            theta = zeros(numel(labels), 1);
            classes = unique(labels(:))';
            for k = classes
                idx = find(labels == k);
                u = rand(rs, numel(idx), 1);
                thetaK = ones(numel(idx), 1);
                thetaK(u <= 0.4) = 0.1;
                thetaK(u > 0.4 & u <= 0.8) = 0.2;
                thetaK = thetaK ./ max(1e-12, max(thetaK));
                theta(idx) = thetaK;
            end
        end

        function [AHat, labels, timing] = runNonRandom(A, K, KPrime, rs, normalizeRows)
            totalStart = tic;
            t0 = tic;
            U = sec72.Common.topEigvecsSymmetric(A, KPrime);
            timing.nr_eig_sec = toc(t0);

            t0 = tic;
            labels = sec72.Common.kmeansOnRows(U, K, rs, normalizeRows);
            timing.nr_kmeans_sec = toc(t0);

            t0 = tic;
            AHat = A;
            timing.nr_copy_sec = toc(t0);
            timing = sec72.Common.finalizeTiming(timing, totalStart);
        end

        function [AHat, labels, timing] = runRandomProjection(A, K, KPrime, r, q, rs, normalizeRows)
            totalStart = tic;
            n = size(A, 1);

            t0 = tic;
            Omega = randn(rs, n, KPrime + r);
            timing.rp_draw_omega_sec = toc(t0);

            t0 = tic;
            Y = Omega;
            for iter = 1:(2 * q + 1)
                Y = A * Y;
            end
            timing.rp_power_iter_sec = toc(t0);

            t0 = tic;
            [Q, ~] = qr(Y, 0);
            timing.rp_qr_sec = toc(t0);

            t0 = tic;
            C = Q' * A * Q;
            C = 0.5 * (C + C');
            timing.rp_build_core_sec = toc(t0);

            t0 = tic;
            AHat = Q * C * Q';
            timing.rp_reconstruct_sec = toc(t0);

            t0 = tic;
            Uc = sec72.Common.topEigvecsSymmetric(C, KPrime);
            timing.rp_small_eig_sec = toc(t0);

            t0 = tic;
            Urp = Q * Uc;
            timing.rp_lift_sec = toc(t0);

            t0 = tic;
            labels = sec72.Common.kmeansOnRows(Urp, K, rs, normalizeRows);
            timing.rp_kmeans_sec = toc(t0);
            timing = sec72.Common.finalizeTiming(timing, totalStart);
        end

        function [AHat, labels, timing] = runRandomSampling(A, K, KPrime, p, rs, normalizeRows)
            totalStart = tic;
            n = size(A, 1);

            t0 = tic;
            tri = find(triu(true(n), 1));
            mask = double(rand(rs, numel(tri), 1) < p);
            timing.rs_sample_mask_sec = toc(t0);

            t0 = tic;
            AS = zeros(size(A));
            AS(tri) = A(tri) .* mask ./ p;
            AS = AS + AS';
            AS(1:n + 1:end) = 0;
            timing.rs_build_sampled_matrix_sec = toc(t0);

            t0 = tic;
            [vals, vecs] = sec72.Common.topEigpairsSymmetric(AS, KPrime);
            timing.rs_eig_sec = toc(t0);

            t0 = tic;
            AHat = vecs * diag(vals) * vecs';
            timing.rs_reconstruct_sec = toc(t0);

            t0 = tic;
            AHat = 0.5 * (AHat + AHat');
            timing.rs_symmetrize_sec = toc(t0);

            t0 = tic;
            labels = sec72.Common.kmeansOnRows(vecs, K, rs, normalizeRows);
            timing.rs_kmeans_sec = toc(t0);
            timing = sec72.Common.finalizeTiming(timing, totalStart);
        end

        function U = topEigvecsSymmetric(M, k)
            [~, U] = sec72.Common.topEigpairsSymmetric(M, k);
        end

        function [vals, vecs] = topEigpairsSymmetric(M, k)
            M = 0.5 * (M + M');
            n = size(M, 1);
            useFull = n <= max(40, k + 2) || k >= n;
            if useFull
                [V, D] = eig(full(M), 'vector');
                vals = real(D(:));
                vecs = real(V);
            else
                opts.tol = 1e-8;
                opts.maxit = 1000;
                try
                    [V, D] = eigs(M, k, 'largestreal', opts);
                    vals = real(diag(D));
                    vecs = real(V);
                catch
                    [V, D] = eig(full(M), 'vector');
                    vals = real(D(:));
                    vecs = real(V);
                end
            end
            [vals, order] = sort(vals, 'descend');
            order = order(1:k);
            vals = vals(1:k);
            vecs = vecs(:, order);
        end

        function Xn = normalizeRowsL2(X)
            norms = sqrt(sum(X .^ 2, 2));
            Xn = zeros(size(X));
            idx = norms > 1e-12;
            Xn(idx, :) = X(idx, :) ./ norms(idx);
        end

        function labels = kmeansOnRows(U, K, rs, normalizeRows)
            if normalizeRows
                X = sec72.Common.normalizeRowsL2(U);
            else
                X = U;
            end
            X = real(double(X));
            n = size(X, 1);
            if n < K
                error('kmeansOnRows requires n >= K.');
            end

            bestObjective = inf;
            bestLabels = ones(n, 1);
            nStarts = 20;
            maxIter = 100;

            for start = 1:nStarts
                centers = sec72.Common.initKMeansPlusPlus(X, K, rs);
                labels = zeros(n, 1);
                for iter = 1:maxIter
                    oldLabels = labels;
                    dist = sec72.Common.pairwiseSquaredDistances(X, centers);
                    [~, labels] = min(dist, [], 2);
                    for c = 1:K
                        idx = labels == c;
                        if any(idx)
                            centers(c, :) = mean(X(idx, :), 1);
                        else
                            centers(c, :) = X(randi(rs, n), :);
                        end
                    end
                    if isequal(labels, oldLabels)
                        break;
                    end
                end
                dist = sec72.Common.pairwiseSquaredDistances(X, centers);
                minDist = min(dist, [], 2);
                objective = sum(minDist);
                if objective < bestObjective
                    bestObjective = objective;
                    bestLabels = labels;
                end
            end
            labels = bestLabels;
        end

        function centers = initKMeansPlusPlus(X, K, rs)
            n = size(X, 1);
            centers = zeros(K, size(X, 2));
            first = randi(rs, n);
            centers(1, :) = X(first, :);
            minDist = sec72.Common.pairwiseSquaredDistances(X, centers(1, :));

            for c = 2:K
                total = sum(minDist);
                if total <= 0 || ~isfinite(total)
                    idx = randi(rs, n);
                else
                    threshold = rand(rs) * total;
                    cumulative = cumsum(minDist);
                    idx = find(cumulative >= threshold, 1, 'first');
                    if isempty(idx)
                        idx = n;
                    end
                end
                centers(c, :) = X(idx, :);
                distNew = sec72.Common.pairwiseSquaredDistances(X, centers(c, :));
                minDist = min(minDist, distNew);
            end
        end

        function dist = pairwiseSquaredDistances(X, C)
            x2 = sum(X .^ 2, 2);
            c2 = sum(C .^ 2, 2)';
            dist = x2 + c2 - 2 * (X * C');
            dist = max(dist, 0);
        end

        function [errP, errTheta, errB] = evaluateMetrics(AHat, yPred, P, BTrue, ThetaTrue, yTrue, K, thetaMode)
            errP = sec72.Common.spectralNormSym(AHat - P);
            switch char(thetaMode)
                case 'exact'
                    [errTheta, ThetaHat] = sec72.Common.thetaErrorExact(ThetaTrue, yTrue, yPred, K);
                case 'hungarian'
                    [errTheta, ThetaHat] = sec72.Common.thetaErrorExact(ThetaTrue, yTrue, yPred, K);
                otherwise
                    error('Unknown thetaMode: %s', thetaMode);
            end
            BHat = sec72.Common.estimateBHat(AHat, ThetaHat);
            errB = max(abs(BHat - BTrue), [], 'all');
        end

        function val = spectralNormSym(M)
            M = 0.5 * (M + M');
            n = size(M, 1);
            opts.tol = 1e-8;
            opts.maxit = 1000;
            try
                lambda = eigs(M, 1, 'largestabs', opts);
                val = abs(real(lambda(1)));
            catch
                ev = eig(full(M), 'vector');
                val = max(abs(real(ev)));
            end
        end

        function [errTheta, ThetaHatBest, bestPerm] = thetaErrorExact(ThetaTrue, yTrue, yPred, K)
            permsK = perms(1:K);
            eyeK = eye(K);
            bestVal = inf;
            ThetaHatBest = [];
            bestPerm = [];

            for i = 1:size(permsK, 1)
                perm = permsK(i, :);
                mapped = perm(yPred);
                ThetaHat = eyeK(mapped, :);
                val = 0;
                for k = 1:K
                    idx = find(yTrue == k);
                    nk = numel(idx);
                    if nk == 0
                        continue;
                    end
                    diff = ThetaHat(idx, :) - ThetaTrue(idx, :);
                    val = val + nnz(diff) / (2 * nk);
                end
                if val < bestVal
                    bestVal = val;
                    ThetaHatBest = ThetaHat;
                    bestPerm = perm;
                end
            end
            errTheta = bestVal;
        end

        function BHat = estimateBHat(AHat, ThetaHat)
            num = ThetaHat' * AHat * ThetaHat;
            counts = sum(ThetaHat, 1);
            den = counts' * counts;
            BHat = zeros(size(num));
            idx = den > 0;
            BHat(idx) = num(idx) ./ den(idx);
        end

        function timing = finalizeTiming(timing, totalStart)
            totalSec = toc(totalStart);
            names = fieldnames(timing);
            stepSum = 0;
            for i = 1:numel(names)
                if endsWith(names{i}, '_sec')
                    stepSum = stepSum + double(timing.(names{i}));
                end
            end
            timing.algo_total_sec = totalSec;
            timing.algo_step_sum_sec = stepSum;
            timing.algo_other_sec = max(0, totalSec - stepSum);
        end

        function fields = timingFieldNames()
            fields = {
                'instance_gen_sec', 'metric_eval_sec', ...
                'algo_total_sec', 'algo_step_sum_sec', 'algo_other_sec', 'pipeline_total_sec', ...
                'nr_eig_sec', 'nr_kmeans_sec', 'nr_copy_sec', ...
                'rs_sample_mask_sec', 'rs_build_sampled_matrix_sec', 'rs_eig_sec', ...
                'rs_reconstruct_sec', 'rs_symmetrize_sec', 'rs_kmeans_sec', ...
                'rp_draw_omega_sec', 'rp_power_iter_sec', 'rp_qr_sec', ...
                'rp_build_core_sec', 'rp_reconstruct_sec', 'rp_small_eig_sec', ...
                'rp_lift_sec', 'rp_kmeans_sec'
            };
        end

        function record = attachTimingBreakdown(record, timing, instanceSec, metricSec)
            fields = sec72.Common.timingFieldNames();
            for i = 1:numel(fields)
                record.(fields{i}) = NaN;
            end
            record.instance_gen_sec = instanceSec;
            record.metric_eval_sec = metricSec;
            timingFields = fieldnames(timing);
            for i = 1:numel(timingFields)
                record.(timingFields{i}) = timing.(timingFields{i});
            end
            record.pipeline_total_sec = timing.algo_total_sec + instanceSec + metricSec;
        end

        function summary = summarizeMetrics(raw)
            rows = struct([]);
            models = unique(raw.model)';
            nValues = unique(raw.n)';
            methodOrder = sec72.Common.summaryMethodNames();
            for modelId = models
                for n = nValues
                    for mi = 1:numel(methodOrder)
                        methodName = methodOrder{mi};
                        mask = raw.model == modelId & raw.n == n & strcmp(raw.method, methodName);
                        if ~any(mask)
                            continue;
                        end
                        row = struct();
                        row.model = modelId;
                        row.n = n;
                        row.method = methodName;
                        row.error_P_mean = mean(raw.error_P(mask));
                        row.error_P_std = sec72.Common.sampleStd(raw.error_P(mask));
                        row.error_Theta_mean = mean(raw.error_Theta(mask));
                        row.error_Theta_std = sec72.Common.sampleStd(raw.error_Theta(mask));
                        row.error_B_mean = mean(raw.error_B(mask));
                        row.error_B_std = sec72.Common.sampleStd(raw.error_B(mask));
                        row.time_mean = mean(raw.time_sec(mask));
                        row.time_std = sec72.Common.sampleStd(raw.time_sec(mask));
                        rows = sec72.Common.appendRecord(rows, row);
                    end
                end
            end
            summary = struct2table(rows);
        end

        function s = sampleStd(values)
            values = values(:);
            if numel(values) < 2
                s = NaN;
            else
                s = std(values, 0);
            end
        end

        function timingRaw = extractTimingBreakdown(raw, idCols)
            names = raw.Properties.VariableNames;
            secCols = names(endsWith(names, '_sec'));
            cols = [idCols, secCols];
            cols = cols(ismember(cols, names));
            timingRaw = raw(:, cols);
        end

        function timingSummary = summarizeTimingBreakdown(timingRaw, groupCols)
            methodOrder = sec72.Common.summaryMethodNames();
            names = timingRaw.Properties.VariableNames;
            timingCols = names(endsWith(names, '_sec'));
            models = unique(timingRaw.model)';
            nValues = unique(timingRaw.n)';
            rows = struct([]);
            for modelId = models
                for n = nValues
                    for mi = 1:numel(methodOrder)
                        methodName = methodOrder{mi};
                        mask = timingRaw.model == modelId & timingRaw.n == n & strcmp(timingRaw.method, methodName);
                        if ~any(mask)
                            continue;
                        end
                        row = struct();
                        for gi = 1:numel(groupCols)
                            g = groupCols{gi};
                            row.(g) = timingRaw.(g)(find(mask, 1, 'first'));
                        end
                        row.method = methodName;
                        for ci = 1:numel(timingCols)
                            col = timingCols{ci};
                            row.([col, '_mean']) = mean(timingRaw.(col)(mask), 'omitnan');
                            row.([col, '_std']) = sec72.Common.sampleStd(timingRaw.(col)(mask & ~isnan(timingRaw.(col))));
                        end
                        rows = sec72.Common.appendRecord(rows, row);
                    end
                end
            end
            timingSummary = struct2table(rows);
        end

        function printProgress(doneSteps, totalSteps, xName, xValue, rep, reps, methodName, tGlobal)
            elapsed = toc(tGlobal);
            ratio = doneSteps / max(1, totalSteps);
            rate = doneSteps / max(eps, elapsed);
            remain = totalSteps - doneSteps;
            eta = remain / max(eps, rate);
            width = 34;
            filled = floor(width * ratio);
            bar = [repmat('#', 1, filled), repmat('-', 1, width - filled)];
            fprintf('\r[%s] %4d/%4d (%5.1f%%) | %s=%s rep=%02d/%02d method=%-17s | elapsed=%s eta=%s', ...
                bar, doneSteps, totalSteps, ratio * 100, xName, char(string(xValue)), rep, reps, ...
                methodName, sec72.Common.formatDuration(elapsed), sec72.Common.formatDuration(eta));
        end

        function text = formatDuration(sec)
            sec = max(0, sec);
            h = floor(sec / 3600);
            m = floor(mod(sec, 3600) / 60);
            s = floor(mod(sec, 60));
            if h > 0
                text = sprintf('%02d:%02d:%02d', h, m, s);
            else
                text = sprintf('%02d:%02d', m, s);
            end
        end

        function plotModels123Metrics(summary, outPng)
            models = [1, 2, 3];
            ycols = {'error_P_mean', 'error_Theta_mean', 'error_B_mean'};
            ylabels = {'Error for P', 'Error for Theta', 'Error for B'};
            sec72.Common.plotMetricGrid(summary, models, ycols, ylabels, outPng, [15.5, 11.0]);
        end

        function plotModels456Metrics(summary, outPng)
            models = [4, 5, 6];
            ycols = {'error_P_mean', 'error_Theta_mean'};
            ylabels = {'Error for P', 'Error for Theta'};
            sec72.Common.plotMetricGrid(summary, models, ycols, ylabels, outPng, [15.5, 7.8]);
        end

        function plotMetricGrid(summary, models, ycols, ylabels, outPng, figSize)
            methods = sec72.Common.methodNames();
            fig = figure('Visible', 'off', 'Color', 'w', 'Units', 'inches', 'Position', [1, 1, figSize]);
            tiledlayout(numel(ycols), numel(models), 'TileSpacing', 'compact', 'Padding', 'compact');
            legendHandles = gobjects(1, numel(methods));

            for i = 1:numel(ycols)
                for j = 1:numel(models)
                    ax = nexttile;
                    hold(ax, 'on');
                    for mi = 1:numel(methods)
                        methodName = methods{mi};
                        d = summary(summary.model == models(j) & strcmp(summary.method, methodName), :);
                        if isempty(d)
                            continue;
                        end
                        d = sortrows(d, 'n');
                        h = plot(ax, d.n, d.(ycols{i}), '-o', 'LineWidth', 2.0, ...
                            'Color', sec72.Common.methodColor(methodName), 'DisplayName', methodName);
                        if i == 1 && j == 1
                            legendHandles(mi) = h;
                        end
                    end
                    if i == 1
                        title(ax, sprintf('Model %d', models(j)));
                    end
                    if j == 1
                        ylabel(ax, ylabels{i});
                    end
                    if i == numel(ycols)
                        xlabel(ax, 'n');
                    end
                    grid(ax, 'on');
                    ax.GridAlpha = 0.3;
                    hold(ax, 'off');
                end
            end
            valid = isgraphics(legendHandles);
            if any(valid)
                legend(legendHandles(valid), methods(valid), 'Orientation', 'horizontal', ...
                    'Location', 'northoutside', 'Box', 'off');
            end
            print(fig, char(outPng), '-dpng', '-r180');
            close(fig);
        end

        function plotRuntime(summary, models, outPng)
            methods = sec72.Common.methodNames();
            fig = figure('Visible', 'off', 'Color', 'w', 'Units', 'inches', 'Position', [1, 1, 14.5, 4.2]);
            tiledlayout(1, numel(models), 'TileSpacing', 'compact', 'Padding', 'compact');
            legendHandles = gobjects(1, numel(methods));
            for j = 1:numel(models)
                ax = nexttile;
                hold(ax, 'on');
                for mi = 1:numel(methods)
                    methodName = methods{mi};
                    d = summary(summary.model == models(j) & strcmp(summary.method, methodName), :);
                    if isempty(d)
                        continue;
                    end
                    d = sortrows(d, 'n');
                    h = plot(ax, d.n, d.time_mean, '-o', 'LineWidth', 2.0, ...
                        'Color', sec72.Common.methodColor(methodName), 'DisplayName', methodName);
                    if j == 1
                        legendHandles(mi) = h;
                    end
                end
                title(ax, sprintf('Model %d', models(j)));
                xlabel(ax, 'n');
                if j == 1
                    ylabel(ax, 'Runtime (sec)');
                end
                grid(ax, 'on');
                ax.GridAlpha = 0.3;
                hold(ax, 'off');
            end
            valid = isgraphics(legendHandles);
            if any(valid)
                legend(legendHandles(valid), methods(valid), 'Orientation', 'horizontal', ...
                    'Location', 'northoutside', 'Box', 'off');
            end
            print(fig, char(outPng), '-dpng', '-r180');
            close(fig);
        end
    end
end
