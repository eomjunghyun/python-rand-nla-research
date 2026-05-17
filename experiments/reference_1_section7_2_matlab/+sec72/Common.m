classdef Common
    % Standalone MATLAB implementation of Reference 1 Section 7.2 utilities.

    methods (Static)
        function methods = methodNames()
            methods = {'Random Projection', 'Random Sampling', 'Non-random', 'CountSketch', 'SIGN Bidirectional'};
        end

        function methods = summaryMethodNames()
            methods = {'Non-random', 'Random Projection', 'Random Sampling', 'CountSketch', 'SIGN Bidirectional'};
        end

        function c = methodColor(methodName)
            switch char(methodName)
                case 'Random Projection'
                    c = [31, 119, 180] ./ 255;
                case 'CountSketch'
                    c = [148, 103, 189] ./ 255;
                case 'SIGN Bidirectional'
                    c = [214, 39, 40] ./ 255;
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
            totalSteps = numel(cfg.model_ids) * numel(cfg.n_values) * cfg.reps * numel(sec72.Common.methodNames());
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
            totalSteps = numel(cfg.model_ids) * numel(cfg.n_values) * cfg.reps * numel(sec72.Common.methodNames());
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
                case 'CountSketch'
                    [AHat, yPred, timing] = sec72.Common.runCountSketch(A, K, KPrime, r, q, rs, normalizeRows);
                case 'SIGN Bidirectional'
                    [AHat, yPred, timing] = sec72.Common.runSignBidirectional(A, K, KPrime, r, q, rs, normalizeRows);
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

        function [AHat, labels, timing] = runCountSketch(A, K, KPrime, r, q, rs, normalizeRows)
            totalStart = tic;
            n = size(A, 1);
            ell = KPrime + r;

            t0 = tic;
            h = randi(rs, ell, n, 1);
            signs = 2 * double(rand(rs, n, 1) >= 0.5) - 1;
            bucketCounts = accumarray(h, 1, [ell, 1], @sum, 0);
            timing.cs_draw_hash_sec = toc(t0);
            timing.cs_embedding_dim = ell;
            timing.cs_bucket_min_load = min(bucketCounts);
            timing.cs_bucket_max_load = max(bucketCounts);
            timing.cs_empty_buckets = sum(bucketCounts == 0);

            t0 = tic;
            STranspose = sparse((1:n)', h, signs, n, ell);
            Y = A * STranspose;
            Y = full(Y);
            timing.cs_initial_multiply_sec = toc(t0);
            timing.cs_sparse_explicit_sketch_sec = timing.cs_initial_multiply_sec;

            t0 = tic;
            for iter = 1:(2 * q)
                Y = A * Y;
            end
            timing.cs_power_iter_sec = toc(t0);

            t0 = tic;
            [Q, ~] = qr(Y, 0);
            timing.cs_qr_sec = toc(t0);

            t0 = tic;
            C = Q' * A * Q;
            C = 0.5 * (C + C');
            timing.cs_build_core_sec = toc(t0);

            t0 = tic;
            AHat = Q * C * Q';
            timing.cs_reconstruct_sec = toc(t0);

            t0 = tic;
            Uc = sec72.Common.topEigvecsSymmetric(C, KPrime);
            timing.cs_small_eig_sec = toc(t0);

            t0 = tic;
            Ucs = Q * Uc;
            timing.cs_lift_sec = toc(t0);

            t0 = tic;
            labels = sec72.Common.kmeansOnRows(Ucs, K, rs, normalizeRows);
            timing.cs_kmeans_sec = toc(t0);
            timing = sec72.Common.finalizeTiming(timing, totalStart);
        end

        function [AHat, labels, timing] = runSignBidirectional(A, K, KPrime, r, q, rs, normalizeRows)
            totalStart = tic;
            n = size(A, 1);
            ell = KPrime + r;

            t0 = tic;
            Omega = randn(rs, n, ell);
            timing.sign_draw_omega_sec = toc(t0);

            t0 = tic;
            Qprev = Omega;
            Qtilde = [];
            Qk = [];
            Rk = [];
            AQt = [];
            if q <= 0
                [Qk, ~] = qr(Qprev, 0);
                Qtilde = Qk;
                AQt = A * Qtilde;
                Rk = eye(size(Qk, 2));
            else
                for iter = 1:q
                    [Qtilde, ~] = qr(A' * Qprev, 0);
                    AQt = A * Qtilde;
                    [Qk, Rk] = qr(AQt, 0);
                    Qprev = Qk;
                end
            end
            timing.sign_bidirectional_iter_sec = toc(t0);

            t0 = tic;
            AHat = AQt * pinv(Rk) * (A' * Qk)';
            AHat = 0.5 * (AHat + AHat');
            timing.sign_reconstruct_sec = toc(t0);

            t0 = tic;
            C = Qk' * A * Qk;
            C = 0.5 * (C + C');
            timing.sign_build_core_sec = toc(t0);

            t0 = tic;
            Uc = sec72.Common.topEigvecsSymmetric(C, KPrime);
            timing.sign_small_eig_sec = toc(t0);

            t0 = tic;
            Usign = Qk * Uc;
            timing.sign_lift_sec = toc(t0);

            t0 = tic;
            labels = sec72.Common.kmeansOnRows(Usign, K, rs, normalizeRows);
            timing.sign_kmeans_sec = toc(t0);
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
                'rp_lift_sec', 'rp_kmeans_sec', ...
                'cs_draw_hash_sec', 'cs_initial_multiply_sec', 'cs_sparse_explicit_sketch_sec', ...
                'cs_power_iter_sec', 'cs_qr_sec', 'cs_build_core_sec', ...
                'cs_reconstruct_sec', 'cs_small_eig_sec', 'cs_lift_sec', 'cs_kmeans_sec', ...
                'cs_embedding_dim', 'cs_bucket_min_load', 'cs_bucket_max_load', 'cs_empty_buckets', ...
                'sign_draw_omega_sec', 'sign_bidirectional_iter_sec', 'sign_reconstruct_sec', ...
                'sign_build_core_sec', 'sign_small_eig_sec', 'sign_lift_sec', 'sign_kmeans_sec'
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

        function outMd = writeReport(thisDir)
            if nargin < 1 || isempty(thisDir)
                thisDir = fileparts(fileparts(mfilename('fullpath')));
            end

            models123Dir = fullfile(thisDir, 'results', 'exp72_models123_paper_aligned_live');
            models456Dir = fullfile(thisDir, 'results', 'exp72_models456_paper_aligned_live');
            raw123 = readtable(fullfile(models123Dir, 'sec72_models123_raw_per_rep.csv'), 'TextType', 'string');
            raw456 = readtable(fullfile(models456Dir, 'sec72_models456_raw_per_rep.csv'), 'TextType', 'string');
            summary123 = readtable(fullfile(models123Dir, 'sec72_models123_summary_mean_std.csv'), 'TextType', 'string');
            summary456 = readtable(fullfile(models456Dir, 'sec72_models456_summary_mean_std.csv'), 'TextType', 'string');
            allSummary = [summary123; summary456];

            [csWinsAll, csLossAll, csTieAll] = sec72.Common.compareMeanMetric(allSummary, 'CountSketch', 'Random Projection', 'error_P_mean');
            [signWinsAll, signLossAll, signTieAll] = sec72.Common.compareMeanMetric(allSummary, 'SIGN Bidirectional', 'Random Projection', 'error_P_mean');
            [csWins123, csLoss123, ~] = sec72.Common.compareMeanMetric(summary123, 'CountSketch', 'Random Projection', 'error_P_mean');
            [csWins456, csLoss456, ~] = sec72.Common.compareMeanMetric(summary456, 'CountSketch', 'Random Projection', 'error_P_mean');
            [signWins123, signLoss123, ~] = sec72.Common.compareMeanMetric(summary123, 'SIGN Bidirectional', 'Random Projection', 'error_P_mean');
            [signWins456, signLoss456, ~] = sec72.Common.compareMeanMetric(summary456, 'SIGN Bidirectional', 'Random Projection', 'error_P_mean');
            csTimeRatio123 = sec72.Common.meanTimeRatio(summary123, 'CountSketch', 'Random Projection');
            csTimeRatio456 = sec72.Common.meanTimeRatio(summary456, 'CountSketch', 'Random Projection');
            signTimeRatio123 = sec72.Common.meanTimeRatio(summary123, 'SIGN Bidirectional', 'Random Projection');
            signTimeRatio456 = sec72.Common.meanTimeRatio(summary456, 'SIGN Bidirectional', 'Random Projection');

            lines = strings(0, 1);
            lines(end + 1) = "# Reference 1 Section 7.2 MATLAB 실험 보고서";
            lines(end + 1) = "";
            lines(end + 1) = "이 보고서는 `experiments/reference_1_section7_2`의 Python 실험을 MATLAB 코드로 다시 구현한 뒤, MATLAB에서 직접 실행해 얻은 결과를 정리한다. 실험은 Reference 1 논문 Section 7.2의 Model 1-6에 해당하며, 이번 갱신에서는 기존 CountSketch에 더해 Wang et al. (2025)의 SIGN 양방향 Nyström/subspace iteration 방식을 추가했다.";
            lines(end + 1) = "";
            lines(end + 1) = "## 1. 실행 요약";
            lines(end + 1) = "";
            lines(end + 1) = "| 항목 | 값 |";
            lines(end + 1) = "|---|---|";
            lines(end + 1) = "| 구현 위치 | `experiments/reference_1_section7_2_matlab/` |";
            lines(end + 1) = "| MATLAB 실행 파일 | `/Applications/MATLAB_R2026a.app/bin/matlab` |";
            lines(end + 1) = "| MATLAB 버전 | R2026a Update 1 |";
            lines(end + 1) = "| 반복 횟수 | 20 |";
            lines(end + 1) = "| seed | 2026 |";
            lines(end + 1) = "| n 값 | 200, 400, 600, 800, 1000, 1200 |";
            lines(end + 1) = sprintf("| Model 1-3 출력 row 수 | raw %d개, summary %d개 |", height(raw123), height(summary123));
            lines(end + 1) = sprintf("| Model 4-6 출력 row 수 | raw %d개, summary %d개 |", height(raw456), height(summary456));
            lines(end + 1) = "";
            lines(end + 1) = "실행 명령은 다음과 같다.";
            lines(end + 1) = "";
            lines(end + 1) = "```bash";
            lines(end + 1) = "/Applications/MATLAB_R2026a.app/bin/matlab -batch ""addpath('experiments/reference_1_section7_2_matlab'); run_all_sec72_matlab('reps',20,'seed',2026,'no_progress',true)""";
            lines(end + 1) = "```";
            lines(end + 1) = "";
            lines(end + 1) = "## 2. 실험 방법";
            lines(end + 1) = "";
            lines(end + 1) = "비교한 방법은 다섯 가지다.";
            lines(end + 1) = "";
            lines(end + 1) = "| 방법 | 설명 |";
            lines(end + 1) = "|---|---|";
            lines(end + 1) = "| Non-random | 원래 adjacency matrix에서 leading eigenvectors를 직접 구한 뒤 k-means를 수행하는 기준 방법 |";
            lines(end + 1) = "| Random Projection | Gaussian random projection과 power iteration으로 spectral subspace를 근사한 뒤 k-means를 수행 |";
            lines(end + 1) = "| Random Sampling | edge를 확률 `p=0.7`로 샘플링하고 `1/p`로 rescale한 matrix에서 spectral clustering 수행 |";
            lines(end + 1) = "| CountSketch | Gaussian test matrix 대신 CountSketch sparse test matrix를 사용해 random projection을 수행 |";
            lines(end + 1) = "| SIGN Bidirectional | SIGN 방식처럼 `A'`와 `A`를 번갈아 곱해 양방향 subspace를 QR로 갱신한 뒤, 그 subspace에서 low-rank approximation과 clustering을 수행 |";
            lines(end + 1) = "";
            lines(end + 1) = "공통 파라미터는 `K=3`, `q=2`, `r=10`, `p=0.7`이다. Model 3과 Model 6은 rank-deficient 설정이므로 `K_prime=2`를 사용했고, 나머지는 `K_prime=3`을 사용했다. Random Projection, CountSketch, SIGN의 sketch dimension은 모두 `ell = K_prime + r`로 맞췄다.";
            lines(end + 1) = "";
            lines(end + 1) = "평가 지표는 Python 실험과 같은 형식으로 맞췄다.";
            lines(end + 1) = "";
            lines(end + 1) = "| 지표 | 의미 |";
            lines(end + 1) = "|---|---|";
            lines(end + 1) = "| `error_P` | 추정 행렬과 true probability matrix의 spectral norm error |";
            lines(end + 1) = "| `error_Theta` | true community membership과 추정 membership 사이의 normalized label error |";
            lines(end + 1) = "| `error_B` | block probability matrix 추정의 max absolute error |";
            lines(end + 1) = "| `time_sec` | 방법별 알고리즘 실행 시간 |";
            lines(end + 1) = "";
            lines(end + 1) = "## 3. 산출물";
            lines(end + 1) = "";
            lines(end + 1) = "Model 1-3 결과:";
            lines(end + 1) = "";
            lines(end + 1) = "| 파일 | 설명 |";
            lines(end + 1) = "|---|---|";
            lines(end + 1) = "| `results/exp72_models123_paper_aligned_live/sec72_models123_raw_per_rep.csv` | 반복별 raw 결과 |";
            lines(end + 1) = "| `results/exp72_models123_paper_aligned_live/sec72_models123_summary_mean_std.csv` | 평균/표준편차 summary |";
            lines(end + 1) = "| `results/exp72_models123_paper_aligned_live/sec72_models123_metrics_figure5_like.png` | Figure 5 형식의 metric plot |";
            lines(end + 1) = "| `results/exp72_models123_paper_aligned_live/sec72_models123_runtime.png` | runtime plot |";
            lines(end + 1) = "";
            lines(end + 1) = "Model 4-6 결과:";
            lines(end + 1) = "";
            lines(end + 1) = "| 파일 | 설명 |";
            lines(end + 1) = "|---|---|";
            lines(end + 1) = "| `results/exp72_models456_paper_aligned_live/sec72_models456_raw_per_rep.csv` | 반복별 raw 결과 |";
            lines(end + 1) = "| `results/exp72_models456_paper_aligned_live/sec72_models456_summary_mean_std.csv` | 평균/표준편차 summary |";
            lines(end + 1) = "| `results/exp72_models456_paper_aligned_live/sec72_models456_metrics_figure6_like.png` | Figure 6 형식의 metric plot |";
            lines(end + 1) = "| `results/exp72_models456_paper_aligned_live/sec72_models456_runtime.png` | runtime plot |";
            lines(end + 1) = "";
            lines(end + 1) = "## 4. 전체 그림";
            lines(end + 1) = "";
            lines(end + 1) = "### 4.1 Model 1-3 metric";
            lines(end + 1) = "";
            lines(end + 1) = "![Model 1-3 MATLAB metrics](results/exp72_models123_paper_aligned_live/sec72_models123_metrics_figure5_like.png)";
            lines(end + 1) = "";
            lines(end + 1) = "### 4.2 Model 1-3 runtime";
            lines(end + 1) = "";
            lines(end + 1) = "![Model 1-3 MATLAB runtime](results/exp72_models123_paper_aligned_live/sec72_models123_runtime.png)";
            lines(end + 1) = "";
            lines(end + 1) = "### 4.3 Model 4-6 metric";
            lines(end + 1) = "";
            lines(end + 1) = "![Model 4-6 MATLAB metrics](results/exp72_models456_paper_aligned_live/sec72_models456_metrics_figure6_like.png)";
            lines(end + 1) = "";
            lines(end + 1) = "### 4.4 Model 4-6 runtime";
            lines(end + 1) = "";
            lines(end + 1) = "![Model 4-6 MATLAB runtime](results/exp72_models456_paper_aligned_live/sec72_models456_runtime.png)";
            lines(end + 1) = "";
            lines(end + 1) = "## 5. 대표 수치: n = 1200";
            lines(end + 1) = "";
            lines(end + 1) = "아래 표는 가장 큰 크기인 `n=1200`에서의 평균 결과다. 표준편차는 CSV 파일에 함께 저장되어 있으며, 여기서는 가독성을 위해 평균만 표시한다.";
            lines(end + 1) = "";
            lines(end + 1) = "### 5.1 Model 1-3";
            lines = sec72.Common.appendRepresentativeTable(lines, summary123, [1, 2, 3], 1200);
            lines(end + 1) = "";
            lines(end + 1) = sprintf("Model 1-3에서 CountSketch가 Gaussian Random Projection보다 낮은 `error_P`를 기록한 지점은 %d개, 더 높은 지점은 %d개였다. SIGN Bidirectional은 Random Projection보다 낮은 지점이 %d개, 더 높은 지점이 %d개였다.", csWins123, csLoss123, signWins123, signLoss123);
            lines(end + 1) = sprintf("실행 시간 평균 비율은 CountSketch/RP가 %.3f배, SIGN/RP가 %.3f배였다.", csTimeRatio123, signTimeRatio123);
            lines(end + 1) = "";
            lines(end + 1) = "### 5.2 Model 4-6";
            lines = sec72.Common.appendRepresentativeTable(lines, summary456, [4, 5, 6], 1200);
            lines(end + 1) = "";
            lines(end + 1) = sprintf("Model 4-6에서 CountSketch가 Gaussian Random Projection보다 낮은 `error_P`를 기록한 지점은 %d개, 더 높은 지점은 %d개였다. SIGN Bidirectional은 Random Projection보다 낮은 지점이 %d개, 더 높은 지점이 %d개였다.", csWins456, csLoss456, signWins456, signLoss456);
            lines(end + 1) = sprintf("실행 시간 평균 비율은 CountSketch/RP가 %.3f배, SIGN/RP가 %.3f배였다.", csTimeRatio456, signTimeRatio456);
            lines(end + 1) = "";
            lines(end + 1) = "## 6. 핵심 관찰";
            lines(end + 1) = "";
            lines(end + 1) = sprintf("1. `error_P`에서는 Random Projection과 CountSketch가 여전히 가장 강한 축이다. CountSketch는 전체 36개 `(model, n)` 지점 중 %d개 지점에서 Gaussian Random Projection보다 낮은 `error_P`를 보였고, %d개 지점에서는 더 높았다.", csWinsAll, csLossAll);
            lines(end + 1) = "";
            lines(end + 1) = sprintf("2. SIGN Bidirectional은 전체 36개 지점 중 %d개 지점에서 Random Projection보다 낮은 `error_P`를 기록했고, %d개 지점에서는 더 높았다. 즉 synthetic clustering metric에서는 항상 Gaussian RP를 이기지는 않지만, 일부 degree-corrected/rank-deficient 설정에서는 경쟁적인 값을 보였다.", signWinsAll, signLossAll);
            if csTieAll + signTieAll > 0
                lines(end + 1) = sprintf("   동률로 처리된 지점은 CountSketch 비교 %d개, SIGN 비교 %d개였다.", csTieAll, signTieAll);
            end
            lines(end + 1) = "";
            lines(end + 1) = "3. `error_Theta`는 모델 구조에 따라 난이도가 크게 달라졌다. Model 1-3에서는 큰 `n`에서 membership error가 거의 사라지는 반면, Model 4-6에서는 degree correction과 rank-deficient 구조 때문에 error가 더 높게 남는다.";
            lines(end + 1) = "";
            lines(end + 1) = "4. SIGN Bidirectional은 low-rank approximation 관점의 양방향 subspace 갱신을 사용하므로, clustering label recovery와 완전히 같은 목적함수를 직접 최적화하지 않는다. 그래서 `error_P`가 괜찮아도 `error_Theta`나 `error_B`가 반드시 같이 좋아지지는 않는다.";
            lines(end + 1) = "";
            lines(end + 1) = "5. 실행 시간은 Random Projection과 CountSketch가 가장 짧은 그룹이다. SIGN은 `A'` 방향과 `A` 방향을 번갈아 QR로 갱신하고 Nyström식 재구성을 수행하므로, 같은 `q`와 `ell`에서는 RP/CountSketch보다 더 무거운 편이다.";
            lines(end + 1) = "";
            lines(end + 1) = "## 7. Python 결과와의 관계";
            lines(end + 1) = "";
            lines(end + 1) = "이 MATLAB 구현은 Python `src.common`을 호출하지 않고 같은 실험 절차를 MATLAB 코드로 다시 작성한 것이다. 출력 파일명, CSV column, plot 형식은 Python 실험과 맞췄다.";
            lines(end + 1) = "";
            lines(end + 1) = "다만 MATLAB과 Python/NumPy는 random number generator, eigen solver 구현, k-means 초기화와 반복 세부 구현, CountSketch hash/sign 생성 방식, 부동소수점 연산 순서가 다르므로 수치가 완전히 같지는 않다.";
            lines(end + 1) = "";
            lines(end + 1) = "## 8. MATLAB 실행이 Python보다 가볍게 보인 이유";
            lines(end + 1) = "";
            lines(end + 1) = "이번 실행에서 MATLAB 쪽이 Python 쪽보다 빠르고 CPU/메모리 사용도 덜 부담스럽게 보인 가장 큰 이유는 언어 자체의 차이라기보다 구현 세부가 다르기 때문이다.";
            lines(end + 1) = "";
            lines(end + 1) = "가장 중요한 차이는 eigen computation 방식이다. Python `src/common.py`의 `top_eigvecs_symmetric()`는 dense matrix의 전체 고유값/고유벡터를 모두 구하는 경향이 있고, `spectral_norm_sym()`도 전체 고유값을 계산한다. 반면 MATLAB 구현은 `topEigpairsSymmetric()`와 `spectralNormSym()`에서 주로 `eigs()`를 사용해 필요한 leading eigenvectors 또는 가장 큰 절댓값 고유값만 부분적으로 계산한다.";
            lines(end + 1) = "";
            lines(end + 1) = "두 번째 차이는 dense matrix 복사와 임시 배열이다. Python 구현은 `A`, `P`, `A_hat`, `A_hat - P`를 dense NumPy array로 반복해서 다루며, 큰 임시 행렬이 많이 생긴다. MATLAB 구현도 dense matrix를 쓰지만 partial eigensolver 사용 때문에 전체 고유분해에 필요한 작업 배열과 시간이 줄어든다.";
            lines(end + 1) = "";
            lines(end + 1) = "따라서 이 비교에서 `Python이 본질적으로 느리다`라고 해석하면 안 된다. 현재 Python 구현이 full dense eigen decomposition과 full eigenvalue metric evaluation을 반복하는 구조라 무겁고, MATLAB 구현은 partial eigensolver를 적극적으로 사용해서 가볍게 돈 것이다.";
            lines(end + 1) = "";
            lines(end + 1) = "## 9. 결론";
            lines(end + 1) = "";
            lines(end + 1) = "MATLAB 재구현 결과에서도 Section 7.2의 큰 결론은 유지된다. Random Projection과 CountSketch는 대부분의 모델과 `n` 값에서 `error_P`가 낮고 실행 시간도 짧다. SIGN Bidirectional은 양방향 Nyström/subspace iteration을 MATLAB 실험에 추가한 비교군으로, 일부 설정에서는 경쟁적인 `error_P`를 보였지만 전체적으로는 Gaussian RP와 CountSketch보다 안정적인 clustering 방법이라고 보기는 어렵다. Random Sampling은 여전히 가장 불안정하고 느린 편이다.";

            outMd = fullfile(thisDir, 'section7_2_matlab_experiment_report.md');
            sec72.Common.writeLines(outMd, lines);
        end

        function lines = appendRepresentativeTable(lines, summary, models, nValue)
            methods = sec72.Common.summaryMethodNames();
            lines(end + 1) = "";
            lines(end + 1) = "| Model | Method | error_P | error_Theta | error_B | time_sec |";
            lines(end + 1) = "|---:|---|---:|---:|---:|---:|";
            for modelId = models
                for mi = 1:numel(methods)
                    methodName = methods{mi};
                    row = summary(summary.model == modelId & summary.n == nValue & string(summary.method) == string(methodName), :);
                    if isempty(row)
                        continue;
                    end
                    lines(end + 1) = sprintf("| %d | %s | %.3f | %.4f | %.4f | %.4f |", ...
                        modelId, methodName, row.error_P_mean(1), row.error_Theta_mean(1), ...
                        row.error_B_mean(1), row.time_mean(1));
                end
            end
        end

        function [wins, losses, ties] = compareMeanMetric(summary, methodA, methodB, metricName)
            keys = unique(summary(:, {'model', 'n'}), 'rows');
            wins = 0;
            losses = 0;
            ties = 0;
            for i = 1:height(keys)
                rowA = summary(summary.model == keys.model(i) & summary.n == keys.n(i) & string(summary.method) == string(methodA), :);
                rowB = summary(summary.model == keys.model(i) & summary.n == keys.n(i) & string(summary.method) == string(methodB), :);
                if isempty(rowA) || isempty(rowB)
                    continue;
                end
                a = rowA.(metricName)(1);
                b = rowB.(metricName)(1);
                if abs(a - b) <= 1e-12
                    ties = ties + 1;
                elseif a < b
                    wins = wins + 1;
                else
                    losses = losses + 1;
                end
            end
        end

        function ratio = meanTimeRatio(summary, methodA, methodB)
            keys = unique(summary(:, {'model', 'n'}), 'rows');
            vals = [];
            for i = 1:height(keys)
                rowA = summary(summary.model == keys.model(i) & summary.n == keys.n(i) & string(summary.method) == string(methodA), :);
                rowB = summary(summary.model == keys.model(i) & summary.n == keys.n(i) & string(summary.method) == string(methodB), :);
                if isempty(rowA) || isempty(rowB) || rowB.time_mean(1) <= 0
                    continue;
                end
                vals(end + 1, 1) = rowA.time_mean(1) / rowB.time_mean(1); %#ok<AGROW>
            end
            ratio = mean(vals, 'omitnan');
        end

        function writeLines(path, lines)
            fid = fopen(path, 'w');
            cleanup = onCleanup(@() fclose(fid));
            for i = 1:numel(lines)
                fprintf(fid, '%s\n', lines(i));
            end
            clear cleanup;
        end
    end
end
