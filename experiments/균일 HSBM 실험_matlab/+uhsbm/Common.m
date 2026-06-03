classdef Common
    % Standalone MATLAB utilities for uniform HSBM hypergraph experiments.

    methods (Static)
        function labels = methodLabels()
            labels = {'Non-random eigs', 'Gaussian RP', 'Random sampling', 'CountSketch RP'};
        end

        function keys = methodKeys()
            keys = {'non_random', 'gaussian_random_projection', 'random_sampling', 'countsketch_random_projection'};
        end

        function c = methodColor(methodName)
            switch char(methodName)
                case 'Non-random eigs'
                    c = [44, 160, 44] ./ 255;
                case 'Gaussian RP'
                    c = [31, 119, 180] ./ 255;
                case 'Random sampling'
                    c = [255, 127, 14] ./ 255;
                case 'CountSketch RP'
                    c = [148, 103, 189] ./ 255;
                otherwise
                    c = [0, 0, 0];
            end
        end

        function outputs = runUniformAll(thisDir, varargin)
            opts = uhsbm.Common.parseOptions(varargin{:});
            specs = uhsbm.Common.uniformSpecs(opts);
            resultsRoot = fullfile(thisDir, 'results');
            if ~exist(resultsRoot, 'dir')
                mkdir(resultsRoot);
            end

            outputs = struct();
            for i = 1:numel(specs)
                outputs.(specs(i).sweep) = uhsbm.Common.runSpec(specs(i), resultsRoot, opts);
            end
            outputs.report_md = uhsbm.Common.writeUniformReport(thisDir, specs, opts);
        end

        function outputs = runRedesignedAll(thisDir, varargin)
            opts = uhsbm.Common.parseOptions(varargin{:});
            specs = uhsbm.Common.redesignedSpecs(opts);
            resultsRoot = fullfile(thisDir, 'results');
            if ~exist(resultsRoot, 'dir')
                mkdir(resultsRoot);
            end

            outputs = struct();
            for i = 1:numel(specs)
                outputs.(specs(i).name) = uhsbm.Common.runSpec(specs(i), resultsRoot, opts);
            end
            outputs.diagnostics = uhsbm.Common.runRedesignedDiagnostics(specs, resultsRoot, opts);
            outputs.report_md = uhsbm.Common.writeRedesignedReport(thisDir, specs, opts);
        end

        function opts = parseOptions(varargin)
            p = inputParser;
            addParameter(p, 'reps', [], @(x) isempty(x) || (isscalar(x) && isnumeric(x)));
            addParameter(p, 'seed', [], @(x) isempty(x) || (isscalar(x) && isnumeric(x)));
            addParameter(p, 'smoke', false, @(x) islogical(x) || isnumeric(x));
            addParameter(p, 'no_progress', false, @(x) islogical(x) || isnumeric(x));
            addParameter(p, 'overwrite', true, @(x) islogical(x) || isnumeric(x));
            parse(p, varargin{:});
            opts = p.Results;
            opts.smoke = logical(opts.smoke);
            opts.no_progress = logical(opts.no_progress);
            opts.overwrite = logical(opts.overwrite);
        end

        function specs = uniformSpecs(opts)
            reps = 10;
            seed = 20260506;
            if ~isempty(opts.reps), reps = double(opts.reps); end
            if ~isempty(opts.seed), seed = double(opts.seed); end

            specs = repmat(uhsbm.Common.baseSpec(), 1, 3);
            specs(1) = uhsbm.Common.baseSpec();
            specs(1).kind = 'uniform';
            specs(1).name = 'EXP-20260506-008_uniform_hsbm_K_rho16_eigsh_methods_matlab';
            specs(1).sweep = 'K';
            specs(1).title = 'Uniform HSBM K sweep - MATLAB method comparison';
            specs(1).x_col = 'K';
            specs(1).x_values = [2, 4, 6, 8, 10, 12];
            specs(1).n = 5000;
            specs(1).K = NaN;
            specs(1).rho_n = 16;
            specs(1).reps = reps;
            specs(1).seed = seed;
            specs(1).rp_oversampling = 160;
            specs(1).rp_power_iter = 4;
            specs(1).random_sampling_p = 0.7;

            specs(2) = specs(1);
            specs(2).name = 'EXP-20260506-007_uniform_hsbm_n_rho16_eigsh_methods_matlab';
            specs(2).sweep = 'n';
            specs(2).title = 'Uniform HSBM n scaling - MATLAB method comparison';
            specs(2).x_col = 'n';
            specs(2).x_values = 2000:2000:10000;
            specs(2).n = NaN;
            specs(2).K = 3;
            specs(2).rho_n = 16;

            specs(3) = specs(1);
            specs(3).name = 'EXP-20260506-009_uniform_hsbm_rho_eigsh_methods_matlab';
            specs(3).sweep = 'rho_n';
            specs(3).title = 'Uniform HSBM rho_n sweep - MATLAB method comparison';
            specs(3).x_col = 'rho_n';
            specs(3).x_values = [2, 4, 8, 16, 32, 64];
            specs(3).n = 5000;
            specs(3).K = 3;
            specs(3).rho_n = NaN;

            if opts.smoke
                for i = 1:numel(specs)
                    specs(i).x_values = specs(i).x_values(1);
                    specs(i).reps = 1;
                    specs(i).n = min(600, specs(i).n);
                    if isnan(specs(i).n) && strcmp(specs(i).x_col, 'n')
                        specs(i).x_values = 600;
                    end
                    specs(i).rp_oversampling = 12;
                    specs(i).rp_power_iter = 1;
                    specs(i).kmeans_n_init = 5;
                end
            end
        end

        function specs = redesignedSpecs(opts)
            reps = 5;
            seed = 20260507;
            if ~isempty(opts.reps), reps = double(opts.reps); end
            if ~isempty(opts.seed), seed = double(opts.seed); end

            names = { ...
                'density_background_fixed_gap', ...
                'K_compensated_reference_signal', ...
                'n_scaling_reference_signal', ...
                'rho_density_signal_control', ...
                'K_compensated_rank_scaling', ...
                'n_scaling_fixed_density_signal'};
            titles = { ...
                'Density sweep with stronger fixed signal gap', ...
                'K sweep with K^2 compensation and reference signal', ...
                'n scaling at reference K=6 signal regime', ...
                'Weak-gap diagnostic: rho_n sweep with density-signal separation', ...
                'Weak-gap diagnostic: K sweep with rho_n compensation', ...
                'Weak-gap diagnostic: n scaling at fixed density-signal regime'};
            xcols = {'density_level', 'K', 'n', 'rho_n', 'K', 'n'};
            xvals = {[1, 2, 3, 4, 5], [3, 4, 6, 8, 10], [3000, 6000, 9000, 12000, 15000], ...
                [4, 8, 16, 32, 64, 128], [3, 4, 6, 8, 10], [3000, 6000, 9000, 12000, 15000]};

            specs = repmat(uhsbm.Common.baseSpec(), 1, numel(names));
            for i = 1:numel(names)
                specs(i) = uhsbm.Common.baseSpec();
                specs(i).kind = 'redesigned';
                specs(i).name = names{i};
                specs(i).sweep = names{i};
                specs(i).title = titles{i};
                specs(i).x_col = xcols{i};
                specs(i).x_values = xvals{i};
                specs(i).reps = reps;
                specs(i).seed = seed * 100 + i;
                specs(i).rp_oversampling = 30;
                specs(i).rp_power_iter = 1;
                specs(i).random_sampling_p = 0.3;
                specs(i).clip_probs = false;
            end

            specs(1).n = 6000; specs(1).K = 6;
            specs(2).n = 6000;
            specs(3).K = 6; specs(3).rho_n = 16;
            specs(4).n = 6000; specs(4).K = 6;
            specs(5).n = 6000;
            specs(6).K = 6; specs(6).rho_n = 32;

            if opts.smoke
                for i = 1:numel(specs)
                    specs(i).x_values = specs(i).x_values(1);
                    specs(i).reps = 1;
                    specs(i).n = min(700, specs(i).n);
                    if isnan(specs(i).n) && strcmp(specs(i).x_col, 'n')
                        specs(i).x_values = 700;
                    end
                    specs(i).rp_oversampling = 8;
                    specs(i).rp_power_iter = 1;
                    specs(i).kmeans_n_init = 5;
                end
            end
        end

        function spec = baseSpec()
            spec = struct();
            spec.kind = '';
            spec.name = '';
            spec.sweep = '';
            spec.title = '';
            spec.x_col = '';
            spec.x_values = [];
            spec.n = NaN;
            spec.K = NaN;
            spec.m = 3;
            spec.a_in = 36;
            spec.b_out = 4;
            spec.rho_n = NaN;
            spec.center = 10;
            spec.base_gap = 4;
            spec.reps = 1;
            spec.seed = 1;
            spec.rp_oversampling = 30;
            spec.rp_power_iter = 1;
            spec.random_sampling_p = 0.3;
            spec.kmeans_n_init = 20;
            spec.eigs_tol = 1e-6;
            spec.clip_probs = true;
        end

        function out = runSpec(spec, resultsRoot, opts)
            outDir = fullfile(resultsRoot, spec.name);
            if ~exist(outDir, 'dir')
                mkdir(outDir);
            end
            rawPath = fullfile(outDir, [spec.name, '_raw.csv']);
            summaryPath = fullfile(outDir, [spec.name, '_summary.csv']);
            configPath = fullfile(outDir, [spec.name, '_config.json']);
            plotPath = fullfile(outDir, [spec.name, '_summary.png']);

            if exist(rawPath, 'file') && ~opts.overwrite
                raw = readtable(rawPath, 'TextType', 'string');
            else
                raw = uhsbm.Common.runRaw(spec, opts);
                writetable(raw, rawPath);
            end
            summary = uhsbm.Common.summarizeRaw(raw);
            writetable(summary, summaryPath);
            uhsbm.Common.writeJsonConfig(spec, configPath);
            uhsbm.Common.plotSummary(summary, spec, plotPath);

            out = struct('raw', rawPath, 'summary', summaryPath, 'config', configPath, 'plot', plotPath);
        end

        function raw = runRaw(spec, opts)
            records = repmat(uhsbm.Common.emptyRecord(), 0, 1);
            total = numel(spec.x_values) * spec.reps;
            step = 0;
            for xv = spec.x_values
                for rep = 1:spec.reps
                    step = step + 1;
                    if ~opts.no_progress
                        fprintf('[%s] %s=%g rep %d/%d (%d/%d)\n', spec.name, spec.x_col, xv, rep, spec.reps, step, total);
                    end
                    rows = uhsbm.Common.runOneInstance(spec, xv, rep);
                    records = [records; rows(:)]; %#ok<AGROW>
                end
            end
            raw = struct2table(records);
        end

        function rows = runOneInstance(spec, xValue, rep)
            [n, K, rhoN, aIn, bOut] = uhsbm.Common.concreteParams(spec, xValue);
            seed = uhsbm.Common.normalizeSeed(double(spec.seed) + uhsbm.Common.valueToSeedComponent(xValue) * 100000 + rep);
            rng(seed, 'twister');

            pIn = aIn * rhoN / (n ^ (spec.m - 1));
            pOut = bOut * rhoN / (n ^ (spec.m - 1));
            if spec.clip_probs
                pIn = min(max(pIn, 0), 1);
                pOut = min(max(pOut, 0), 1);
            elseif pIn < 0 || pIn > 1 || pOut < 0 || pOut > 1
                error('Invalid probability: pIn=%g, pOut=%g', pIn, pOut);
            end

            t0 = tic;
            labels = uhsbm.Common.makeBalancedLabels(n, K);
            [edges, genStats] = uhsbm.Common.sampleUniformHsbm3(labels, K, pIn, pOut);
            generationSec = toc(t0);

            t0 = tic;
            [theta, degreeStats] = uhsbm.Common.buildTheta(n, edges);
            buildSec = toc(t0);
            expected = uhsbm.Common.expectedStats(labels, K, spec.m, pIn, pOut);

            methods = uhsbm.Common.methodKeys();
            methodLabels = uhsbm.Common.methodLabels();
            rows = repmat(uhsbm.Common.emptyRecord(), numel(methods), 1);

            for j = 1:numel(methods)
                methodSeed = uhsbm.Common.normalizeSeed(seed + j * 10000000);
                rng(methodSeed, 'twister');
                tMethod = tic;
                [pred, stats] = uhsbm.Common.spectralCluster(theta, K, spec, methods{j});
                methodSec = toc(tMethod);

                tMetric = tic;
                mis = uhsbm.Common.misclassification(labels, pred, K);
                ari = uhsbm.Common.adjustedRandIndex(labels, pred, K);
                nmi = uhsbm.Common.normalizedMutualInfo(labels, pred, K);
                metricSec = toc(tMetric);

                rec = uhsbm.Common.emptyRecord();
                rec.experiment = string(spec.name);
                rec.sweep = string(spec.sweep);
                rec.x_col = string(spec.x_col);
                rec.x_value = xValue;
                rec.density_level = NaN;
                if strcmp(spec.x_col, 'density_level'), rec.density_level = xValue; end
                rec.rep = rep;
                rec.seed = seed;
                rec.n = n;
                rec.K = K;
                rec.m = spec.m;
                rec.rho_n = rhoN;
                rec.a_in = aIn;
                rec.b_out = bOut;
                rec.signal_gap = aIn - bOut;
                rec.p_in = pIn;
                rec.p_out = pOut;
                rec.num_hyperedges_total = size(edges, 1);
                rec.theta_nnz = nnz(theta);
                rec.theta_density = nnz(theta) / (n * n);
                rec.generation_wall_sec = generationSec;
                rec.hypergraph_laplacian_build_wall_sec = buildSec;
                rec.sampling_mode = string(genStats.sampling_mode);
                rec.num_isolated_nodes = degreeStats.num_isolated_nodes;
                rec.isolated_fraction = degreeStats.isolated_fraction;
                rec.hypergraph_degree_mean = degreeStats.hypergraph_degree_mean;
                rec.hypergraph_degree_max = degreeStats.hypergraph_degree_max;
                rec.expected_hyperedges_total = expected.expected_hyperedges_total;
                rec.expected_hyperedges_per_n = expected.expected_hyperedges_per_n;
                rec.expected_degree_mean = expected.expected_degree_mean;
                rec.candidate_within_fraction = expected.candidate_within_fraction;
                rec.method = string(methodLabels{j});
                rec.method_key = string(methods{j});
                rec.misclassification_rate = mis;
                rec.ARI = ari;
                rec.NMI = nmi;
                rec.metric_wall_sec = metricSec;
                rec.method_wall_sec = methodSec;

                rec = uhsbm.Common.attachStats(rec, stats);
                rec.algorithm_total_wall_sec = rec.generation_wall_sec + rec.hypergraph_laplacian_build_wall_sec + ...
                    rec.eigen_decomposition_wall_sec + rec.embedding_normalize_wall_sec + rec.kmeans_wall_sec;
                rows(j) = rec;
            end
        end

        function [n, K, rhoN, aIn, bOut] = concreteParams(spec, xValue)
            if strcmp(spec.kind, 'uniform')
                n = spec.n;
                K = spec.K;
                rhoN = spec.rho_n;
                if strcmp(spec.x_col, 'n'), n = xValue; end
                if strcmp(spec.x_col, 'K'), K = xValue; end
                if strcmp(spec.x_col, 'rho_n'), rhoN = xValue; end
                aIn = spec.a_in;
                bOut = spec.b_out;
                n = double(n); K = double(K); rhoN = double(rhoN);
                return;
            end

            switch spec.name
                case 'density_background_fixed_gap'
                    schedule = [16, 36, 4; 24, 36, 4; 32, 40, 8; 48, 44, 12; 64, 52, 20];
                    row = schedule(round(xValue), :);
                    n = spec.n; K = spec.K; rhoN = row(1); aIn = row(2); bOut = row(3);
                case 'K_compensated_reference_signal'
                    K = xValue; n = spec.n; rhoN = 16 * (K / 6) ^ 2; aIn = 36; bOut = 4;
                case 'n_scaling_reference_signal'
                    n = xValue; K = spec.K; rhoN = spec.rho_n; aIn = 36; bOut = 4;
                otherwise
                    n = spec.n; K = spec.K;
                    if strcmp(spec.x_col, 'n'), n = xValue; end
                    if strcmp(spec.x_col, 'K'), K = xValue; end
                    if strcmp(spec.x_col, 'rho_n')
                        rhoN = xValue;
                    elseif strcmp(spec.x_col, 'K')
                        rhoN = 16 * (K / 4) ^ 2;
                    else
                        rhoN = spec.rho_n;
                    end
                    if strcmp(spec.x_col, 'rho_n')
                        gap = spec.base_gap * sqrt(16 / rhoN);
                    elseif strcmp(spec.x_col, 'n')
                        gap = spec.base_gap * sqrt(16 / spec.rho_n);
                    else
                        gap = spec.base_gap;
                    end
                    aIn = spec.center + gap / 2;
                    bOut = spec.center - gap / 2;
            end
            n = double(n); K = double(K); rhoN = double(rhoN); aIn = double(aIn); bOut = double(bOut);
        end

        function labels = makeBalancedLabels(n, K)
            base = floor(n / K);
            remn = mod(n, K);
            labels = zeros(n, 1);
            pos = 1;
            for k = 1:K
                nk = base + double(k <= remn);
                labels(pos:pos + nk - 1) = k;
                pos = pos + nk;
            end
            labels = labels(randperm(n));
        end

        function [edges, stats] = sampleUniformHsbm3(labels, K, pIn, pOut)
            n = numel(labels);
            m = 3;
            withinEdges = zeros(0, 3);
            totalWithin = 0;
            for k = 1:K
                nodes = find(labels == k);
                nk = numel(nodes);
                if nk < m, continue; end
                cand = nchoosek(nk, m);
                totalWithin = totalWithin + cand;
                draws = uhsbm.Common.binomialDraw(cand, pIn);
                if draws > 0
                    withinEdges = [withinEdges; uhsbm.Common.sampleTriplesFromNodes(nodes, draws)]; %#ok<AGROW>
                end
            end
            total = nchoosek(n, m);
            mixedTotal = total - totalWithin;
            mixedDraws = uhsbm.Common.binomialDraw(mixedTotal, pOut);
            mixedEdges = uhsbm.Common.sampleMixedTriples(labels, mixedDraws);
            edges = unique([withinEdges; mixedEdges], 'rows');
            stats = struct('sampling_mode', 'sparse', 'num_candidates_total', total, ...
                'num_candidates_within', totalWithin, 'num_candidates_mixed', mixedTotal);
        end

        function edges = sampleTriplesFromNodes(nodes, draws)
            if draws <= 0
                edges = zeros(0, 3);
                return;
            end
            nk = numel(nodes);
            keys = zeros(0, 1);
            vals = zeros(0, 3);
            batch = max(2000, ceil(draws * 1.35));
            attempts = 0;
            while size(vals, 1) < draws && attempts < 120
                local = randi(nk, batch, 3);
                ok = local(:,1) ~= local(:,2) & local(:,1) ~= local(:,3) & local(:,2) ~= local(:,3);
                local = sort(local(ok, :), 2);
                tri = nodes(local);
                newKeys = uhsbm.Common.edgeKeys(tri, max(nodes));
                vals = [vals; tri]; %#ok<AGROW>
                keys = [keys; newKeys]; %#ok<AGROW>
                [~, ia] = unique(keys, 'stable');
                vals = vals(ia, :);
                keys = keys(ia);
                batch = max(2000, ceil((draws - size(vals, 1)) * 1.8));
                attempts = attempts + 1;
            end
            if size(vals, 1) < draws
                error('Could not sample enough within hyperedges.');
            end
            edges = vals(1:draws, :);
        end

        function edges = sampleMixedTriples(labels, draws)
            if draws <= 0
                edges = zeros(0, 3);
                return;
            end
            n = numel(labels);
            keys = zeros(0, 1);
            vals = zeros(0, 3);
            batch = max(4000, ceil(draws * 1.6));
            attempts = 0;
            while size(vals, 1) < draws && attempts < 160
                tri = randi(n, batch, 3);
                ok = tri(:,1) ~= tri(:,2) & tri(:,1) ~= tri(:,3) & tri(:,2) ~= tri(:,3);
                tri = sort(tri(ok, :), 2);
                labs = labels(tri);
                mixed = ~(labs(:,1) == labs(:,2) & labs(:,1) == labs(:,3));
                tri = tri(mixed, :);
                newKeys = uhsbm.Common.edgeKeys(tri, n);
                vals = [vals; tri]; %#ok<AGROW>
                keys = [keys; newKeys]; %#ok<AGROW>
                [~, ia] = unique(keys, 'stable');
                vals = vals(ia, :);
                keys = keys(ia);
                batch = max(4000, ceil((draws - size(vals, 1)) * 2.2));
                attempts = attempts + 1;
            end
            if size(vals, 1) < draws
                error('Could not sample enough mixed hyperedges.');
            end
            edges = vals(1:draws, :);
        end

        function x = binomialDraw(N, p)
            N = double(N);
            p = double(p);
            if N <= 0 || p <= 0
                x = 0;
                return;
            end
            if p >= 1
                x = round(N);
                return;
            end
            mu = N * p;
            sigma = sqrt(N * p * (1 - p));
            if N <= 1000000
                x = sum(rand(round(N), 1) < p);
            elseif mu < 30
                x = uhsbm.Common.poissonDraw(mu);
            else
                x = round(mu + sigma * randn());
                x = min(max(x, 0), round(N));
            end
        end

        function x = poissonDraw(lambda)
            if lambda <= 0
                x = 0;
                return;
            end
            if lambda > 30
                x = max(0, round(lambda + sqrt(lambda) * randn()));
                return;
            end
            L = exp(-lambda);
            k = 0;
            prodVal = 1;
            while prodVal > L
                k = k + 1;
                prodVal = prodVal * rand();
            end
            x = k - 1;
        end

        function keys = edgeKeys(edges, n)
            if isempty(edges)
                keys = zeros(0, 1);
                return;
            end
            e = double(edges);
            keys = (e(:,1) - 1) .* double(n) .* double(n) + (e(:,2) - 1) .* double(n) + e(:,3);
        end

        function [theta, stats] = buildTheta(n, edges)
            if isempty(edges)
                theta = sparse(n, n);
                stats = struct('num_isolated_nodes', n, 'isolated_fraction', 1, ...
                    'hypergraph_degree_mean', 0, 'hypergraph_degree_max', 0);
                return;
            end
            deg = accumarray(edges(:), 1, [n, 1], @sum, 0);
            invSqrt = zeros(n, 1);
            mask = deg > 0;
            invSqrt(mask) = 1 ./ sqrt(double(deg(mask)));
            E = size(edges, 1);
            ii = zeros(E * 9, 1);
            jj = zeros(E * 9, 1);
            vv = zeros(E * 9, 1);
            pos = 1;
            for a = 1:3
                for b = 1:3
                    idx = pos:pos + E - 1;
                    ia = edges(:, a);
                    jb = edges(:, b);
                    ii(idx) = ia;
                    jj(idx) = jb;
                    vv(idx) = (1 / 3) .* invSqrt(ia) .* invSqrt(jb);
                    pos = pos + E;
                end
            end
            theta = sparse(ii, jj, vv, n, n);
            theta = (theta + theta') * 0.5;
            stats = struct();
            stats.num_isolated_nodes = sum(deg == 0);
            stats.isolated_fraction = mean(deg == 0);
            stats.hypergraph_degree_mean = mean(deg);
            stats.hypergraph_degree_max = max(deg);
        end

        function s = expectedStats(labels, K, m, pIn, pOut)
            n = numel(labels);
            total = nchoosek(n, m);
            within = 0;
            for k = 1:K
                nk = sum(labels == k);
                if nk >= m
                    within = within + nchoosek(nk, m);
                end
            end
            mixed = total - within;
            expectedEdges = within * pIn + mixed * pOut;
            s = struct();
            s.expected_hyperedges_total = expectedEdges;
            s.expected_hyperedges_per_n = expectedEdges / n;
            s.expected_degree_mean = m * expectedEdges / n;
            s.candidate_within_fraction = within / total;
        end

        function [labels, stats] = spectralCluster(theta, K, spec, methodKey)
            stats = uhsbm.Common.emptyStats();
            totalTic = tic;
            tEig = tic;
            switch methodKey
                case 'non_random'
                    [vals, U] = uhsbm.Common.topEig(theta, K, spec.eigs_tol);
                    stats.non_random_eigensolver = string('eigs');
                case 'gaussian_random_projection'
                    [vals, U, stats] = uhsbm.Common.gaussianRp(theta, K, spec, stats);
                case 'random_sampling'
                    t0 = tic;
                    [sampled, rsStats] = uhsbm.Common.sampleTheta(theta, spec.random_sampling_p);
                    stats.rs_sample_matrix_wall_sec = toc(t0);
                    stats.rs_original_upper_nnz = rsStats.original_upper_nnz;
                    stats.rs_sampled_upper_nnz = rsStats.sampled_upper_nnz;
                    stats.rs_sampling_probability = spec.random_sampling_p;
                    stats.rs_sampled_theta_nnz = nnz(sampled);
                    [vals, U] = uhsbm.Common.topEig(sampled, K, spec.eigs_tol);
                case 'countsketch_random_projection'
                    [vals, U, stats] = uhsbm.Common.countSketchRp(theta, K, spec, stats);
                otherwise
                    error('Unknown method: %s', methodKey);
            end
            stats.eigen_decomposition_wall_sec = toc(tEig);

            t0 = tic;
            U = uhsbm.Common.normalizeRows(U);
            stats.embedding_normalize_wall_sec = toc(t0);

            t0 = tic;
            labels = uhsbm.Common.simpleKmeans(U, K, spec.kmeans_n_init, 300);
            stats.kmeans_wall_sec = toc(t0);
            stats.spectral_clustering_wall_sec = toc(totalTic);
            stats.top_eigenvalue_max = max(vals);
            stats.top_eigenvalue_min = min(vals);
        end

        function [vals, vecs] = topEig(A, K, tol)
            n = size(A, 1);
            opts = struct();
            opts.tol = tol;
            opts.maxit = 1000;
            try
                if n <= K + 2
                    [V, D] = eig(full(A));
                    vals = diag(D);
                else
                    [V, D] = eigs(A, K, 'largestreal', opts);
                    vals = diag(D);
                end
            catch
                [V, D] = eig(full(A));
                vals = diag(D);
            end
            [vals, order] = sort(real(vals), 'descend');
            V = real(V(:, order));
            vals = vals(1:K);
            vecs = V(:, 1:K);
        end

        function [vals, U, stats] = gaussianRp(theta, K, spec, stats)
            n = size(theta, 1);
            ell = K + spec.rp_oversampling;
            t0 = tic;
            Y = randn(n, ell);
            stats.rp_draw_omega_sec = toc(t0);
            t0 = tic;
            for i = 1:(2 * spec.rp_power_iter + 1)
                Y = theta * Y;
            end
            stats.rp_power_iter_sec = toc(t0);
            t0 = tic;
            [Q, ~] = qr(Y, 0);
            stats.rp_qr_sec = toc(t0);
            t0 = tic;
            B = Q' * (theta * Q);
            B = (B + B') * 0.5;
            stats.rp_build_core_sec = toc(t0);
            t0 = tic;
            [vals, core] = uhsbm.Common.topEig(sparse(B), K, spec.eigs_tol);
            stats.rp_small_eig_sec = toc(t0);
            t0 = tic;
            U = Q * core;
            stats.rp_lift_sec = toc(t0);
        end

        function [vals, U, stats] = countSketchRp(theta, K, spec, stats)
            n = size(theta, 1);
            ell = K + spec.rp_oversampling;
            t0 = tic;
            cols = randi(ell, n, 1);
            signs = 2 * (rand(n, 1) > 0.5) - 1;
            omega = sparse((1:n)', cols, signs, n, ell);
            bucketCounts = accumarray(cols, 1, [ell, 1], @sum, 0);
            stats.cs_draw_hash_sec = toc(t0);
            stats.cs_embedding_dim = ell;
            stats.cs_bucket_min_load = min(bucketCounts);
            stats.cs_bucket_max_load = max(bucketCounts);
            stats.cs_empty_buckets = sum(bucketCounts == 0);
            t0 = tic;
            Y = full(theta * omega);
            stats.cs_initial_multiply_sec = toc(t0);
            t0 = tic;
            for i = 1:(2 * spec.rp_power_iter)
                Y = theta * Y;
            end
            stats.cs_power_iter_sec = toc(t0);
            t0 = tic;
            [Q, ~] = qr(Y, 0);
            stats.cs_qr_sec = toc(t0);
            t0 = tic;
            B = Q' * (theta * Q);
            B = (B + B') * 0.5;
            stats.cs_build_core_sec = toc(t0);
            t0 = tic;
            [vals, core] = uhsbm.Common.topEig(sparse(B), K, spec.eigs_tol);
            stats.cs_small_eig_sec = toc(t0);
            t0 = tic;
            U = Q * core;
            stats.cs_lift_sec = toc(t0);
        end

        function [sampled, s] = sampleTheta(theta, p)
            upper = triu(theta);
            [i, j, v] = find(upper);
            keep = rand(numel(v), 1) < p;
            i = i(keep); j = j(keep); v = v(keep) ./ p;
            off = i ~= j;
            sampled = sparse([i; j(off)], [j; i(off)], [v; v(off)], size(theta, 1), size(theta, 2));
            s = struct('original_upper_nnz', nnz(upper), 'sampled_upper_nnz', sum(keep));
        end

        function U = normalizeRows(U)
            nr = sqrt(sum(U .^ 2, 2));
            nr(nr == 0) = 1;
            U = U ./ nr;
        end

        function labels = simpleKmeans(X, K, replicates, maxIter)
            n = size(X, 1);
            bestLabels = ones(n, 1);
            bestInertia = inf;
            for rep = 1:replicates
                idx = randperm(n, K);
                centers = X(idx, :);
                labels = ones(n, 1);
                for it = 1:maxIter
                    dist = uhsbm.Common.sqDistances(X, centers);
                    [~, newLabels] = min(dist, [], 2);
                    if it > 1 && all(newLabels == labels)
                        break;
                    end
                    labels = newLabels;
                    for k = 1:K
                        mask = labels == k;
                        if any(mask)
                            centers(k, :) = mean(X(mask, :), 1);
                        else
                            centers(k, :) = X(randi(n), :);
                        end
                    end
                end
                dist = uhsbm.Common.sqDistances(X, centers);
                rowIdx = (1:n)';
                inertia = sum(dist(sub2ind(size(dist), rowIdx, labels)));
                if inertia < bestInertia
                    bestInertia = inertia;
                    bestLabels = labels;
                end
            end
            labels = bestLabels;
        end

        function D = sqDistances(X, C)
            x2 = sum(X .^ 2, 2);
            c2 = sum(C .^ 2, 2)';
            D = max(x2 + c2 - 2 * (X * C'), 0);
        end

        function mis = misclassification(yTrue, yPred, K)
            conf = zeros(K, K);
            for i = 1:numel(yTrue)
                if yPred(i) >= 1 && yPred(i) <= K
                    conf(yTrue(i), yPred(i)) = conf(yTrue(i), yPred(i)) + 1;
                end
            end
            correct = uhsbm.Common.maxAssignment(conf);
            mis = 1 - correct / numel(yTrue);
        end

        function best = maxAssignment(conf)
            K = size(conf, 1);
            nMasks = 2 ^ K;
            dp = -inf(K + 1, nMasks);
            dp(1, 1) = 0;
            for row = 1:K
                for mask = 0:(nMasks - 1)
                    current = dp(row, mask + 1);
                    if ~isfinite(current), continue; end
                    for col = 1:K
                        bit = bitshift(1, col - 1);
                        if bitand(mask, bit) == 0
                            newMask = bitor(mask, bit);
                            dp(row + 1, newMask + 1) = max(dp(row + 1, newMask + 1), current + conf(row, col));
                        end
                    end
                end
            end
            best = dp(K + 1, nMasks);
        end

        function ari = adjustedRandIndex(yTrue, yPred, K)
            labelsPred = unique(yPred(:))';
            C = zeros(K, numel(labelsPred));
            for i = 1:K
                for j = 1:numel(labelsPred)
                    C(i, j) = sum(yTrue == i & yPred == labelsPred(j));
                end
            end
            nij = sum(uhsbm.Common.choose2(C(:)));
            ai = sum(uhsbm.Common.choose2(sum(C, 2)));
            bj = sum(uhsbm.Common.choose2(sum(C, 1)));
            total = uhsbm.Common.choose2(numel(yTrue));
            expected = ai * bj / max(total, eps);
            maxIndex = 0.5 * (ai + bj);
            denom = maxIndex - expected;
            if abs(denom) < eps
                ari = double(nij == maxIndex);
            else
                ari = (nij - expected) / denom;
            end
        end

        function nmi = normalizedMutualInfo(yTrue, yPred, K)
            labelsPred = unique(yPred(:))';
            n = numel(yTrue);
            C = zeros(K, numel(labelsPred));
            for i = 1:K
                for j = 1:numel(labelsPred)
                    C(i, j) = sum(yTrue == i & yPred == labelsPred(j));
                end
            end
            pi = sum(C, 2) / n;
            pj = sum(C, 1) / n;
            pij = C / n;
            mi = 0;
            for i = 1:size(C, 1)
                for j = 1:size(C, 2)
                    if pij(i, j) > 0 && pi(i) > 0 && pj(j) > 0
                        mi = mi + pij(i, j) * log(pij(i, j) / (pi(i) * pj(j)));
                    end
                end
            end
            h1 = -sum(pi(pi > 0) .* log(pi(pi > 0)));
            h2 = -sum(pj(pj > 0) .* log(pj(pj > 0)));
            denom = (h1 + h2) / 2;
            if denom <= eps
                nmi = 1;
            else
                nmi = mi / denom;
            end
        end

        function v = choose2(x)
            v = x .* (x - 1) ./ 2;
        end

        function rec = emptyRecord()
            rec = struct( ...
                'experiment', string(missing), 'sweep', string(missing), 'x_col', string(missing), ...
                'x_value', NaN, 'density_level', NaN, 'rep', NaN, 'seed', NaN, ...
                'n', NaN, 'K', NaN, 'm', NaN, 'rho_n', NaN, 'a_in', NaN, 'b_out', NaN, ...
                'signal_gap', NaN, 'p_in', NaN, 'p_out', NaN, 'num_hyperedges_total', NaN, ...
                'theta_nnz', NaN, 'theta_density', NaN, 'generation_wall_sec', NaN, ...
                'hypergraph_laplacian_build_wall_sec', NaN, 'sampling_mode', string(missing), ...
                'num_isolated_nodes', NaN, 'isolated_fraction', NaN, 'hypergraph_degree_mean', NaN, ...
                'hypergraph_degree_max', NaN, 'expected_hyperedges_total', NaN, ...
                'expected_hyperedges_per_n', NaN, 'expected_degree_mean', NaN, ...
                'candidate_within_fraction', NaN, 'method', string(missing), 'method_key', string(missing), ...
                'misclassification_rate', NaN, 'ARI', NaN, 'NMI', NaN, 'metric_wall_sec', NaN, ...
                'method_wall_sec', NaN, 'algorithm_total_wall_sec', NaN);
            stats = uhsbm.Common.emptyStats();
            names = fieldnames(stats);
            for i = 1:numel(names)
                rec.(names{i}) = stats.(names{i});
            end
        end

        function stats = emptyStats()
            stats = struct( ...
                'non_random_eigensolver', string(missing), 'eigen_decomposition_wall_sec', NaN, ...
                'embedding_normalize_wall_sec', NaN, 'kmeans_wall_sec', NaN, ...
                'spectral_clustering_wall_sec', NaN, 'top_eigenvalue_max', NaN, 'top_eigenvalue_min', NaN, ...
                'rp_draw_omega_sec', NaN, 'rp_power_iter_sec', NaN, 'rp_qr_sec', NaN, ...
                'rp_build_core_sec', NaN, 'rp_small_eig_sec', NaN, 'rp_lift_sec', NaN, ...
                'rs_sample_matrix_wall_sec', NaN, 'rs_original_upper_nnz', NaN, ...
                'rs_sampled_upper_nnz', NaN, 'rs_sampling_probability', NaN, 'rs_sampled_theta_nnz', NaN, ...
                'cs_draw_hash_sec', NaN, 'cs_embedding_dim', NaN, 'cs_bucket_min_load', NaN, ...
                'cs_bucket_max_load', NaN, 'cs_empty_buckets', NaN, 'cs_initial_multiply_sec', NaN, ...
                'cs_power_iter_sec', NaN, 'cs_qr_sec', NaN, 'cs_build_core_sec', NaN, ...
                'cs_small_eig_sec', NaN, 'cs_lift_sec', NaN);
        end

        function rec = attachStats(rec, stats)
            names = fieldnames(stats);
            for i = 1:numel(names)
                rec.(names{i}) = stats.(names{i});
            end
        end

        function summary = summarizeRaw(raw)
            methods = uhsbm.Common.methodLabels();
            groups = unique(raw(:, {'experiment', 'sweep', 'x_col', 'x_value', 'method'}), 'rows', 'stable');
            rows = repmat(uhsbm.Common.emptySummaryRecord(), height(groups), 1);
            numericFields = uhsbm.Common.summaryNumericFields();
            for i = 1:height(groups)
                mask = raw.experiment == groups.experiment(i) & raw.x_value == groups.x_value(i) & raw.method == groups.method(i);
                sub = raw(mask, :);
                rec = uhsbm.Common.emptySummaryRecord();
                rec.experiment = groups.experiment(i);
                rec.sweep = groups.sweep(i);
                rec.x_col = groups.x_col(i);
                rec.x_value = groups.x_value(i);
                rec.method = groups.method(i);
                rec.reps = height(sub);
                for f = 1:numel(numericFields)
                    name = numericFields{f};
                    if ismember(name, sub.Properties.VariableNames)
                        vals = sub.(name);
                        rec.([name, '_mean']) = mean(vals, 'omitnan');
                        rec.([name, '_std']) = std(vals, 0, 'omitnan');
                    end
                end
                rows(i) = rec;
            end
            summary = struct2table(rows);
            if ismember('spectral_clustering_wall_sec_mean', summary.Properties.VariableNames)
                for i = 1:height(summary)
                    base = summary(summary.x_value == summary.x_value(i) & summary.method == "Non-random eigs", :);
                    if ~isempty(base)
                        summary.non_random_spectral_sec_mean(i) = base.spectral_clustering_wall_sec_mean(1);
                        summary.spectral_speedup_vs_non_random(i) = base.spectral_clustering_wall_sec_mean(1) / summary.spectral_clustering_wall_sec_mean(i);
                    end
                end
            end
            [~, methodRank] = ismember(cellstr(summary.method), methods);
            [~, order] = sortrows([summary.x_value, methodRank]);
            summary = summary(order, :);
        end

        function fields = summaryNumericFields()
            fields = {'n', 'K', 'rho_n', 'a_in', 'b_out', 'signal_gap', 'p_in', 'p_out', ...
                'num_hyperedges_total', 'theta_nnz', 'theta_density', 'num_isolated_nodes', ...
                'isolated_fraction', 'hypergraph_degree_mean', 'hypergraph_degree_max', ...
                'expected_hyperedges_total', 'expected_hyperedges_per_n', 'expected_degree_mean', ...
                'candidate_within_fraction', 'misclassification_rate', 'ARI', 'NMI', ...
                'generation_wall_sec', 'hypergraph_laplacian_build_wall_sec', ...
                'eigen_decomposition_wall_sec', 'embedding_normalize_wall_sec', 'kmeans_wall_sec', ...
                'spectral_clustering_wall_sec', 'metric_wall_sec', 'method_wall_sec', ...
                'algorithm_total_wall_sec', 'rp_power_iter_sec', 'rp_qr_sec', 'rp_build_core_sec', ...
                'rs_sample_matrix_wall_sec', 'rs_sampled_upper_nnz', 'cs_initial_multiply_sec', ...
                'cs_power_iter_sec', 'cs_qr_sec', 'cs_build_core_sec', 'cs_embedding_dim', ...
                'top_eigenvalue_max', 'top_eigenvalue_min'};
        end

        function rec = emptySummaryRecord()
            rec = struct('experiment', string(missing), 'sweep', string(missing), 'x_col', string(missing), ...
                'x_value', NaN, 'method', string(missing), 'reps', NaN, ...
                'non_random_spectral_sec_mean', NaN, 'spectral_speedup_vs_non_random', NaN);
            fields = uhsbm.Common.summaryNumericFields();
            for i = 1:numel(fields)
                rec.([fields{i}, '_mean']) = NaN;
                rec.([fields{i}, '_std']) = NaN;
            end
        end

        function plotSummary(summary, spec, plotPath)
            f = figure('Visible', 'off', 'Position', [100, 100, 1250, 820]);
            panels = { ...
                'misclassification_rate_mean', 'Misclassification'; ...
                'theta_nnz_mean', 'Theta nnz'; ...
                'spectral_clustering_wall_sec_mean', 'Spectral time (sec)'; ...
                'spectral_speedup_vs_non_random', 'Spectral speedup vs non-random'};
            methods = uhsbm.Common.methodLabels();
            for p = 1:4
                subplot(2, 2, p);
                hold on;
                col = panels{p, 1};
                for m = 1:numel(methods)
                    sub = summary(summary.method == string(methods{m}), :);
                    if isempty(sub), continue; end
                    plot(sub.x_value, sub.(col), '-o', 'LineWidth', 1.8, 'Color', uhsbm.Common.methodColor(methods{m}), 'DisplayName', methods{m});
                end
                grid on;
                xlabel(spec.x_col, 'Interpreter', 'none');
                ylabel(panels{p, 2});
                if p == 1, legend('Location', 'best', 'Interpreter', 'none'); end
            end
            sgtitle(spec.title, 'Interpreter', 'none');
            exportgraphics(f, plotPath, 'Resolution', 180);
            close(f);
        end

        function writeJsonConfig(spec, path)
            cfg = spec;
            cfg.methods = string(uhsbm.Common.methodLabels());
            text = jsonencode(cfg, PrettyPrint=true);
            fid = fopen(path, 'w');
            fprintf(fid, '%s\n', text);
            fclose(fid);
        end

        function out = runRedesignedDiagnostics(specs, resultsRoot, opts)
            diagDir = fullfile(resultsRoot, 'diagnostics');
            if ~exist(diagDir, 'dir'), mkdir(diagDir); end
            spectrumPath = fullfile(diagDir, 'spectral_gap_diagnostics.csv');
            paramPath = fullfile(diagDir, 'randomization_parameter_diagnostic.csv');

            rows = repmat(struct('experiment', string(missing), 'x_col', string(missing), 'x_value', NaN, ...
                'n', NaN, 'K', NaN, 'rho_n', NaN, 'a_in', NaN, 'b_out', NaN, ...
                'num_hyperedges_total', NaN, 'theta_nnz', NaN, 'lambda_K', NaN, ...
                'lambda_Kplus1', NaN, 'relative_eigengap_after_K', NaN), 0, 1);
            for sidx = [1, 2]
                spec = specs(sidx);
                for xv = spec.x_values
                    [theta, ~, meta] = uhsbm.Common.buildDiagnosticTheta(spec, xv, 1);
                    kk = min(size(theta, 1) - 2, meta.K + 10);
                    vals = eigs(theta, kk, 'largestreal');
                    vals = sort(real(vals), 'descend');
                    row = struct();
                    row.experiment = string(spec.name);
                    row.x_col = string(spec.x_col);
                    row.x_value = xv;
                    row.n = meta.n; row.K = meta.K; row.rho_n = meta.rho_n;
                    row.a_in = meta.a_in; row.b_out = meta.b_out;
                    row.num_hyperedges_total = meta.num_hyperedges_total;
                    row.theta_nnz = nnz(theta);
                    row.lambda_K = vals(meta.K);
                    row.lambda_Kplus1 = vals(meta.K + 1);
                    row.relative_eigengap_after_K = (row.lambda_K - row.lambda_Kplus1) / max(abs(row.lambda_K), eps);
                    rows(end + 1) = row; %#ok<AGROW>
                end
            end
            writetable(struct2table(rows), spectrumPath);

            base = specs(1);
            [theta, labels, meta] = uhsbm.Common.buildDiagnosticTheta(base, 5, 1);
            configs = { ...
                'Non-random eigs', 'non_random', 'baseline', 30, 1, 0.3; ...
                'Gaussian RP r=30 q=1', 'gaussian_random_projection', 'fast', 30, 1, 0.3; ...
                'Gaussian RP r=160 q=3', 'gaussian_random_projection', 'wide', 160, 3, 0.3; ...
                'CountSketch RP r=30 q=1', 'countsketch_random_projection', 'fast', 30, 1, 0.3; ...
                'CountSketch RP r=160 q=3', 'countsketch_random_projection', 'wide', 160, 3, 0.3; ...
                'Random sampling p=0.3', 'random_sampling', 'fast', 30, 1, 0.3; ...
                'Random sampling p=0.7', 'random_sampling', 'less_sparse', 30, 1, 0.7; ...
                'Random sampling p=0.9', 'random_sampling', 'near_full', 30, 1, 0.9; ...
                'Random sampling p=1.0', 'random_sampling', 'full_control', 30, 1, 1.0};
            paramRows = repmat(struct('method', string(missing), 'setting', string(missing), ...
                'misclassification_rate', NaN, 'ARI', NaN, 'NMI', NaN, ...
                'spectral_clustering_wall_sec', NaN, 'spectral_speedup_vs_non_random', NaN), 0, 1);
            baseSec = NaN;
            for i = 1:size(configs, 1)
                diagSpec = base;
                diagSpec.rp_oversampling = configs{i, 4};
                diagSpec.rp_power_iter = configs{i, 5};
                diagSpec.random_sampling_p = configs{i, 6};
                rng(uhsbm.Common.normalizeSeed(meta.seed + i * 1009), 'twister');
                [pred, st] = uhsbm.Common.spectralCluster(theta, meta.K, diagSpec, configs{i, 2});
                row = struct();
                row.method = string(configs{i, 1});
                row.setting = string(configs{i, 3});
                row.misclassification_rate = uhsbm.Common.misclassification(labels, pred, meta.K);
                row.ARI = uhsbm.Common.adjustedRandIndex(labels, pred, meta.K);
                row.NMI = uhsbm.Common.normalizedMutualInfo(labels, pred, meta.K);
                row.spectral_clustering_wall_sec = st.spectral_clustering_wall_sec;
                if i == 1, baseSec = st.spectral_clustering_wall_sec; end
                row.spectral_speedup_vs_non_random = baseSec / st.spectral_clustering_wall_sec;
                paramRows(end + 1) = row; %#ok<AGROW>
            end
            writetable(struct2table(paramRows), paramPath);
            out = struct('spectral_gap', spectrumPath, 'randomization_parameter', paramPath);

            if opts.smoke
                % Keep smoke diagnostics present but intentionally small.
            end
        end

        function [theta, labels, meta] = buildDiagnosticTheta(spec, xValue, rep)
            [n, K, rhoN, aIn, bOut] = uhsbm.Common.concreteParams(spec, xValue);
            seed = uhsbm.Common.normalizeSeed(double(spec.seed) + uhsbm.Common.valueToSeedComponent(xValue) * 100000 + rep);
            rng(seed, 'twister');
            labels = uhsbm.Common.makeBalancedLabels(n, K);
            pIn = aIn * rhoN / (n ^ (spec.m - 1));
            pOut = bOut * rhoN / (n ^ (spec.m - 1));
            [edges, ~] = uhsbm.Common.sampleUniformHsbm3(labels, K, pIn, pOut);
            [theta, ~] = uhsbm.Common.buildTheta(n, edges);
            meta = struct('seed', seed, 'n', n, 'K', K, 'rho_n', rhoN, 'a_in', aIn, 'b_out', bOut, ...
                'num_hyperedges_total', size(edges, 1));
        end

        function path = writeUniformReport(thisDir, specs, opts)
            path = fullfile(thisDir, 'uniform_hsbm_matlab_experiment_report.md');
            fid = fopen(path, 'w');
            fprintf(fid, '# 균일 HSBM MATLAB 실험 보고서\n\n');
            fprintf(fid, '이 보고서는 `experiments/균일 HSBM 실험`의 method comparison을 MATLAB 코드로 다시 구현해 실행한 결과입니다. Python 공용 모듈은 호출하지 않았고, 하이퍼그래프 생성과 normalized operator 구성부터 metric 계산까지 MATLAB 안에서 수행했습니다.\n\n');
            fprintf(fid, '## 실행 요약\n\n');
            fprintf(fid, '| 항목 | 값 |\n|---|---|\n');
            fprintf(fid, '| 구현 위치 | `experiments/균일 HSBM 실험_matlab/` |\n');
            fprintf(fid, '| MATLAB 버전 | `%s` |\n', version);
            fprintf(fid, '| 반복 횟수 | `%d` |\n', specs(1).reps);
            fprintf(fid, '| seed | `%d` |\n', specs(1).seed);
            fprintf(fid, '| smoke 실행 | `%s` |\n\n', string(opts.smoke));
            fprintf(fid, '비교 방법은 `Non-random eigs`, `Gaussian RP`, `Random sampling`, `CountSketch RP` 네 가지입니다.\n\n');
            fprintf(fid, '## 전체 요약\n\n');
            uhsbm.Common.writeAllSummaryTable(fid, fullfile(thisDir, 'results'), specs);
            for i = 1:numel(specs)
                uhsbm.Common.writeSpecSection(fid, fullfile(thisDir, 'results'), specs(i));
            end
            fprintf(fid, '## 해석 메모\n\n');
            fprintf(fid, '- 오분류율은 label permutation을 DP assignment로 맞춘 뒤 계산했습니다.\n');
            fprintf(fid, '- ARI와 NMI는 MATLAB 구현으로 직접 계산했습니다.\n');
            fprintf(fid, '- MATLAB과 Python은 RNG, `eigs`, `kmeans` 초기화가 달라 수치가 완전히 같지는 않습니다.\n');
            fprintf(fid, '- 이번 MATLAB 폴더의 결과는 기존 Python 결과를 덮어쓰지 않고 별도 `results/`에 저장했습니다.\n');
            fclose(fid);
        end

        function path = writeRedesignedReport(thisDir, specs, opts)
            path = fullfile(thisDir, 'redesigned_uniform_hsbm_matlab_experiment_report.md');
            fid = fopen(path, 'w');
            fprintf(fid, '# 균일 HSBM density-signal 재설계 MATLAB 실험 보고서\n\n');
            fprintf(fid, '이 보고서는 Python 재설계 실험을 MATLAB 코드로 다시 구현해 실행한 결과입니다. strong-signal sweep과 weak-gap diagnostic sweep을 분리해 저장했고, 결과는 이 MATLAB 폴더 아래에만 남겼습니다.\n\n');
            fprintf(fid, '## 실행 요약\n\n');
            fprintf(fid, '| 항목 | 값 |\n|---|---|\n');
            fprintf(fid, '| 구현 위치 | `experiments/균일 HSBM density-signal 재설계 실험_matlab/` |\n');
            fprintf(fid, '| MATLAB 버전 | `%s` |\n', version);
            fprintf(fid, '| 반복 횟수 | `%d` |\n', specs(1).reps);
            fprintf(fid, '| seed 기준 | `%d` |\n', floor(specs(1).seed / 100));
            fprintf(fid, '| smoke 실행 | `%s` |\n\n', string(opts.smoke));
            fprintf(fid, '## 전체 요약\n\n');
            uhsbm.Common.writeAllSummaryTable(fid, fullfile(thisDir, 'results'), specs);

            diagPath = fullfile(thisDir, 'results', 'diagnostics', 'spectral_gap_diagnostics.csv');
            if exist(diagPath, 'file')
                diag = readtable(diagPath, 'TextType', 'string');
                fprintf(fid, '## 스펙트럼 진단\n\n');
                fprintf(fid, '| block | x | K | Theta nnz | lambda_K | lambda_K+1 | relative gap |\n');
                fprintf(fid, '|---|---:|---:|---:|---:|---:|---:|\n');
                for i = 1:height(diag)
                    fprintf(fid, '| %s | %.4g | %d | %.0f | %.6f | %.6f | %.6f |\n', ...
                        diag.experiment(i), diag.x_value(i), diag.K(i), diag.theta_nnz(i), ...
                        diag.lambda_K(i), diag.lambda_Kplus1(i), diag.relative_eigengap_after_K(i));
                end
                fprintf(fid, '\n');
            end

            paramPath = fullfile(thisDir, 'results', 'diagnostics', 'randomization_parameter_diagnostic.csv');
            if exist(paramPath, 'file')
                param = readtable(paramPath, 'TextType', 'string');
                fprintf(fid, '## 랜덤화 파라미터 진단\n\n');
                fprintf(fid, '| method | setting | 오분류율 | ARI | NMI | spectral초 | speedup |\n');
                fprintf(fid, '|---|---|---:|---:|---:|---:|---:|\n');
                for i = 1:height(param)
                    fprintf(fid, '| %s | %s | %.4f | %.4f | %.4f | %.4f | %.4f |\n', ...
                        param.method(i), param.setting(i), param.misclassification_rate(i), ...
                        param.ARI(i), param.NMI(i), param.spectral_clustering_wall_sec(i), ...
                        param.spectral_speedup_vs_non_random(i));
                end
                fprintf(fid, '\n');
            end

            for i = 1:numel(specs)
                uhsbm.Common.writeSpecSection(fid, fullfile(thisDir, 'results'), specs(i));
            end
            fprintf(fid, '## 해석 메모\n\n');
            fprintf(fid, '- `speedup`은 generation/build를 제외한 spectral clustering 단계 기준입니다.\n');
            fprintf(fid, '- randomized method의 speedup은 Non-random과 정확도가 비슷할 때만 의미 있게 해석해야 합니다.\n');
            fprintf(fid, '- MATLAB 구현은 Python과 같은 설계를 따르지만 RNG와 solver 차이로 결과값은 달라질 수 있습니다.\n');
            fclose(fid);
        end

        function writeAllSummaryTable(fid, resultsRoot, specs)
            fprintf(fid, '| block | method | 평균 오분류율 | 평균 ARI | 평균 NMI | 평균 Theta nnz | 평균 spectral초 | 평균 speedup |\n');
            fprintf(fid, '|---|---|---:|---:|---:|---:|---:|---:|\n');
            for s = 1:numel(specs)
                summary = readtable(fullfile(resultsRoot, specs(s).name, [specs(s).name, '_summary.csv']), 'TextType', 'string');
                methods = unique(summary.method, 'stable');
                for m = 1:numel(methods)
                    sub = summary(summary.method == methods(m), :);
                    fprintf(fid, '| %s | %s | %.4f | %.4f | %.4f | %.1f | %.4f | %.4f |\n', ...
                        specs(s).name, methods(m), mean(sub.misclassification_rate_mean, 'omitnan'), ...
                        mean(sub.ARI_mean, 'omitnan'), mean(sub.NMI_mean, 'omitnan'), ...
                        mean(sub.theta_nnz_mean, 'omitnan'), mean(sub.spectral_clustering_wall_sec_mean, 'omitnan'), ...
                        mean(sub.spectral_speedup_vs_non_random, 'omitnan'));
                end
            end
            fprintf(fid, '\n');
        end

        function writeSpecSection(fid, resultsRoot, spec)
            summaryPath = fullfile(resultsRoot, spec.name, [spec.name, '_summary.csv']);
            summary = readtable(summaryPath, 'TextType', 'string');
            fprintf(fid, '## %s\n\n', spec.title);
            xvals = unique(summary.x_value, 'stable')';
            for xv = xvals
                sub = summary(summary.x_value == xv, :);
                fprintf(fid, '### %s = %.4g\n\n', spec.x_col, xv);
                fprintf(fid, '| 방법 | 오분류율 | ARI | NMI | spectral초 | speedup |\n');
                fprintf(fid, '|---|---:|---:|---:|---:|---:|\n');
                [~, bestIdx] = min(sub.misclassification_rate_mean);
                for i = 1:height(sub)
                    row = sprintf('| %s | %.4f | %.4f | %.4f | %.4f | %.4f |', ...
                        sub.method(i), sub.misclassification_rate_mean(i), sub.ARI_mean(i), ...
                        sub.NMI_mean(i), sub.spectral_clustering_wall_sec_mean(i), ...
                        sub.spectral_speedup_vs_non_random(i));
                    if i == bestIdx
                        row = strrep(row, '| ', '| **');
                        row = strrep(row, ' |', '** |');
                    end
                    fprintf(fid, '%s\n', row);
                end
                fprintf(fid, '\n');
            end
            relPlot = fullfile('results', spec.name, [spec.name, '_summary.png']);
            fprintf(fid, '![%s](%s)\n\n', spec.title, relPlot);
        end

        function c = valueToSeedComponent(value)
            c = round(double(value) * 1000);
        end

        function seed = normalizeSeed(seed)
            seed = mod(round(double(seed)), 2 ^ 32 - 1);
            if seed < 0
                seed = seed + (2 ^ 32 - 1);
            end
        end
    end
end
