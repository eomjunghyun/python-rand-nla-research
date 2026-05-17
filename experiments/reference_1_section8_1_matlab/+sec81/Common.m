classdef Common
    % Standalone MATLAB implementation of Reference 1 Section 8.1 utilities.

    methods (Static)
        function output = runDatasetByName(datasetName, varargin)
            thisDir = fileparts(fileparts(mfilename('fullpath')));
            repoRoot = fileparts(fileparts(thisDir));

            p = inputParser;
            addParameter(p, 'embedding_rank', []);
            addParameter(p, 'reps', 20);
            addParameter(p, 'seed', 2026);
            addParameter(p, 'q', 2);
            addParameter(p, 'r', 10);
            addParameter(p, 'p_values', [0.7, 0.8]);
            addParameter(p, 'sign_k', 2);
            addParameter(p, 'force_prepare', false);
            addParameter(p, 'outdir', '');
            addParameter(p, 'no_progress', false);
            addParameter(p, 'no_plot', false);
            parse(p, varargin{:});

            datasets = sec81.Common.loadOrPrepareData(thisDir, repoRoot, p.Results.force_prepare);
            ds = sec81.Common.getDataset(datasets, datasetName);
            if isempty(ds)
                error('Unknown dataset: %s', char(datasetName));
            end

            if isempty(p.Results.embedding_rank)
                embeddingRank = ds.target_rank;
            else
                embeddingRank = double(p.Results.embedding_rank);
            end
            if embeddingRank < 1
                error('embedding_rank must be positive.');
            end
            if embeddingRank >= size(ds.A, 1)
                error('embedding_rank must be smaller than the number of nodes.');
            end

            pValues = sec81.Common.parsePValues(p.Results.p_values);
            runInfo = sec81.Common.runInfo(ds, embeddingRank);
            if strlength(string(p.Results.outdir)) > 0
                outdir = char(p.Results.outdir);
            else
                outdir = fullfile(thisDir, 'results', runInfo.outdir_name);
            end
            if ~exist(outdir, 'dir')
                mkdir(outdir);
            end

            fprintf('Running %s: nodes=%d, edges=%d, cluster_count=%d, embedding_rank=%d\n', ...
                ds.display_name, size(ds.A, 1), nnz(triu(ds.A, 1)), ds.cluster_count, embeddingRank);

            [raw, pairwiseRaw] = sec81.Common.runExperiment(ds, embeddingRank, ...
                p.Results.reps, p.Results.seed, p.Results.q, p.Results.r, ...
                pValues, p.Results.sign_k, ~p.Results.no_progress);
            summary = sec81.Common.summarizeRaw(raw, pValues, ds.has_ground_truth);
            table2 = sec81.Common.buildTable2Like(summary, pValues, ds.has_ground_truth);

            rawCsv = fullfile(outdir, sprintf('%s_raw_per_rep.csv', runInfo.prefix));
            summaryCsv = fullfile(outdir, sprintf('%s_summary_mean_std.csv', runInfo.prefix));
            tableCsv = fullfile(outdir, sprintf('%s_%s_like.csv', runInfo.prefix, runInfo.table_id));
            tableMd = fullfile(outdir, sprintf('%s_%s_like.md', runInfo.prefix, runInfo.table_id));
            pairCsv = fullfile(outdir, sprintf('%s_pairwise_ari_raw.csv', runInfo.prefix));
            pairMatCsv = fullfile(outdir, sprintf('%s_pairwise_ari_mean_matrix.csv', runInfo.prefix));
            heatmapPng = fullfile(outdir, sprintf('%s_pairwise_ari_heatmap.png', runInfo.prefix));
            metaJson = fullfile(outdir, sprintf('%s_meta.json', runInfo.prefix));

            writetable(raw, rawCsv);
            writetable(summary, summaryCsv);
            writetable(table2, tableCsv);
            sec81.Common.writeTable2Markdown(table2, tableMd, runInfo, ds, p.Results.reps);
            writetable(pairwiseRaw, pairCsv);
            pairMat = sec81.Common.pairwiseMeanMatrix(pairwiseRaw);
            sec81.Common.writeMatrixCsv(pairMat.methods, pairMat.matrix, pairMatCsv);
            if ~p.Results.no_plot
                sec81.Common.plotPairwiseHeatmap(pairMat.methods, pairMat.matrix, heatmapPng);
            end
            sec81.Common.writeMetaJson(metaJson, ds, embeddingRank, p.Results, pValues, runInfo);

            fprintf('Done: %s\n', outdir);
            fprintf('Raw CSV     : %s\n', rawCsv);
            fprintf('Summary CSV : %s\n', summaryCsv);
            fprintf('Table MD    : %s\n', tableMd);

            output = struct();
            output.dataset = string(ds.name);
            output.display_name = string(ds.display_name);
            output.has_ground_truth = ds.has_ground_truth;
            output.paper_rank = ds.target_rank;
            output.embedding_rank = embeddingRank;
            output.cluster_count = ds.cluster_count;
            output.outdir = string(outdir);
            output.prefix = string(runInfo.prefix);
            output.table_id = string(runInfo.table_id);
            output.raw_csv = string(rawCsv);
            output.summary_csv = string(summaryCsv);
            output.table_csv = string(tableCsv);
            output.table_md = string(tableMd);
            output.pairwise_csv = string(pairCsv);
            output.pairwise_matrix_csv = string(pairMatCsv);
            output.heatmap_png = string(heatmapPng);
            output.meta_json = string(metaJson);
            output.summary = summary;
        end

        function datasets = loadOrPrepareData(thisDir, repoRoot, forcePrepare)
            dataDir = fullfile(thisDir, 'data');
            if ~exist(dataDir, 'dir')
                mkdir(dataDir);
            end
            cachePath = fullfile(dataDir, 'section8_1_matlab_inputs.mat');
            if exist(cachePath, 'file') && ~forcePrepare
                loaded = load(cachePath, 'datasets');
                datasets = loaded.datasets;
                return;
            end

            fprintf('Preparing Section 8.1 MATLAB input cache...\n');
            datasets = struct([]);
            datasets = sec81.Common.appendStruct(datasets, sec81.Common.prepareEmailDataset(repoRoot));
            datasets = sec81.Common.appendStruct(datasets, sec81.Common.preparePoliticalBlogDataset(repoRoot));
            [coauthor, citation] = sec81.Common.prepareStatisticiansDatasets(repoRoot);
            datasets = sec81.Common.appendStruct(datasets, coauthor);
            datasets = sec81.Common.appendStruct(datasets, citation);
            save(cachePath, 'datasets', '-v7.3');
            fprintf('Saved MATLAB input cache: %s\n', cachePath);
        end

        function ds = getDataset(datasets, datasetName)
            ds = [];
            key = string(datasetName);
            for i = 1:numel(datasets)
                if strcmp(string(datasets(i).name), key)
                    ds = datasets(i);
                    return;
                end
            end
        end

        function ds = prepareEmailDataset(repoRoot)
            edgePath = fullfile(repoRoot, 'data', 'email-Eu-core.txt');
            labelPath = fullfile(repoRoot, 'data', 'email-Eu-core-department-labels.txt');
            edges = readmatrix(edgePath, 'FileType', 'text');
            labelsRaw = readmatrix(labelPath, 'FileType', 'text');

            maxNode = max([edges(:); labelsRaw(:, 1)]) + 1;
            n = double(maxNode);
            rows = double(edges(:, 1)) + 1;
            cols = double(edges(:, 2)) + 1;
            keep = rows ~= cols;
            rows = rows(keep);
            cols = cols(keep);
            A = sparse([rows; cols], [cols; rows], 1, n, n);
            A = spones(A);
            A(1:n + 1:end) = 0;

            yAll = nan(n, 1);
            yAll(double(labelsRaw(:, 1)) + 1) = double(labelsRaw(:, 2));
            [A, idx, compSizes] = sec81.Common.largestComponent(A);
            y = yAll(idx);
            if any(isnan(y))
                error('Email LCC contains nodes with missing labels.');
            end
            y = sec81.Common.remapLabels(y);

            ds = sec81.Common.baseDatasetStruct();
            ds.name = "email_eu";
            ds.display_name = "European email network";
            ds.A = A;
            ds.labels = y;
            ds.has_ground_truth = true;
            ds.target_rank = numel(unique(y));
            ds.cluster_count = numel(unique(y));
            ds.table_id = "table2a";
            ds.default_prefix = "email_eu";
            ds.changed_prefix = "email_eu";
            ds.default_outdir = "exp8_1_email_eu_core_table2_like";
            ds.changed_outdir = "exp8_1_email_eu_core_rank30_table2_like";
            ds.changed_rank = 30;
            ds.meta = struct( ...
                'directed_nodes', n, ...
                'directed_edges', size(edges, 1), ...
                'lcc_nodes', size(A, 1), ...
                'lcc_edges', nnz(triu(A, 1)), ...
                'num_classes', ds.cluster_count, ...
                'component_sizes_top10', compSizes(1:min(10, numel(compSizes))) ...
            );
        end

        function ds = preparePoliticalBlogDataset(repoRoot)
            gmlPath = fullfile(repoRoot, 'data', 'reference_1_section8_1', 'raw', 'polblogs', 'polblogs.gml');
            [nodeIds, nodeValues, nodeLabels, edgeSources, edgeTargets, selfLoops] = sec81.Common.parsePolblogsGml(gmlPath);
            sortedNodeIds = sort(nodeIds(:));
            n = numel(sortedNodeIds);
            nodeMap = containers.Map('KeyType', 'double', 'ValueType', 'double');
            valueMap = containers.Map('KeyType', 'double', 'ValueType', 'double');
            labelMap = containers.Map('KeyType', 'double', 'ValueType', 'char');
            for i = 1:n
                nodeMap(sortedNodeIds(i)) = i;
            end
            for i = 1:numel(nodeIds)
                valueMap(nodeIds(i)) = nodeValues(i);
                labelMap(nodeIds(i)) = nodeLabels{i};
            end

            rows = zeros(numel(edgeSources), 1);
            cols = zeros(numel(edgeSources), 1);
            used = false(numel(edgeSources), 1);
            for i = 1:numel(edgeSources)
                u = edgeSources(i);
                v = edgeTargets(i);
                if u == v || ~isKey(nodeMap, u) || ~isKey(nodeMap, v)
                    continue;
                end
                a = nodeMap(u);
                b = nodeMap(v);
                if a > b
                    tmp = a;
                    a = b;
                    b = tmp;
                end
                rows(i) = a;
                cols(i) = b;
                used(i) = true;
            end
            edgePairs = unique([rows(used), cols(used)], 'rows');
            A = sparse([edgePairs(:, 1); edgePairs(:, 2)], [edgePairs(:, 2); edgePairs(:, 1)], 1, n, n);
            A = spones(A);
            A(1:n + 1:end) = 0;
            [A, idx, compSizes] = sec81.Common.largestComponent(A);
            lccNodeIds = sortedNodeIds(idx);

            y = zeros(numel(lccNodeIds), 1);
            nodeNames = strings(numel(lccNodeIds), 1);
            for i = 1:numel(lccNodeIds)
                y(i) = valueMap(lccNodeIds(i));
                nodeNames(i) = string(labelMap(lccNodeIds(i)));
            end
            y = sec81.Common.remapLabels(y);

            ds = sec81.Common.baseDatasetStruct();
            ds.name = "political_blog";
            ds.display_name = "Political blog network";
            ds.A = A;
            ds.labels = y;
            ds.has_ground_truth = true;
            ds.target_rank = 2;
            ds.cluster_count = 2;
            ds.table_id = "table2b";
            ds.default_prefix = "political_blog";
            ds.changed_prefix = "political_blog_rank5";
            ds.default_outdir = "exp8_1_political_blog_table2_like";
            ds.changed_outdir = "exp8_1_political_blog_rank5_table2_like";
            ds.changed_rank = 5;
            ds.meta = struct( ...
                'raw_nodes', numel(nodeIds), ...
                'raw_directed_edges', numel(edgeSources), ...
                'raw_self_loops', selfLoops, ...
                'lcc_nodes', size(A, 1), ...
                'lcc_edges', nnz(triu(A, 1)), ...
                'num_classes', ds.cluster_count, ...
                'component_sizes_top10', compSizes(1:min(10, numel(compSizes))), ...
                'node_names_preview', nodeNames(1:min(5, numel(nodeNames))) ...
            );
        end

        function [coauthor, citation] = prepareStatisticiansDatasets(repoRoot)
            base = fullfile(repoRoot, 'data', 'reference_1_section8_1', 'raw', 'scc2016', ...
                'SCC2016-with-abs', 'SCC2016', 'Data');
            authorPaperPath = fullfile(base, 'authorPaperBiadj.txt');
            paperCitPath = fullfile(base, 'paperCitAdj.txt');
            authorListPath = fullfile(base, 'authorList.txt');

            authorNames = readlines(authorListPath);
            authorNames = strtrim(authorNames(authorNames ~= ""));

            fprintf('Reading statisticians author-paper matrix...\n');
            authorPaper = sparse(readmatrix(authorPaperPath, 'FileType', 'text'));
            authorPaper = spones(authorPaper);
            fprintf('Reading statisticians paper-citation matrix...\n');
            paperCit = sparse(readmatrix(paperCitPath, 'FileType', 'text'));
            paperCit = spones(paperCit);

            fprintf('Building coauthor graph...\n');
            coauthorRaw = spones(authorPaper * authorPaper');
            coauthorRaw(1:size(coauthorRaw, 1) + 1:end) = 0;
            coauthorRaw = spones(coauthorRaw);
            [Aco, coIdx, coSizes] = sec81.Common.largestComponent(coauthorRaw);

            coauthor = sec81.Common.baseDatasetStruct();
            coauthor.name = "statisticians_coauthor";
            coauthor.display_name = "Statisticians coauthor network (No true labels)";
            coauthor.A = Aco;
            coauthor.labels = [];
            coauthor.has_ground_truth = false;
            coauthor.target_rank = 3;
            coauthor.cluster_count = 3;
            coauthor.table_id = "table2c";
            coauthor.default_prefix = "statisticians_coauthor";
            coauthor.changed_prefix = "statisticians_coauthor_rank5";
            coauthor.default_outdir = "exp8_1_statisticians_coauthor_table2_like";
            coauthor.changed_outdir = "exp8_1_statisticians_coauthor_rank5_table2_like";
            coauthor.changed_rank = 5;
            coauthor.meta = struct( ...
                'raw_authors', size(coauthorRaw, 1), ...
                'raw_edges', nnz(triu(coauthorRaw, 1)), ...
                'lcc_nodes', size(Aco, 1), ...
                'lcc_edges', nnz(triu(Aco, 1)), ...
                'component_sizes_top10', coSizes(1:min(10, numel(coSizes))), ...
                'node_names_preview', authorNames(coIdx(1:min(5, numel(coIdx)))) ...
            );

            fprintf('Building citation graph...\n');
            citationDirected = spones(authorPaper * paperCit * authorPaper');
            citationDirected(1:size(citationDirected, 1) + 1:end) = 0;
            citationUndirected = spones(citationDirected + citationDirected');
            citationUndirected(1:size(citationUndirected, 1) + 1:end) = 0;
            [Aci, ciIdx, ciSizes] = sec81.Common.largestComponent(citationUndirected);

            citation = sec81.Common.baseDatasetStruct();
            citation.name = "statisticians_citation";
            citation.display_name = "Statisticians citation network (No true labels)";
            citation.A = Aci;
            citation.labels = [];
            citation.has_ground_truth = false;
            citation.target_rank = 3;
            citation.cluster_count = 3;
            citation.table_id = "table2d";
            citation.default_prefix = "statisticians_citation";
            citation.changed_prefix = "statisticians_citation_rank5";
            citation.default_outdir = "exp8_1_statisticians_citation_table2_like";
            citation.changed_outdir = "exp8_1_statisticians_citation_rank5_table2_like";
            citation.changed_rank = 5;
            citation.meta = struct( ...
                'raw_authors', size(citationUndirected, 1), ...
                'raw_directed_arcs', nnz(citationDirected), ...
                'raw_undirected_edges', nnz(triu(citationUndirected, 1)), ...
                'lcc_nodes', size(Aci, 1), ...
                'lcc_edges', nnz(triu(Aci, 1)), ...
                'component_sizes_top10', ciSizes(1:min(10, numel(ciSizes))), ...
                'node_names_preview', authorNames(ciIdx(1:min(5, numel(ciIdx)))) ...
            );
        end

        function ds = baseDatasetStruct()
            ds = struct();
            ds.name = "";
            ds.display_name = "";
            ds.A = sparse([]);
            ds.labels = [];
            ds.has_ground_truth = false;
            ds.target_rank = NaN;
            ds.cluster_count = NaN;
            ds.table_id = "";
            ds.default_prefix = "";
            ds.changed_prefix = "";
            ds.default_outdir = "";
            ds.changed_outdir = "";
            ds.changed_rank = NaN;
            ds.meta = struct();
        end

        function [nodeIds, nodeValues, nodeLabels, edgeSources, edgeTargets, selfLoops] = parsePolblogsGml(gmlPath)
            lines = readlines(gmlPath);
            nodeIds = [];
            nodeValues = [];
            nodeLabels = {};
            edgeSources = [];
            edgeTargets = [];
            selfLoops = 0;
            currentType = "";
            currentId = NaN;
            currentValue = NaN;
            currentLabel = "";
            currentSource = NaN;
            currentTarget = NaN;

            for i = 1:numel(lines)
                s = strtrim(lines(i));
                if s == "node ["
                    currentType = "node";
                    currentId = NaN;
                    currentValue = NaN;
                    currentLabel = "";
                    continue;
                elseif s == "edge ["
                    currentType = "edge";
                    currentSource = NaN;
                    currentTarget = NaN;
                    continue;
                elseif s == "]"
                    if currentType == "node"
                        nodeIds(end + 1, 1) = currentId; %#ok<AGROW>
                        nodeValues(end + 1, 1) = currentValue; %#ok<AGROW>
                        nodeLabels{end + 1, 1} = char(currentLabel); %#ok<AGROW>
                    elseif currentType == "edge"
                        edgeSources(end + 1, 1) = currentSource; %#ok<AGROW>
                        edgeTargets(end + 1, 1) = currentTarget; %#ok<AGROW>
                        if currentSource == currentTarget
                            selfLoops = selfLoops + 1;
                        end
                    end
                    currentType = "";
                    continue;
                end

                if currentType == ""
                    continue;
                end
                parts = split(s, " ", 2);
                if numel(parts) < 2
                    continue;
                end
                key = parts(1);
                value = strtrim(parts(2));
                valueText = regexprep(char(value), '^"|"$', '');
                if currentType == "node"
                    if key == "id"
                        currentId = str2double(valueText);
                    elseif key == "value"
                        currentValue = str2double(valueText);
                    elseif key == "label"
                        currentLabel = string(valueText);
                    end
                elseif currentType == "edge"
                    if key == "source"
                        currentSource = str2double(valueText);
                    elseif key == "target"
                        currentTarget = str2double(valueText);
                    end
                end
            end
        end

        function [Aout, idx, sortedSizes] = largestComponent(A)
            A = spones(A);
            A = spones(A + A');
            n = size(A, 1);
            A(1:n + 1:end) = 0;
            G = graph(A);
            bins = conncomp(G);
            counts = accumarray(bins(:), 1);
            [~, gid] = max(counts);
            idx = find(bins(:) == gid);
            Aout = A(idx, idx);
            Aout = spones(Aout);
            Aout(1:size(Aout, 1) + 1:end) = 0;
            sortedSizes = sort(counts, 'descend');
        end

        function y = remapLabels(yRaw)
            vals = unique(yRaw(:), 'stable');
            y = zeros(numel(yRaw), 1);
            for i = 1:numel(vals)
                y(yRaw == vals(i)) = i;
            end
        end

        function pValues = parsePValues(value)
            if isnumeric(value)
                pValues = double(value(:))';
                return;
            end
            parts = split(string(value), ",");
            pValues = str2double(strtrim(parts));
            pValues = pValues(~isnan(pValues))';
        end

        function info = runInfo(ds, embeddingRank)
            info = struct();
            if embeddingRank == ds.target_rank
                info.prefix = char(ds.default_prefix);
                info.outdir_name = char(ds.default_outdir);
                info.rank_label = sprintf('rank%d', ds.target_rank);
            else
                info.prefix = char(ds.changed_prefix);
                info.outdir_name = char(ds.changed_outdir);
                info.rank_label = sprintf('rank%d', embeddingRank);
            end
            info.table_id = char(ds.table_id);
        end

        function methods = methodNames(pValues)
            methods = [{'Random Projection'}, ...
                arrayfun(@(p) sprintf('Random Sampling (p=%g)', p), pValues, 'UniformOutput', false), ...
                {'CountSketch', 'SIGN Bidirectional', 'Non-random'}];
        end

        function methods = summaryMethodNames(pValues, includeNonRandom)
            methods = [{'Random Projection'}, ...
                arrayfun(@(p) sprintf('Random Sampling (p=%g)', p), pValues, 'UniformOutput', false), ...
                {'CountSketch', 'SIGN Bidirectional'}];
            if includeNonRandom
                methods = [methods, {'Non-random'}];
            end
        end

        function [raw, pairwiseRaw] = runExperiment(ds, embeddingRank, reps, seed, q, r, pValues, signK, showProgress)
            master = RandStream('mt19937ar', 'Seed', seed);
            rawRecords = struct([]);
            pairRecords = struct([]);
            allMethods = sec81.Common.methodNames(pValues);
            totalSteps = reps * numel(allMethods);
            doneSteps = 0;
            tGlobal = tic;

            A = ds.A;
            K = ds.cluster_count;
            Kembed = embeddingRank;
            for rep = 1:reps
                repSeed = randi(master, 2147483646, 1, 1);
                labelNames = {};
                labelValues = {};
                timingMap = struct();

                [yNr, timing] = sec81.Common.runMethodEmbedding('Non-random', A, Kembed, K, r, q, 1.0, signK, repSeed + 97);
                labelNames{end + 1} = 'Non-random'; %#ok<AGROW>
                labelValues{end + 1} = yNr; %#ok<AGROW>
                timingMap.Non_random = timing;
                doneSteps = doneSteps + 1;
                if showProgress
                    sec81.Common.printProgress(doneSteps, totalSteps, ds.name, rep, reps, 'Non-random', tGlobal);
                end

                for mi = 1:numel(allMethods)
                    method = allMethods{mi};
                    if strcmp(method, 'Non-random')
                        continue;
                    end
                    if startsWith(method, 'Random Sampling')
                        p = sec81.Common.parsePFromMethod(method);
                    else
                        p = 1.0;
                    end
                    seedOffset = sec81.Common.methodSeedOffset(method, p);
                    [yp, timing] = sec81.Common.runMethodEmbedding(method, A, Kembed, K, r, q, p, signK, repSeed + seedOffset);
                    labelNames{end + 1} = method; %#ok<AGROW>
                    labelValues{end + 1} = yp; %#ok<AGROW>
                    timingMap.(sec81.Common.safeField(method)) = timing;
                    doneSteps = doneSteps + 1;
                    if showProgress
                        sec81.Common.printProgress(doneSteps, totalSteps, ds.name, rep, reps, method, tGlobal);
                    end
                end

                if ds.has_ground_truth
                    yRef = ds.labels;
                    evalMode = "truth";
                else
                    yRef = yNr;
                    evalMode = "relative_to_non_random";
                end

                evalMethods = sec81.Common.summaryMethodNames(pValues, ds.has_ground_truth);
                for mi = 1:numel(evalMethods)
                    method = evalMethods{mi};
                    labelIdx = find(strcmp(labelNames, method), 1);
                    if isempty(labelIdx)
                        continue;
                    end
                    yPred = labelValues{labelIdx};
                    [f1, nmi, ari] = sec81.Common.evaluateClustering(yRef, yPred, K);
                    timing = timingMap.(sec81.Common.safeField(method));

                    rec = struct();
                    rec.dataset = string(ds.name);
                    rec.rep = rep;
                    rec.method = string(method);
                    rec.eval_mode = evalMode;
                    rec.F1 = f1;
                    rec.NMI = nmi;
                    rec.ARI = ari;
                    rec.time_rand_sec = timing.time_rand_sec;
                    rec.time_post_sec = timing.time_post_sec;
                    rec.time_total_sec = timing.time_total_sec;
                    rawRecords = sec81.Common.appendStruct(rawRecords, rec);
                end

                for i = 1:numel(labelNames)
                    for j = (i + 1):numel(labelNames)
                        prec = struct();
                        prec.dataset = string(ds.name);
                        prec.rep = rep;
                        prec.method_i = string(labelNames{i});
                        prec.method_j = string(labelNames{j});
                        prec.ari = sec81.Common.adjustedRandIndex(labelValues{i}, labelValues{j});
                        pairRecords = sec81.Common.appendStruct(pairRecords, prec);
                    end
                end
            end
            if showProgress
                fprintf('\n');
            end
            raw = struct2table(rawRecords);
            pairwiseRaw = struct2table(pairRecords);
        end

        function [labels, timing] = runMethodEmbedding(method, A, Kembed, K, r, q, p, signK, seed)
            rs = RandStream('mt19937ar', 'Seed', sec81.Common.normalizeSeed(seed));
            tTotal = tic;
            timeRand = NaN;
            timePost = NaN;

            switch char(method)
                case 'Non-random'
                    t0 = tic;
                    U = sec81.Common.eigvecsSparseLA(A, Kembed);
                    eigSec = toc(t0);
                    t1 = tic;
                    labels = sec81.Common.kmeansOnRows(U, K, rs, false);
                    kmSec = toc(t1);
                    timeRand = 0.0;
                    timePost = eigSec + kmSec;
                case 'Random Projection'
                    t0 = tic;
                    U = sec81.Common.eigvecsRandomProjection(A, Kembed, r, q, rs);
                    embedSec = toc(t0);
                    t1 = tic;
                    labels = sec81.Common.kmeansOnRows(U, K, rs, false);
                    kmSec = toc(t1);
                    timePost = embedSec + kmSec;
                case 'CountSketch'
                    t0 = tic;
                    U = sec81.Common.eigvecsCountSketch(A, Kembed, r, q, rs);
                    embedSec = toc(t0);
                    t1 = tic;
                    labels = sec81.Common.kmeansOnRows(U, K, rs, false);
                    kmSec = toc(t1);
                    timePost = embedSec + kmSec;
                case 'SIGN Bidirectional'
                    t0 = tic;
                    U = sec81.Common.eigvecsSignBidirectional(A, Kembed, signK, rs);
                    embedSec = toc(t0);
                    t1 = tic;
                    labels = sec81.Common.kmeansOnRows(U, K, rs, false);
                    kmSec = toc(t1);
                    timePost = embedSec + kmSec;
                otherwise
                    if startsWith(method, 'Random Sampling')
                        t0 = tic;
                        [As, sampleSec] = sec81.Common.sampleAdjacency(A, p, rs);
                        U = sec81.Common.eigvecsSparseLA(As, Kembed);
                        eigSec = toc(t0) - sampleSec;
                        t1 = tic;
                        labels = sec81.Common.kmeansOnRows(U, K, rs, false);
                        kmSec = toc(t1);
                        timeRand = sampleSec;
                        timePost = eigSec + kmSec;
                    else
                        error('Unknown method: %s', method);
                    end
            end

            timing = struct();
            timing.time_rand_sec = timeRand;
            timing.time_post_sec = timePost;
            timing.time_total_sec = toc(tTotal);
        end

        function U = eigvecsSparseLA(A, k)
            [~, U] = sec81.Common.topEigpairsSymmetric(A, k, 'la');
        end

        function U = eigvecsRandomProjection(A, k, r, q, rs)
            n = size(A, 1);
            ell = k + r;
            Omega = randn(rs, n, ell);
            Y = Omega;
            for iter = 1:(2 * q + 1)
                Y = A * Y;
            end
            [Q, ~] = qr(Y, 0);
            B = Q' * (A * Q);
            B = 0.5 * (B + B');
            [~, V] = sec81.Common.topEigpairsSymmetric(B, k, 'abs');
            U = Q * V;
        end

        function U = eigvecsCountSketch(A, k, r, q, rs)
            n = size(A, 1);
            ell = k + r;
            h = randi(rs, ell, n, 1);
            signs = 2 * double(rand(rs, n, 1) >= 0.5) - 1;
            STranspose = sparse((1:n)', h, signs, n, ell);
            Y = full(A * STranspose);
            for iter = 1:(2 * q)
                Y = A * Y;
            end
            [Q, ~] = qr(Y, 0);
            B = Q' * (A * Q);
            B = 0.5 * (B + B');
            [~, V] = sec81.Common.topEigpairsSymmetric(B, k, 'abs');
            U = Q * V;
        end

        function U = eigvecsSignBidirectional(A, k, signK, rs)
            n = size(A, 1);
            Qprev = randn(rs, n, k);
            Qk = Qprev;
            for iter = 1:signK
                [Qtilde, ~] = qr(A' * Qprev, 0);
                [Qk, ~] = qr(A * Qtilde, 0);
                Qprev = Qk;
            end
            B = Qk' * (A * Qk);
            B = 0.5 * (B + B');
            [~, V] = sec81.Common.topEigpairsSymmetric(B, k, 'abs');
            U = Qk * V;
        end

        function [As, sampleSec] = sampleAdjacency(A, p, rs)
            n = size(A, 1);
            [rows, cols, vals] = find(triu(A, 1));
            t0 = tic;
            keep = rand(rs, numel(vals), 1) < p;
            sampleSec = toc(t0);
            rr = rows(keep);
            cc = cols(keep);
            vv = vals(keep) ./ p;
            As = sparse([rr; cc], [cc; rr], [vv; vv], n, n);
            As = spones(As) .* As;
            As(1:n + 1:end) = 0;
        end

        function [vals, vecs] = topEigpairsSymmetric(M, k, mode)
            M = 0.5 * (M + M');
            n = size(M, 1);
            useFull = n <= max(60, k + 5) || k >= n;
            if useFull
                [V, D] = eig(full(M), 'vector');
                vals = real(D(:));
                vecs = real(V);
            else
                opts.tol = 1e-8;
                opts.maxit = 1000;
                try
                    if strcmp(mode, 'abs')
                        [V, D] = eigs(M, k, 'largestabs', opts);
                    else
                        [V, D] = eigs(M, k, 'largestreal', opts);
                    end
                    vals = real(diag(D));
                    vecs = real(V);
                catch
                    [V, D] = eig(full(M), 'vector');
                    vals = real(D(:));
                    vecs = real(V);
                end
            end
            if strcmp(mode, 'abs')
                [~, order] = sort(abs(vals), 'descend');
            else
                [~, order] = sort(vals, 'descend');
            end
            order = order(1:k);
            vals = vals(order);
            vecs = vecs(:, order);
        end

        function labels = kmeansOnRows(U, K, rs, normalizeRows)
            if normalizeRows
                U = sec81.Common.normalizeRowsL2(U);
            end
            X = real(double(U));
            n = size(X, 1);
            bestObjective = inf;
            bestLabels = ones(n, 1);
            nStarts = 20;
            maxIter = 100;
            for start = 1:nStarts
                centers = sec81.Common.initKMeansPlusPlus(X, K, rs);
                labels = zeros(n, 1);
                for iter = 1:maxIter
                    oldLabels = labels;
                    dist = sec81.Common.pairwiseSquaredDistances(X, centers);
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
                dist = sec81.Common.pairwiseSquaredDistances(X, centers);
                objective = sum(min(dist, [], 2));
                if objective < bestObjective
                    bestObjective = objective;
                    bestLabels = labels;
                end
            end
            labels = bestLabels;
        end

        function Xn = normalizeRowsL2(X)
            norms = sqrt(sum(X .^ 2, 2));
            Xn = zeros(size(X));
            idx = norms > 1e-12;
            Xn(idx, :) = X(idx, :) ./ norms(idx);
        end

        function centers = initKMeansPlusPlus(X, K, rs)
            n = size(X, 1);
            centers = zeros(K, size(X, 2));
            first = randi(rs, n);
            centers(1, :) = X(first, :);
            minDist = sec81.Common.pairwiseSquaredDistances(X, centers(1, :));
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
                distNew = sec81.Common.pairwiseSquaredDistances(X, centers(c, :));
                minDist = min(minDist, distNew);
            end
        end

        function dist = pairwiseSquaredDistances(X, C)
            x2 = sum(X .^ 2, 2);
            c2 = sum(C .^ 2, 2)';
            dist = x2 + c2 - 2 * (X * C');
            dist = max(dist, 0);
        end

        function [f1, nmi, ari] = evaluateClustering(yTrue, yPred, K)
            yTrue = double(yTrue(:));
            yPred = double(yPred(:));
            yAligned = sec81.Common.alignLabelsHungarian(yTrue, yPred, K);
            f1 = sec81.Common.macroF1(yTrue, yAligned, K);
            nmi = sec81.Common.normalizedMutualInfo(yTrue, yPred);
            ari = sec81.Common.adjustedRandIndex(yTrue, yPred);
        end

        function yAligned = alignLabelsHungarian(yTrue, yPred, K)
            conf = zeros(K, K);
            for i = 1:numel(yTrue)
                if yTrue(i) >= 1 && yTrue(i) <= K && yPred(i) >= 1 && yPred(i) <= K
                    conf(yTrue(i), yPred(i)) = conf(yTrue(i), yPred(i)) + 1;
                end
            end
            nClass = sum(conf, 2);
            score = zeros(K, K);
            for i = 1:K
                if nClass(i) > 0
                    score(i, :) = conf(i, :) ./ nClass(i);
                end
            end
            assignment = sec81.Common.hungarianMaximize(score);
            predToTrue = zeros(K, 1);
            for trueClass = 1:K
                predClass = assignment(trueClass);
                if predClass >= 1 && predClass <= K
                    predToTrue(predClass) = trueClass;
                end
            end
            for c = 1:K
                if predToTrue(c) == 0
                    predToTrue(c) = c;
                end
            end
            yAligned = predToTrue(yPred);
        end

        function assignment = hungarianMaximize(score)
            cost = max(score(:)) - score;
            n = size(cost, 1);
            cost = cost - min(cost, [], 2);
            cost = cost - min(cost, [], 1);
            star = false(n, n);
            prime = false(n, n);
            rowCover = false(n, 1);
            colCover = false(1, n);
            epsVal = 1e-12;

            for i = 1:n
                for j = 1:n
                    if abs(cost(i, j)) <= epsVal && ~rowCover(i) && ~colCover(j)
                        star(i, j) = true;
                        rowCover(i) = true;
                        colCover(j) = true;
                    end
                end
            end
            rowCover(:) = false;
            colCover(:) = any(star, 1);

            while sum(colCover) < n
                [row, col] = sec81.Common.findUncoveredZero(cost, rowCover, colCover, epsVal);
                while row == 0
                    uncovered = cost(~rowCover, ~colCover);
                    if isempty(uncovered)
                        minVal = 0;
                    else
                        minVal = min(uncovered, [], 'all');
                    end
                    cost(rowCover, :) = cost(rowCover, :) + minVal;
                    cost(:, ~colCover) = cost(:, ~colCover) - minVal;
                    [row, col] = sec81.Common.findUncoveredZero(cost, rowCover, colCover, epsVal);
                end
                prime(row, col) = true;
                starCol = find(star(row, :), 1);
                if isempty(starCol)
                    star = sec81.Common.augmentPath(star, prime, row, col);
                    prime(:, :) = false;
                    rowCover(:) = false;
                    colCover(:) = any(star, 1);
                else
                    rowCover(row) = true;
                    colCover(starCol) = false;
                end
            end

            assignment = zeros(n, 1);
            for i = 1:n
                j = find(star(i, :), 1);
                if ~isempty(j)
                    assignment(i) = j;
                end
            end
        end

        function [row, col] = findUncoveredZero(cost, rowCover, colCover, epsVal)
            row = 0;
            col = 0;
            for i = 1:size(cost, 1)
                if rowCover(i)
                    continue;
                end
                for j = 1:size(cost, 2)
                    if ~colCover(j) && abs(cost(i, j)) <= epsVal
                        row = i;
                        col = j;
                        return;
                    end
                end
            end
        end

        function star = augmentPath(star, prime, row, col)
            path = [row, col];
            done = false;
            while ~done
                starRow = find(star(:, path(end, 2)), 1);
                if isempty(starRow)
                    done = true;
                else
                    path(end + 1, :) = [starRow, path(end, 2)]; %#ok<AGROW>
                    primeCol = find(prime(path(end, 1), :), 1);
                    path(end + 1, :) = [path(end, 1), primeCol]; %#ok<AGROW>
                end
            end
            for i = 1:size(path, 1)
                r = path(i, 1);
                c = path(i, 2);
                star(r, c) = ~star(r, c);
            end
        end

        function f1 = macroF1(yTrue, yPred, K)
            vals = zeros(K, 1);
            for c = 1:K
                tp = sum(yTrue == c & yPred == c);
                fp = sum(yTrue ~= c & yPred == c);
                fn = sum(yTrue == c & yPred ~= c);
                denom = 2 * tp + fp + fn;
                if denom == 0
                    vals(c) = 0;
                else
                    vals(c) = 2 * tp / denom;
                end
            end
            f1 = mean(vals);
        end

        function nmi = normalizedMutualInfo(yA, yB)
            yA = sec81.Common.remapLabels(yA);
            yB = sec81.Common.remapLabels(yB);
            n = numel(yA);
            ka = max(yA);
            kb = max(yB);
            cont = zeros(ka, kb);
            for i = 1:n
                cont(yA(i), yB(i)) = cont(yA(i), yB(i)) + 1;
            end
            ai = sum(cont, 2);
            bj = sum(cont, 1);
            mi = 0.0;
            for i = 1:ka
                for j = 1:kb
                    nij = cont(i, j);
                    if nij > 0
                        mi = mi + (nij / n) * log((n * nij) / (ai(i) * bj(j)));
                    end
                end
            end
            ha = -sum((ai(ai > 0) / n) .* log(ai(ai > 0) / n));
            hb = -sum((bj(bj > 0) / n) .* log(bj(bj > 0) / n));
            denom = (ha + hb) / 2;
            if denom <= 0
                nmi = 1.0;
            else
                nmi = mi / denom;
            end
        end

        function ari = adjustedRandIndex(yA, yB)
            yA = sec81.Common.remapLabels(yA);
            yB = sec81.Common.remapLabels(yB);
            n = numel(yA);
            ka = max(yA);
            kb = max(yB);
            cont = zeros(ka, kb);
            for i = 1:n
                cont(yA(i), yB(i)) = cont(yA(i), yB(i)) + 1;
            end
            sumComb = sum(sec81.Common.choose2(cont), 'all');
            rowComb = sum(sec81.Common.choose2(sum(cont, 2)));
            colComb = sum(sec81.Common.choose2(sum(cont, 1)));
            totalComb = n * (n - 1) / 2;
            if totalComb == 0
                ari = 1.0;
                return;
            end
            expected = rowComb * colComb / totalComb;
            maxIndex = 0.5 * (rowComb + colComb);
            denom = maxIndex - expected;
            if abs(denom) < 1e-15
                ari = 1.0;
            else
                ari = (sumComb - expected) / denom;
            end
        end

        function c = choose2(x)
            c = x .* (x - 1) / 2;
        end

        function summary = summarizeRaw(raw, pValues, includeNonRandom)
            methods = sec81.Common.summaryMethodNames(pValues, includeNonRandom);
            rows = struct([]);
            for i = 1:numel(methods)
                method = methods{i};
                mask = strcmp(string(raw.method), method);
                if ~any(mask)
                    continue;
                end
                rec = struct();
                rec.dataset = raw.dataset(find(mask, 1));
                rec.method = string(method);
                rec.F1_mean = mean(raw.F1(mask));
                rec.F1_std = sec81.Common.sampleStd(raw.F1(mask));
                rec.NMI_mean = mean(raw.NMI(mask));
                rec.NMI_std = sec81.Common.sampleStd(raw.NMI(mask));
                rec.ARI_mean = mean(raw.ARI(mask));
                rec.ARI_std = sec81.Common.sampleStd(raw.ARI(mask));
                rec.time_rand_mean = mean(raw.time_rand_sec(mask), 'omitnan');
                rec.time_rand_std = sec81.Common.sampleStd(raw.time_rand_sec(mask));
                rec.time_post_mean = mean(raw.time_post_sec(mask), 'omitnan');
                rec.time_post_std = sec81.Common.sampleStd(raw.time_post_sec(mask));
                rec.time_total_mean = mean(raw.time_total_sec(mask), 'omitnan');
                rec.time_total_std = sec81.Common.sampleStd(raw.time_total_sec(mask));
                rows = sec81.Common.appendStruct(rows, rec);
            end
            summary = struct2table(rows);
        end

        function s = sampleStd(x)
            x = x(~isnan(x));
            if numel(x) <= 1
                s = 0.0;
            else
                s = std(x, 0);
            end
        end

        function tbl = buildTable2Like(summary, pValues, includeNonRandom)
            methods = sec81.Common.summaryMethodNames(pValues, includeNonRandom);
            dispMethods = strings(0, 1);
            f1 = strings(0, 1);
            nmi = strings(0, 1);
            ari = strings(0, 1);
            for i = 1:numel(methods)
                method = methods{i};
                idx = find(strcmp(string(summary.method), method), 1);
                if isempty(idx)
                    continue;
                end
                dispMethods(end + 1, 1) = sec81.Common.displayMethod(method); %#ok<AGROW>
                f1(end + 1, 1) = sec81.Common.formatMeanStd(summary.F1_mean(idx), summary.F1_std(idx)); %#ok<AGROW>
                nmi(end + 1, 1) = sec81.Common.formatMeanStd(summary.NMI_mean(idx), summary.NMI_std(idx)); %#ok<AGROW>
                ari(end + 1, 1) = sec81.Common.formatMeanStd(summary.ARI_mean(idx), summary.ARI_std(idx)); %#ok<AGROW>
            end
            tbl = table(dispMethods, f1, nmi, ari, 'VariableNames', {'Methods', 'F 1', 'NMI', 'ARI'});
        end

        function s = displayMethod(method)
            method = char(method);
            if strcmp(method, 'Non-random')
                s = "Non-Random";
            elseif startsWith(method, 'Random Sampling')
                p = sec81.Common.parsePFromMethod(method);
                s = string(sprintf('Random Sampling (p= %.1f)', p));
            else
                s = string(method);
            end
        end

        function s = formatMeanStd(mu, sigma)
            if isnan(sigma)
                sigma = 0.0;
            end
            s = string(sprintf('%.3f(%.3f)', mu, sigma));
        end

        function writeTable2Markdown(tbl, outMd, runInfo, ds, reps)
            lines = strings(0, 1);
            lines(end + 1) = sprintf('%s: The clustering performance on the %s.', runInfo.table_id, ds.display_name);
            lines(end + 1) = "";
            lines(end + 1) = "| Methods | F 1 | NMI | ARI |";
            lines(end + 1) = "|---|---:|---:|---:|";
            for i = 1:height(tbl)
                lines(end + 1) = sprintf('| %s | %s | %s | %s |', ...
                    tbl.Methods(i), tbl.("F 1")(i), tbl.NMI(i), tbl.ARI(i));
            end
            lines(end + 1) = "";
            lines(end + 1) = sprintf('Note: Values are mean(std) over %d MATLAB replications.', reps);
            if ~ds.has_ground_truth
                lines(end + 1) = "For this dataset, scores are relative to non-random spectral clustering.";
            end
            sec81.Common.writeLines(outMd, lines);
        end

        function pairMat = pairwiseMeanMatrix(pairwiseRaw)
            methods = unique([string(pairwiseRaw.method_i); string(pairwiseRaw.method_j)], 'stable');
            m = numel(methods);
            mat = eye(m);
            for i = 1:m
                for j = (i + 1):m
                    mask = (pairwiseRaw.method_i == methods(i) & pairwiseRaw.method_j == methods(j)) | ...
                        (pairwiseRaw.method_i == methods(j) & pairwiseRaw.method_j == methods(i));
                    if any(mask)
                        mat(i, j) = mean(pairwiseRaw.ari(mask));
                        mat(j, i) = mat(i, j);
                    end
                end
            end
            pairMat = struct('methods', methods, 'matrix', mat);
        end

        function writeMatrixCsv(methods, mat, outCsv)
            fid = fopen(outCsv, 'w');
            cleanup = onCleanup(@() fclose(fid));
            fprintf(fid, 'method');
            for j = 1:numel(methods)
                fprintf(fid, ',%s', methods(j));
            end
            fprintf(fid, '\n');
            for i = 1:numel(methods)
                fprintf(fid, '%s', methods(i));
                for j = 1:numel(methods)
                    fprintf(fid, ',%.12g', mat(i, j));
                end
                fprintf(fid, '\n');
            end
            clear cleanup;
        end

        function plotPairwiseHeatmap(methods, mat, outPng)
            fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100, 100, 900, 760]);
            imagesc(mat, [-0.1, 1.0]);
            axis equal tight;
            colormap(parula);
            colorbar;
            xticks(1:numel(methods));
            yticks(1:numel(methods));
            xticklabels(methods);
            yticklabels(methods);
            xtickangle(30);
            title('Pairwise ARI (mean across replications)');
            for i = 1:size(mat, 1)
                for j = 1:size(mat, 2)
                    if mat(i, j) < 0.55
                        color = 'w';
                    else
                        color = 'k';
                    end
                    text(j, i, sprintf('%.2f', mat(i, j)), 'HorizontalAlignment', 'center', ...
                        'VerticalAlignment', 'middle', 'Color', color, 'FontSize', 8);
                end
            end
            exportgraphics(fig, outPng, 'Resolution', 180);
            close(fig);
        end

        function writeMetaJson(outJson, ds, embeddingRank, params, pValues, runInfo)
            meta = struct();
            meta.dataset = char(ds.name);
            meta.display_name = char(ds.display_name);
            meta.nodes = size(ds.A, 1);
            meta.edges = nnz(triu(ds.A, 1));
            meta.target_rank = ds.target_rank;
            meta.embedding_rank = embeddingRank;
            meta.cluster_count = ds.cluster_count;
            meta.reps = params.reps;
            meta.seed = params.seed;
            meta.q = params.q;
            meta.r = params.r;
            meta.p_values = pValues;
            meta.sign_k = params.sign_k;
            meta.table_id = runInfo.table_id;
            meta.source_meta = ds.meta;
            txt = jsonencode(meta, 'PrettyPrint', true);
            fid = fopen(outJson, 'w');
            cleanup = onCleanup(@() fclose(fid));
            fwrite(fid, txt);
            fprintf(fid, '\n');
            clear cleanup;
        end

        function writeRankComparison(outputs, outCsv, outMd)
            outDir = fileparts(outCsv);
            if ~exist(outDir, 'dir')
                mkdir(outDir);
            end
            pairs = {
                'email_rank42', 'email_rank30';
                'political_blog_rank2', 'political_blog_rank5';
                'statisticians_coauthor_rank3', 'statisticians_coauthor_rank5';
                'statisticians_citation_rank3', 'statisticians_citation_rank5'
            };
            rows = struct([]);
            pythonCsv = fullfile(fileparts(fileparts(outDir)), 'reference_1_section8_1', 'results', 'section8_1_table2_rank_comparison.csv');
            pyTable = table();
            if exist(pythonCsv, 'file')
                pyTable = readtable(pythonCsv, 'TextType', 'string');
            end

            for pi = 1:size(pairs, 1)
                base = outputs.(pairs{pi, 1});
                changed = outputs.(pairs{pi, 2});
                methods = unique([string(base.summary.method); string(changed.summary.method)], 'stable');
                for mi = 1:numel(methods)
                    method = methods(mi);
                    rec = struct();
                    rec.dataset = string(base.display_name);
                    rec.method = method;
                    rec.paper_rank = base.paper_rank;
                    rec.changed_rank = changed.embedding_rank;
                    [paperF1, paperNMI, paperARI, pyBaseF1, pyBaseNMI, pyBaseARI, pyChgF1, pyChgNMI, pyChgARI] = ...
                        sec81.Common.lookupPythonComparison(pyTable, rec.dataset, method, base.has_ground_truth);
                    rec.paper_F1 = paperF1;
                    rec.paper_NMI = paperNMI;
                    rec.paper_ARI = paperARI;
                    rec.python_paper_rank_F1 = pyBaseF1;
                    rec.python_paper_rank_NMI = pyBaseNMI;
                    rec.python_paper_rank_ARI = pyBaseARI;
                    rec.python_changed_rank_F1 = pyChgF1;
                    rec.python_changed_rank_NMI = pyChgNMI;
                    rec.python_changed_rank_ARI = pyChgARI;
                    rec.matlab_paper_rank_F1 = sec81.Common.summaryMetric(base.summary, method, 'F1');
                    rec.matlab_paper_rank_NMI = sec81.Common.summaryMetric(base.summary, method, 'NMI');
                    rec.matlab_paper_rank_ARI = sec81.Common.summaryMetric(base.summary, method, 'ARI');
                    rec.matlab_changed_rank_F1 = sec81.Common.summaryMetric(changed.summary, method, 'F1');
                    rec.matlab_changed_rank_NMI = sec81.Common.summaryMetric(changed.summary, method, 'NMI');
                    rec.matlab_changed_rank_ARI = sec81.Common.summaryMetric(changed.summary, method, 'ARI');
                    rows = sec81.Common.appendStruct(rows, rec);
                end
            end
            tbl = struct2table(rows);
            writetable(tbl, outCsv);
            sec81.Common.writeComparisonMarkdown(tbl, outMd);
        end

        function [paperF1, paperNMI, paperARI, pyBaseF1, pyBaseNMI, pyBaseARI, pyChgF1, pyChgNMI, pyChgARI] = ...
                lookupPythonComparison(pyTable, dataset, method, hasGroundTruth)
            blanks = repmat("", 1, 9);
            paperF1 = blanks(1); paperNMI = blanks(2); paperARI = blanks(3);
            pyBaseF1 = blanks(4); pyBaseNMI = blanks(5); pyBaseARI = blanks(6);
            pyChgF1 = blanks(7); pyChgNMI = blanks(8); pyChgARI = blanks(9);
            if isempty(pyTable) || ~any(strcmp(method, ["Random Projection", "Random Sampling (p=0.7)", "Random Sampling (p=0.8)", "Non-random"]))
                return;
            end
            pyDataset = dataset;
            if ~hasGroundTruth && ~contains(pyDataset, "(No true labels)")
                pyDataset = pyDataset + " (No true labels)";
            end
            pyMethod = method;
            if ~hasGroundTruth
                if method == "Random Projection"
                    pyMethod = "Random Projection (relative)";
                elseif startsWith(method, "Random Sampling")
                    p = sec81.Common.parsePFromMethod(method);
                    pyMethod = string(sprintf('Random Sampling (relative) (p=%.1f)', p));
                else
                    return;
                end
            end
            mask = pyTable.dataset == pyDataset & pyTable.method == pyMethod;
            if ~any(mask)
                return;
            end
            row = pyTable(find(mask, 1), :);
            paperF1 = row.paper_F1;
            paperNMI = row.paper_NMI;
            paperARI = row.paper_ARI;
            pyBaseF1 = row.reproduction_paper_rank_F1;
            pyBaseNMI = row.reproduction_paper_rank_NMI;
            pyBaseARI = row.reproduction_paper_rank_ARI;
            pyChgF1 = row.reproduction_changed_rank_F1;
            pyChgNMI = row.reproduction_changed_rank_NMI;
            pyChgARI = row.reproduction_changed_rank_ARI;
        end

        function val = summaryMetric(summary, method, metric)
            idx = find(string(summary.method) == string(method), 1);
            if isempty(idx)
                val = "";
                return;
            end
            meanName = sprintf('%s_mean', metric);
            stdName = sprintf('%s_std', metric);
            val = sec81.Common.formatMeanStd(summary.(meanName)(idx), summary.(stdName)(idx));
        end

        function writeComparisonMarkdown(tbl, outMd)
            lines = strings(0, 1);
            lines(end + 1) = "# Section 8.1 MATLAB rank comparison";
            lines(end + 1) = "";
            for di = 1:numel(unique(tbl.dataset, 'stable'))
                datasets = unique(tbl.dataset, 'stable');
                dname = datasets(di);
                d = tbl(tbl.dataset == dname, :);
                lines(end + 1) = sprintf("## %s", dname);
                lines(end + 1) = "";
                lines(end + 1) = "| Method | Paper F1 | Python rank F1 | MATLAB rank F1 | MATLAB changed F1 | MATLAB rank NMI | MATLAB changed NMI | MATLAB rank ARI | MATLAB changed ARI |";
                lines(end + 1) = "|---|---:|---:|---:|---:|---:|---:|---:|---:|";
                for i = 1:height(d)
                    lines(end + 1) = sprintf('| %s | %s | %s | %s | %s | %s | %s | %s | %s |', ...
                        d.method(i), d.paper_F1(i), d.python_paper_rank_F1(i), ...
                        d.matlab_paper_rank_F1(i), d.matlab_changed_rank_F1(i), ...
                        d.matlab_paper_rank_NMI(i), d.matlab_changed_rank_NMI(i), ...
                        d.matlab_paper_rank_ARI(i), d.matlab_changed_rank_ARI(i));
                end
                lines(end + 1) = "";
            end
            sec81.Common.writeLines(outMd, lines);
        end

        function writeReport(comparisonCsv, outMd)
            tbl = readtable(comparisonCsv, 'TextType', 'string');
            lines = strings(0, 1);
            lines(end + 1) = "# Reference 1 Section 8.1 MATLAB 실험 보고서";
            lines(end + 1) = "";
            lines(end + 1) = "이 보고서는 기존 Section 8.1 real network accuracy experiment를 MATLAB로 다시 실행한 결과를 정리한다. 기존 방법인 Random Projection, Random Sampling, Non-random에 CountSketch와 Wang et al. (2025)의 SIGN 양방향 Nyström subspace iteration embedding을 추가했다.";
            lines(end + 1) = "";
            lines(end + 1) = "## 1. 실험 설정";
            lines(end + 1) = "";
            lines(end + 1) = "- 반복 횟수는 20회 기본값이며, 이번 실행의 결과 CSV에는 평균과 표준편차를 `mean(std)` 형식으로 기록했다.";
            lines(end + 1) = "- Random Projection과 CountSketch는 `q=2`, oversampling `r=10`을 사용했다.";
            lines(end + 1) = "- Random Sampling은 `p=0.7`, `p=0.8` 두 확률을 사용했다.";
            lines(end + 1) = "- SIGN Bidirectional은 첨부 논문의 SIGN 구조처럼 `A'`와 `A`를 번갈아 곱하고 QR로 양방향 subspace를 갱신한 뒤, 얻어진 left subspace에서 spectral clustering을 수행했다.";
            lines(end + 1) = "- European email과 Political blog는 ground-truth label 기준으로 평가했고, 두 statisticians 네트워크는 기존 8.1과 같이 Non-random 결과를 reference label로 둔 relative score를 사용했다.";
            lines(end + 1) = "";
            lines(end + 1) = "## 2. 방법 설명";
            lines(end + 1) = "";
            lines(end + 1) = "| 방법 | 의미 |";
            lines(end + 1) = "|---|---|";
            lines(end + 1) = "| Non-random | 원래 adjacency matrix에서 `eigs`로 leading eigenvectors를 직접 구하는 기준 방법 |";
            lines(end + 1) = "| Random Projection | Gaussian sketch와 power iteration으로 spectral subspace를 근사 |";
            lines(end + 1) = "| Random Sampling | edge를 확률 `p`로 남기고 `1/p`로 rescale한 sampled adjacency에서 eigenvectors 계산 |";
            lines(end + 1) = "| CountSketch | Gaussian sketch 대신 hash bucket과 sign으로 만든 sparse embedding 사용 |";
            lines(end + 1) = "| SIGN Bidirectional | Wang et al. (2025)의 generalized Nyström with subspace iteration 구조를 대칭 adjacency embedding에 적용 |";
            lines(end + 1) = "";
            lines(end + 1) = "## 3. MATLAB 결과 요약";
            lines(end + 1) = "";
            datasets = unique(tbl.dataset, 'stable');
            for di = 1:numel(datasets)
                dname = datasets(di);
                d = tbl(tbl.dataset == dname, :);
                lines(end + 1) = sprintf("### %s", dname);
                lines(end + 1) = "";
                lines(end + 1) = "| Method | MATLAB paper-rank F1 | MATLAB changed-rank F1 | MATLAB paper-rank NMI | MATLAB changed-rank NMI | MATLAB paper-rank ARI | MATLAB changed-rank ARI |";
                lines(end + 1) = "|---|---:|---:|---:|---:|---:|---:|";
                for i = 1:height(d)
                    lines(end + 1) = sprintf('| %s | %s | %s | %s | %s | %s | %s |', ...
                        d.method(i), d.matlab_paper_rank_F1(i), d.matlab_changed_rank_F1(i), ...
                        d.matlab_paper_rank_NMI(i), d.matlab_changed_rank_NMI(i), ...
                        d.matlab_paper_rank_ARI(i), d.matlab_changed_rank_ARI(i));
                end
                lines(end + 1) = "";
            end
            lines(end + 1) = "### 주요 관찰";
            lines(end + 1) = "";
            lines(end + 1) = "- European email에서는 rank 42에서 rank 30으로 줄였을 때 대부분의 MATLAB 지표가 유지되거나 올라갔다. CountSketch는 Random Projection과 거의 같은 수준의 F1/NMI/ARI를 냈고, SIGN Bidirectional도 비슷한 범위에 머물렀다.";
            lines(end + 1) = "- Political blog는 기존 Python 재현과 마찬가지로 rank 2가 자연스럽고, rank 5에서는 Random Projection과 CountSketch의 ARI가 거의 사라졌다. SIGN은 rank 5에서 Random Projection/CountSketch보다 덜 무너졌지만 rank 2보다 좋지는 않았다.";
            lines(end + 1) = "- Statisticians coauthor와 citation에서는 CountSketch가 paper-rank 설정에서 Non-random reference를 매우 잘 따라갔다. 반면 SIGN Bidirectional은 이번 spectral clustering embedding 방식에서는 두 statisticians 네트워크에서 상대 점수가 낮아, low-rank approximation 품질과 clustering label 재현성이 항상 같은 방향은 아니라는 점을 보여준다.";
            lines(end + 1) = "- Rank 5 변경은 두 statisticians 네트워크에서 전반적으로 relative score를 낮췄다. 이는 기존 Python 보고서의 결론처럼, cluster count보다 큰 embedding rank가 추가 signal을 주기보다 noise 방향을 KMeans에 넣을 수 있다는 해석과 맞다.";
            lines(end + 1) = "";
            lines(end + 1) = "## 4. 해석";
            lines(end + 1) = "";
            lines(end + 1) = "MATLAB 결과는 Python 결과와 완전히 같은 숫자를 목표로 하지 않는다. `eigs`/ARPACK 호출, QR 부호, KMeans++ 초기화, 난수 생성기가 서로 다르기 때문이다. 따라서 해석은 절대값 하나보다 방법 간 상대적 패턴, rank 변경에 따른 변화, pairwise ARI 안정성을 중심으로 보는 것이 맞다.";
            lines(end + 1) = "";
            lines(end + 1) = "CountSketch는 Gaussian projection보다 sketch 자체가 훨씬 희소하므로 큰 행렬에서는 메모리 이점이 있다. 다만 embedding dimension이 작고 QR/eigs가 뒤따르기 때문에 전체 시간은 데이터셋의 sparsity와 KMeans 반복에 따라 달라진다. SIGN Bidirectional은 한 번의 random sketch에서 시작하지만 매 iteration마다 `A'` 방향과 `A` 방향을 모두 갱신하므로 Random Projection보다 QR 단계가 더 자주 들어간다. 대신 양방향 subspace를 같이 정렬한다는 점이 Wang et al. (2025)의 핵심이다.";
            lines(end + 1) = "";
            lines(end + 1) = "## 5. 산출물";
            lines(end + 1) = "";
            lines(end + 1) = "- `results/section8_1_matlab_rank_comparison.csv`: 논문 rank와 변경 rank의 MATLAB 종합 비교";
            lines(end + 1) = "- `results/section8_1_matlab_rank_comparison.md`: 위 비교표의 Markdown 버전";
            lines(end + 1) = "- 각 실험별 `*_raw_per_rep.csv`, `*_summary_mean_std.csv`, `*_table2*_like.md`, `*_pairwise_ari_mean_matrix.csv`, `*_pairwise_ari_heatmap.png`";
            sec81.Common.writeLines(outMd, lines);
        end

        function p = parsePFromMethod(method)
            token = regexp(char(method), 'p=([0-9.]+)', 'tokens', 'once');
            if isempty(token)
                p = NaN;
            else
                p = str2double(token{1});
            end
        end

        function offset = methodSeedOffset(method, p)
            if strcmp(method, 'Random Projection')
                offset = 11;
            elseif strcmp(method, 'CountSketch')
                offset = 47;
            elseif strcmp(method, 'SIGN Bidirectional')
                offset = 73;
            elseif startsWith(method, 'Random Sampling')
                offset = 31 + round(p * 1000);
            else
                offset = 97;
            end
        end

        function name = safeField(method)
            name = matlab.lang.makeValidName(char(method));
        end

        function seed = normalizeSeed(seed)
            seed = mod(round(double(seed)), 2147483646);
            if seed <= 0
                seed = seed + 2147483646;
            end
        end

        function records = appendStruct(records, record)
            if isempty(records)
                records = record;
            else
                records(end + 1) = record; %#ok<AGROW>
            end
        end

        function printProgress(doneSteps, totalSteps, dataset, rep, reps, method, tGlobal)
            elapsed = toc(tGlobal);
            ratio = doneSteps / max(1, totalSteps);
            if doneSteps > 0
                eta = elapsed * (1 - ratio) / max(ratio, 1e-12);
            else
                eta = 0;
            end
            fprintf('\r[%4d/%4d %5.1f%%] dataset=%s rep=%02d/%02d method=%s elapsed=%s eta=%s', ...
                doneSteps, totalSteps, 100 * ratio, string(dataset), rep, reps, method, ...
                sec81.Common.formatSeconds(elapsed), sec81.Common.formatSeconds(eta));
        end

        function s = formatSeconds(sec)
            sec = max(0, sec);
            h = floor(sec / 3600);
            m = floor(mod(sec, 3600) / 60);
            ss = floor(mod(sec, 60));
            if h > 0
                s = sprintf('%02d:%02d:%02d', h, m, ss);
            else
                s = sprintf('%02d:%02d', m, ss);
            end
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
