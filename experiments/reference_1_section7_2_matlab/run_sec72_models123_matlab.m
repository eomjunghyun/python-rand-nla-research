function outputs = run_sec72_models123_matlab(varargin)
%RUN_SEC72_MODELS123_MATLAB Reproduce Section 7.2 Models 1-3 in MATLAB.

thisDir = fileparts(mfilename('fullpath'));
addpath(thisDir);

p = inputParser;
addParameter(p, 'reps', 20);
addParameter(p, 'seed', 2026);
addParameter(p, 'n_values', [200, 400, 600, 800, 1000, 1200]);
addParameter(p, 'outdir', fullfile(thisDir, 'results', 'exp72_models123_paper_aligned_live'));
addParameter(p, 'theta_mode', 'exact');
addParameter(p, 'detailed_timing', false);
addParameter(p, 'no_plot', false);
addParameter(p, 'no_progress', false);
parse(p, varargin{:});

cfg = struct();
cfg.n_values = sec72.Common.parseNValues(p.Results.n_values);
cfg.model_ids = [1, 2, 3];
cfg.K = 3;
cfg.K_prime_fullrank = 3;
cfg.K_prime_rankdef = 2;
cfg.q = 2;
cfg.r = 10;
cfg.p = 0.7;
cfg.reps = p.Results.reps;
cfg.seed = p.Results.seed;

outdir = char(p.Results.outdir);
if ~exist(outdir, 'dir')
    mkdir(outdir);
end

fprintf('Running Section 7.2 Models 1-3 in MATLAB...\n');
raw = sec72.Common.runExperimentModels123( ...
    cfg, ~p.Results.no_progress, p.Results.theta_mode, p.Results.detailed_timing);
summary = sec72.Common.summarizeMetrics(raw);

rawCsv = fullfile(outdir, 'sec72_models123_raw_per_rep.csv');
summaryCsv = fullfile(outdir, 'sec72_models123_summary_mean_std.csv');
writetable(raw, rawCsv);
writetable(summary, summaryCsv);

timingRawCsv = '';
timingSummaryCsv = '';
if p.Results.detailed_timing
    timingRaw = sec72.Common.extractTimingBreakdown(raw, {'model', 'n', 'rep', 'method'});
    timingSummary = sec72.Common.summarizeTimingBreakdown(timingRaw, {'model', 'n'});
    timingRawCsv = fullfile(outdir, 'sec72_models123_timing_breakdown_raw.csv');
    timingSummaryCsv = fullfile(outdir, 'sec72_models123_timing_breakdown_summary.csv');
    writetable(timingRaw, timingRawCsv);
    writetable(timingSummary, timingSummaryCsv);
end

metricsPng = fullfile(outdir, 'sec72_models123_metrics_figure5_like.png');
runtimePng = fullfile(outdir, 'sec72_models123_runtime.png');
if ~p.Results.no_plot
    sec72.Common.plotModels123Metrics(summary, metricsPng);
    sec72.Common.plotRuntime(summary, [1, 2, 3], runtimePng);
end

fprintf('Done.\n');
fprintf('Raw CSV     : %s\n', rawCsv);
fprintf('Summary CSV : %s\n', summaryCsv);
if p.Results.detailed_timing
    fprintf('Timing Raw  : %s\n', timingRawCsv);
    fprintf('Timing Sum  : %s\n', timingSummaryCsv);
end
if ~p.Results.no_plot
    fprintf('Metrics PNG : %s\n', metricsPng);
    fprintf('Runtime PNG : %s\n', runtimePng);
end

outputs = struct( ...
    'raw_csv', rawCsv, ...
    'summary_csv', summaryCsv, ...
    'timing_raw_csv', timingRawCsv, ...
    'timing_summary_csv', timingSummaryCsv, ...
    'metrics_png', metricsPng, ...
    'runtime_png', runtimePng);
end
