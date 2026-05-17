function outputs = run_sec81_remaining_matlab(varargin)
%RUN_SEC81_REMAINING_MATLAB Run the three non-email Section 8.1 datasets.

thisDir = fileparts(mfilename('fullpath'));
addpath(thisDir);

p = inputParser;
addParameter(p, 'datasets', {'political_blog', 'statisticians_coauthor', 'statisticians_citation'});
addParameter(p, 'embedding_rank', []);
addParameter(p, 'reps', 20);
addParameter(p, 'seed', 2026);
addParameter(p, 'q', 2);
addParameter(p, 'r', 10);
addParameter(p, 'p_values', [0.7, 0.8]);
addParameter(p, 'sign_k', 2);
addParameter(p, 'force_prepare', false);
addParameter(p, 'no_progress', false);
addParameter(p, 'no_plot', false);
parse(p, varargin{:});

datasets = p.Results.datasets;
if ischar(datasets) || isstring(datasets)
    datasets = cellstr(split(string(datasets), ","));
end

commonArgs = {
    'embedding_rank', p.Results.embedding_rank, ...
    'reps', p.Results.reps, ...
    'seed', p.Results.seed, ...
    'q', p.Results.q, ...
    'r', p.Results.r, ...
    'p_values', p.Results.p_values, ...
    'sign_k', p.Results.sign_k, ...
    'force_prepare', p.Results.force_prepare, ...
    'no_progress', p.Results.no_progress, ...
    'no_plot', p.Results.no_plot ...
};

outputs = struct();
for i = 1:numel(datasets)
    name = char(strtrim(string(datasets{i})));
    outputs.(name) = sec81.Common.runDatasetByName(name, commonArgs{:});
end
end
