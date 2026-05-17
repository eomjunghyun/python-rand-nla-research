function outputs = run_all_sec81_matlab(varargin)
%RUN_ALL_SEC81_MATLAB Run all Reference 1 Section 8.1 MATLAB experiments.

thisDir = fileparts(mfilename('fullpath'));
addpath(thisDir);

p = inputParser;
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

commonArgs = {
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
outputs.email_rank42 = run_sec81_email_matlab(commonArgs{:}, 'embedding_rank', []);
outputs.email_rank30 = run_sec81_email_matlab(commonArgs{:}, 'embedding_rank', 30);

outputs.political_blog_rank2 = sec81.Common.runDatasetByName('political_blog', commonArgs{:}, 'embedding_rank', []);
outputs.political_blog_rank5 = sec81.Common.runDatasetByName('political_blog', commonArgs{:}, 'embedding_rank', 5);

outputs.statisticians_coauthor_rank3 = sec81.Common.runDatasetByName('statisticians_coauthor', commonArgs{:}, 'embedding_rank', []);
outputs.statisticians_coauthor_rank5 = sec81.Common.runDatasetByName('statisticians_coauthor', commonArgs{:}, 'embedding_rank', 5);

outputs.statisticians_citation_rank3 = sec81.Common.runDatasetByName('statisticians_citation', commonArgs{:}, 'embedding_rank', []);
outputs.statisticians_citation_rank5 = sec81.Common.runDatasetByName('statisticians_citation', commonArgs{:}, 'embedding_rank', 5);

comparisonCsv = fullfile(thisDir, 'results', 'section8_1_matlab_rank_comparison.csv');
comparisonMd = fullfile(thisDir, 'results', 'section8_1_matlab_rank_comparison.md');
reportMd = fullfile(thisDir, 'section8_1_matlab_experiment_report.md');
sec81.Common.writeRankComparison(outputs, comparisonCsv, comparisonMd);
sec81.Common.writeReport(comparisonCsv, reportMd);

fprintf('Section 8.1 MATLAB experiments complete.\n');
fprintf('Comparison CSV : %s\n', comparisonCsv);
fprintf('Comparison MD  : %s\n', comparisonMd);
fprintf('Report MD      : %s\n', reportMd);
end
