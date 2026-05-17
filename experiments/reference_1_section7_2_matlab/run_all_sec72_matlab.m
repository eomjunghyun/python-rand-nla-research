function outputs = run_all_sec72_matlab(varargin)
%RUN_ALL_SEC72_MATLAB Run both Section 7.2 MATLAB experiment groups.

thisDir = fileparts(mfilename('fullpath'));
addpath(thisDir);

outputs = struct();
outputs.models123 = run_sec72_models123_matlab(varargin{:});
outputs.models456 = run_sec72_models456_matlab(varargin{:});
outputs.report_md = sec72.Common.writeReport(thisDir);
end
