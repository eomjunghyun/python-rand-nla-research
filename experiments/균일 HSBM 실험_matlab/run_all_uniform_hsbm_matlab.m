function outputs = run_all_uniform_hsbm_matlab(varargin)
%RUN_ALL_UNIFORM_HSBM_MATLAB Run MATLAB uniform HSBM method comparisons.

thisDir = fileparts(mfilename('fullpath'));
addpath(thisDir);

outputs = uhsbm.Common.runUniformAll(thisDir, varargin{:});
end
