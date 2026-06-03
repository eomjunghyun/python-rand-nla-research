function outputs = run_all_redesigned_uniform_hsbm_matlab(varargin)
%RUN_ALL_REDESIGNED_UNIFORM_HSBM_MATLAB Run redesigned uniform HSBM MATLAB experiments.

thisDir = fileparts(mfilename('fullpath'));
commonDir = fullfile(thisDir, '..', '균일 HSBM 실험_matlab');
addpath(commonDir);
addpath(thisDir);

outputs = uhsbm.Common.runRedesignedAll(thisDir, varargin{:});
end
