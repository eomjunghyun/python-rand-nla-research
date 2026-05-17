function output = run_sec81_email_matlab(varargin)
%RUN_SEC81_EMAIL_MATLAB Run the European email Section 8.1 MATLAB experiment.

thisDir = fileparts(mfilename('fullpath'));
addpath(thisDir);

output = sec81.Common.runDatasetByName('email_eu', varargin{:});
end
