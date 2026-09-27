% Export the MATLAB `datetime` time axis to plain numeric form.
%
% WHY THIS EXISTS
%   `ThisTimeSlot` is stored as a MATLAB `datetime`, which in a v5 .mat file is
%   an MCOS object. Python's scipy.io.loadmat returns it as a MatlabOpaque with
%   `_Class = 'datetime'` and CANNOT decode it. MATLAB itself reads it fine, so
%   we use MATLAB once to dump the time axis as a MATLAB datenum (double).
%
%   This is a one-off bridge step; nothing downstream needs MATLAB.
%
% ENCODING NOTE
%   This file is deliberately PURE ASCII. MATLAB on this machine parses .m
%   sources as GBK, and files written by external tools in UTF-8 with Chinese
%   comments fail to parse. ASCII sidesteps that entirely.
%
% USAGE  (run from the repository root)
%   matlab -batch "run('scripts/matlab_export_time.m')"
%
% Paths are derived from this file's own location, so the repository can live
% anywhere. src_dir must match `data_dir` in configs/base.yaml, i.e. the four
% Stand*.mat files go into <repository root>/data/.

here = fileparts(mfilename('fullpath'));      % <root>/scripts
root = fileparts(here);                       % <root>
src_dir = fullfile(root, 'data');             % <root>/data
dst_dir = fullfile(root, 'outputs', 'cache', 'matlab_meta');
if ~exist(dst_dir, 'dir')
    mkdir(dst_dir);
end

files = {'StandTestData1.mat', 'StandTestData2.mat', 'StandTestData3.mat'};

fprintf('=== MATLAB time axis export ===\n');
fprintf('matlab version: %s\n', version);

for k = 1:numel(files)
    fname = files{k};
    fprintf('\n--- %s ---\n', fname);

    S = load(fullfile(src_dir, fname));
    if ~isfield(S, 'ThisTimeSlot')
        fprintf('  NO ThisTimeSlot, skipped\n');
        continue;
    end

    t = S.ThisTimeSlot;
    if ~isdatetime(t)
        fprintf('  WARNING: ThisTimeSlot is class %s, not datetime\n', class(t));
    end

    % MATLAB datenum: days since 0000-01-00 (proleptic). 719529 == 1970-01-01.
    d = datenum(t);
    d = d(:);
    strs = string(t);
    strs = strs(:);

    % Resolve the ID variable, whose name differs between files.
    id_names = {'ThisTrainID', 'ThisID1', 'ThisID'};
    idv = [];
    id_used = '';
    for j = 1:numel(id_names)
        if isfield(S, id_names{j})
            idv = double(S.(id_names{j}));
            idv = idv(:);
            id_used = id_names{j};
            break;
        end
    end

    fprintf('  n = %d, class = %s, id variable = %s\n', numel(d), class(t), id_used);
    fprintf('  range: %s  ->  %s\n', string(min(t)), string(max(t)));

    if ~isempty(idv) && numel(idv) == numel(d)
        ids = unique(idv);
        fprintf('  per-ID time span and within-ID monotonicity:\n');
        for j = 1:numel(ids)
            m = (idv == ids(j));
            tj = d(m);
            n_bad = sum(diff(tj) < 0);
            span_days = max(tj) - min(tj);
            fprintf(['    ID=%3d  n=%6d  %s -> %s  span=%8.1f d  ' ...
                     'negative-steps=%6d (%.4f%%)\n'], ...
                ids(j), sum(m), ...
                datestr(min(tj), 'yyyy-mm-dd HH:MM'), ...
                datestr(max(tj), 'yyyy-mm-dd HH:MM'), ...
                span_days, n_bad, 100 * n_bad / max(numel(tj) - 1, 1));
        end

        % Cross-ID ordering: is each ID a contiguous chronological block?
        fprintf('  ID block order by first timestamp:\n');
        first_t = zeros(numel(ids), 1);
        for j = 1:numel(ids)
            m = (idv == ids(j));
            first_t(j) = min(d(m));
        end
        [~, ord] = sort(first_t);
        for j = 1:numel(ids)
            fprintf('    %d) ID=%d  first=%s\n', j, ids(ord(j)), ...
                datestr(first_t(ord(j)), 'yyyy-mm-dd HH:MM'));
        end
    else
        fprintf('  WARNING: id vector missing or length mismatch, skipping per-ID checks\n');
    end

    out_name = sprintf('time_%s.mat', fname(1:end-4));
    save(fullfile(dst_dir, out_name), 'd', 'strs', '-v7');
    fprintf('  saved: %s\n', fullfile(dst_dir, out_name));

    clear S t d strs idv;
end

fprintf('\n=== done ===\n');
