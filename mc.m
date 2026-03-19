clear all; clc; close all;
format longE;
% parameter order: alpha, beta, omega, gamma, lambda 
true_p = [1.33e-6, 0.8, 1e-6, 5.0, 0.2];

json_path = 'experiment/results_04.json';
jsonStr = fileread(json_path);
data = jsondecode(jsonStr);

plotprefix = "Figs/lambda_1";
calib_p = [ data.alpha, data.beta, data.omega, data.gamma, data.lambda
];

init_p = [1.50000000e-06 ...
    5.00000015e-01...
    1.00000000e-06 ...
    0.00000000e+00...
    0.00000000e+00
    ];

N = [30, 60, 120, 252, 512, 1024];

X_true = zeros(5, 4);
X_calib = zeros(5, 4);
X_init = zeros(5, 4);

for i=1:length(N)
    n = N(i);
    X_true(i, :) = four_moments(n, true_p(1), true_p(2), true_p(3), true_p(4), true_p(5));
    X_calib(i, :) = four_moments(n, calib_p(1), calib_p(2), calib_p(3), calib_p(4), calib_p(5));
    X_init(i, :) = four_moments(n, init_p(1), init_p(2), init_p(3), init_p(4), init_p(5));
end


rel_err_cal = rel_err(X_calib, X_true);
rel_err_init = rel_err(X_init, X_true);


% Make table of First Four Moments For Each time period and label the four
% moments mean, variance, skewness and kurtosis 
% Create a table to display the results
momentTable = array2table([X_true, X_calib, X_init], ...
    'VariableNames', {'True_Mean', 'True_Variance', 'True_Skewness', 'True_Kurtosis', ...
                      'Calib_Mean', 'Calib_Variance', 'Calib_Skewness', 'Calib_Kurtosis', ...
                      'Init_Mean', 'Init_Variance', 'Init_Skewness', 'Init_Kurtosis'}, ...
    'RowNames', string(N));
% 
% % Make a table for the relative error for each time period and label the
% % four moments mean, variance, skewness and kurtosis
data = [rel_err_cal, rel_err_init];
% 
varNames = {'Mean Cal','Var Cal','Skew Cal','Kurt Cal', ...
            'Mean Init','Var Init','Skew Init','Kurt Init'};

rel_err_table = array2table(data, 'VariableNames', varNames);
% 
% writetable(momentTable, "moments.csv");
% writetable(rel_err_table, "relative_errors.csv");

trueVars  = ["True_Mean","True_Variance","True_Skewness","True_Kurtosis"];
calibVars = ["Calib_Mean","Calib_Variance","Calib_Skewness","Calib_Kurtosis"];
initVars  = ["Init_Mean","Init_Variance","Init_Skewness","Init_Kurtosis"];
momNames  = ["mean","variance","skewness","kurtosis"];

for k = 1:4
    Tm = table( momentTable{:,trueVars(k)}, ...
                momentTable{:,calibVars(k)}, ...
                momentTable{:,initVars(k)}, ...
                'VariableNames', {['True_',char(momNames(k))], ...
                                  ['Calib_',char(momNames(k))], ...
                                  ['Init_',char(momNames(k))]} );
    Tm.Properties.RowNames = momentTable.Properties.RowNames;
    writetable(Tm, sprintf('%s_moment_table.csv', momNames(k)), 'WriteRowNames', true);
    assignin('base', sprintf('tbl_%s', momNames(k)), Tm);
end


% Plot the relative error for each time period and label each of the four
% moments of calibrated and initial, mean, variance, skewness and kurtosis
% figure;
% hold on; 
% colors = lines(8);
% lineSpec = {'-', '--', ':', '-.'};
% for k = 1:4
%     plot(N', rel_err_cal(:, k), ...
%         'LineStyle', lineSpec{k}, 'Color', colors(k, :), ...
%         'LineWidth', 1.5);
% end
% 
% for k = 1:4
%     plot(N', rel_err_init(:, k), ...
%         'LineStyle', lineSpec{k}, 'Color', colors(k+4, :), ...
%         'LineWidth', 1.5, 'Marker', 'o', 'MarkerSize', 4);
% end
% 
% xlabel('Time Period');
% ylabel('Relative Error');
% title('Relative Error by Moment: Calibrated vs Initial');
% legend(varNames, 'Location', 'best');
% grid on;
% hold off;

% Inputs: N, rel_err_cal (T-by-4), rel_err_init (T-by-4)
% Optional: ensure column shape
N = N(:);
T = size(rel_err_cal,1);
assert(numel(N) == T, 'Length of N must match rows of rel_err_cal');

rel_err_cal = reshape(rel_err_cal, T, 4);
rel_err_init = reshape(rel_err_init, T, 4);

momNames = {'Mean','Variance','Skewness','Kurtosis'};
colors = lines(2); % calibrated, initial
lineSpec = {'-','--'};

figure('Color','w');
tiledlayout(4,1, 'Padding','compact', 'TileSpacing','compact');

for m = 1:4
    ax = nexttile;
    hold(ax,'on');
    plot(ax, N, rel_err_cal(:,m), lineSpec{1}, 'Color', colors(1,:), ...
         'LineWidth', 1.5, 'DisplayName', 'Calibrated');
    plot(ax, N, rel_err_init(:,m), lineSpec{2}, 'Color', colors(2,:), ...
         'LineWidth', 1.5, 'DisplayName', 'Initial');
    hold(ax,'off');
    ylabel(ax, momNames{m});
    grid(ax,'on');
    if m == 1
        title('Relative Error by Moment: Calibrated vs Initial');
    end
    if m == 4
        xlabel('Time Period');
    end
    legend(ax,'show','Location','best');
end

% Adjust figure size (optional)
set(gcf,'Units','normalized','Position',[0.2 0.1 0.6 0.8]);
exportgraphics(gcf,plotprefix + "_4Moms.png","Resolution",300);



% momentTable already created as in your code
T = height(momentTable);
% create a time vector from row names if numeric, otherwise 1:T
try
    time = str2double(momentTable.Properties.RowNames);
    if any(isnan(time))
        time = 1:T;
    end
catch
    time = 1:T;
end

% variable name groups
trueVars = ["True_Mean","True_Variance","True_Skewness","True_Kurtosis"];
calibVars = ["Calib_Mean","Calib_Variance","Calib_Skewness","Calib_Kurtosis"];
initVars  = ["Init_Mean","Init_Variance","Init_Skewness","Init_Kurtosis"];
momNames  = {'Mean','Variance','Skewness','Kurtosis'};

figure('Color','w');
colors = lines(3); % True, Calib, Init
for m = 1:4
    ax = subplot(4,1,m);
    hold(ax,'on');
    p1 = plot(time, momentTable{:,trueVars(m)}, '-','Color',colors(1,:), 'LineWidth',1.5, 'DisplayName','True');
    p2 = plot(time, momentTable{:,calibVars(m)}, '--','Color',colors(2,:), 'LineWidth',1.5, 'DisplayName','Calibrated');
    p3 = plot(time, momentTable{:,initVars(m)}, ':','Color',colors(3,:), 'LineWidth',1.5, 'DisplayName','Initial');
    hold(ax,'off');
    ylabel(ax, momNames{m});
    if m==1
        title('Moments: True vs Calibrated vs Initial');
    end
    if m==4
        xlabel('Time Period');
    end
    grid(ax,'on');
    legend(ax,'show','Location','best');
end

% Optional: tighten layout
set(gcf,'Units','normalized','Position',[0.2 0.1 0.6 0.8]);
exportgraphics(gcf,plotprefix + "_RelErr.png","Resolution",300);

function err =  rel_err(x_p, x)
    err = (abs(x_p - x) ./ abs(x));
end

function X = four_moments(N, alpha, beta, omega, gamma, lambda)
    Rt = mcHN(N, alpha, beta, omega, gamma, lambda);
    m = mean(Rt, "all");
    v = var(Rt, 0, "all");
    s = skewness(Rt, 0, "all");
    k = kurtosis(Rt, 0, "all");

    X = [m, v, s, k];
end

function [Rt] = mcHN(N, alpha, beta, omega, gamma, lambda)
M = 1000;
dt = 1/N;
Rt = zeros(N+1, M);
ht = zeros(N+1, M);
Z = randn(N+1, M);
r = 0.05;

ht(1, :) = (omega + alpha)/(1-beta-alpha*gamma^2);
Rt(1, :) = 0;
for i = 2:N
    ht(i, :) = omega + beta.*ht(i-1, :) + alpha.*(Z(i-1, :) - gamma.*sqrt(ht(i-1, :))).^2;
    Rt(i, :) = (r*dt) + lambda.*ht(i, :) + Z(i, :).*sqrt(ht(i, :));
end

Rt = Rt(2:end, :);
end