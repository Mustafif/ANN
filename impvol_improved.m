function [imp_vol, V0, it, converged] = impvol_improved(S0, K, T, r, V, index, N, m, gamma, tol, itmax)
%
% Improved implied volatility solver for American options using willow tree
% Fixes boundary issues and adds better initial guess
%
% Input
%   S0    -- initial value of stock price
%   K     -- strike prices, matrix
%   T     -- maturities, vector
%   r     -- interest rate
%   V     -- American option prices
%   index -- option type: +1 for call, -1 for put
%   N     -- # of time steps of willow tree
%   m     -- # of tree nodes at each time step
%   gamma -- gamma parameter for sampling z
%   tol   -- tolerance for convergence
%   itmax -- maximum iterations
%
% Output
%   imp_vol   -- implied volatilities of the American options
%   V0        -- final option prices achieved by solver
%   it        -- iterations used for each option
%   converged -- convergence flag for each option
%

% Get dimensions
n = length(T);
k = size(K, 1);

% Generate z values once
z = zq(m, gamma);

% Initialize output matrices
imp_vol = zeros(k, n);
V0 = zeros(k, n);
it = zeros(k, n);
converged = false(k, n);

% Improved bounds - much wider range for GARCH models
sigma_min = 0.0001;  % 0.01% minimum volatility
sigma_max = 3.0;    % 300% maximum volatility

% Statistics tracking
boundary_low_count = 0;
boundary_high_count = 0;
convergence_failures = 0;

% fprintf('Starting improved implied volatility calculation...\n');
% fprintf('Processing %d options with bounds [%.3f, %.1f]\n', k*n, sigma_min, sigma_max);

% Loop over each maturity
for i = 1:n
    T_i = T(i);  % Extract current maturity as scalar

    % Generate probability matrices once per maturity
    [P, q] = gen_PoWiner(T_i, N, z);

    % Loop over each strike
    for j = 1:k
        target_price = V(j, i);
        K_curr = K(j, i);

        % Skip options with negligible value
        if target_price < 1e-8
            imp_vol(j, i) = sigma_min;
            V0(j, i) = target_price;
            it(j, i) = 0;
            converged(j, i) = true;
            continue;
        end

        % Calculate moneyness-based initial guess
        moneyness = K_curr / S0;

        if index == 1  % Call option
            if moneyness < 0.9
                initial_guess = 0.4 + 0.3 * (0.9 - moneyness);  % Higher IV for deep ITM
            elseif moneyness < 1.1
                initial_guess = 0.25 + 0.15 * abs(1 - moneyness);  % ATM base
            else
                initial_guess = 0.2 + 0.1 * (moneyness - 1.1);  % Moderate increase for OTM
            end
        else  % Put option
            if moneyness > 1.1
                initial_guess = 0.4 + 0.3 * (moneyness - 1.1);  % Higher IV for deep ITM
            elseif moneyness > 0.9
                initial_guess = 0.25 + 0.15 * abs(1 - moneyness);  % ATM base
            else
                initial_guess = 0.2 + 0.1 * (0.9 - moneyness);  % Moderate increase for OTM
            end
        end

        % Ensure initial guess is within bounds
        initial_guess = max(sigma_min, min(sigma_max, initial_guess));

        % Test boundary conditions to ensure target can be bracketed
        % Lower bound test
        Xnodes_low = nodes_Winer(T_i, N, z, r, sigma_min);
        nodes_low = S0 .* exp(Xnodes_low);
        price_low = American(nodes_low, P, q, r, T_i, S0, K_curr, index);

        % Upper bound test
        Xnodes_high = nodes_Winer(T_i, N, z, r, sigma_max);
        nodes_high = S0 .* exp(Xnodes_high);
        price_high = American(nodes_high, P, q, r, T_i, S0, K_curr, index);

        % Check if target price can be bracketed
        if target_price <= price_low
            % Target at or below lower bound
            imp_vol(j, i) = sigma_min;
            V0(j, i) = price_low;
            it(j, i) = 0;
            converged(j, i) = true;
            boundary_low_count = boundary_low_count + 1;
            continue;
        elseif target_price >= price_high
            % Target at or above upper bound
            imp_vol(j, i) = sigma_max;
            V0(j, i) = price_high;
            it(j, i) = 0;
            converged(j, i) = true;
            boundary_high_count = boundary_high_count + 1;
            continue;
        end

        % Initialize bisection with improved bounds
        sigma_low = sigma_min;
        sigma_high = sigma_max;
        sigma = initial_guess;

        % Get initial price at guess
        Xnodes = nodes_Winer(T_i, N, z, r, sigma);
        nodes = S0 .* exp(Xnodes);
        current_price = American(nodes, P, q, r, T_i, S0, K_curr, index);

        % Bisection iteration counter
        iter_count = 0;

        % Bisection method with improved logic
        while abs(current_price - target_price) > tol && iter_count < itmax
            % Update bounds based on current price
            if current_price > target_price
                sigma_high = sigma;  % Reduce volatility
            else
                sigma_low = sigma;   % Increase volatility
            end

            % New midpoint
            sigma = (sigma_low + sigma_high) / 2;

            % Check for convergence in bounds (prevents infinite loops)
            if (sigma_high - sigma_low) < 1e-10
                break;
            end

            % Compute new option price
            Xnodes = nodes_Winer(T_i, N, z, r, sigma);
            nodes = S0 .* exp(Xnodes);
            current_price = American(nodes, P, q, r, T_i, S0, K_curr, index);

            iter_count = iter_count + 1;
        end

        % Store results
        imp_vol(j, i) = sigma;
        V0(j, i) = current_price;
        it(j, i) = iter_count;
        converged(j, i) = (abs(current_price - target_price) <= tol);

        % Track convergence failures
        if ~converged(j, i)
            convergence_failures = convergence_failures + 1;
            % fprintf('Warning: Convergence failed for K=%.2f, T=%.3f, m=%.2f, Target=%.4f, Final=%.4f, Error=%.2e\n', ...
            %         K_curr, T_i, moneyness, target_price, current_price, abs(current_price - target_price));
        end
    end

    % Progress indicator
    if n > 1 && mod(i, max(1, floor(n/5))) == 0
        % fprintf('Completed %d/%d maturities...\n', i, n);
    end
end

% Calculate summary statistics
total_options = k * n;
num_converged = sum(converged(:));
convergence_rate = 100 * num_converged / total_options;
avg_iterations = mean(it(converged));

% Display comprehensive summary
% fprintf('\n=== IMPROVED IMPLIED VOLATILITY SOLVER RESULTS ===\n');
% fprintf('Total options processed: %d\n', total_options);
% fprintf('Converged: %d (%.1f%%)\n', num_converged, convergence_rate);
% fprintf('Failed to converge: %d (%.1f%%)\n', convergence_failures, 100*convergence_failures/total_options);
% fprintf('Hit lower boundary: %d (%.1f%%)\n', boundary_low_count, 100*boundary_low_count/total_options);
% fprintf('Hit upper boundary: %d (%.1f%%)\n', boundary_high_count, 100*boundary_high_count/total_options);
% fprintf('Average iterations: %.1f\n', avg_iterations);

% Volatility statistics
all_ivs = imp_vol(:);
% fprintf('\nImplied Volatility Statistics:\n');
% fprintf('  Range: [%.4f, %.4f] (%.1f%% - %.1f%% annualized)\n', ...
%         min(all_ivs), max(all_ivs), min(all_ivs)*sqrt(252)*100, max(all_ivs)*sqrt(252)*100);
% fprintf('  Mean: %.4f (%.1f%% annualized)\n', mean(all_ivs), mean(all_ivs)*sqrt(252)*100);
% fprintf('  Std:  %.4f (%.1f%% annualized)\n', std(all_ivs), std(all_ivs)*sqrt(252)*100);

% Separate statistics for calls and puts if mixed
if any(index == 1) && any(index == -1)
    % This assumes the function is called separately for calls and puts
    % If processing both simultaneously, this section would need modification
    option_type = (index == 1) * 'Call' + (index == -1) * 'Put ';
    % fprintf('  Option type: %s\n', option_type);
end

% Quality warnings
if boundary_low_count > total_options * 0.1
    % fprintf('\nWARNING: %.1f%% of options hit lower bound - consider reducing sigma_min\n', ...
    %         100*boundary_low_count/total_options);
end
if boundary_high_count > total_options * 0.1
    % fprintf('WARNING: %.1f%% of options hit upper bound - consider increasing sigma_max\n', ...
    %         100*boundary_high_count/total_options);
end
if convergence_failures > total_options * 0.05
    % fprintf('WARNING: %.1f%% convergence failure rate - consider increasing itmax or adjusting tol\n', ...
    %         100*convergence_failures/total_options);
end

% fprintf('=== END SOLVER SUMMARY ===\n\n');

end
