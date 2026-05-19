function [S, h0] = mcDuan(M, N, S0, Z, r, omega, alpha, beta, theta,lambda)
% Monte Carlo simulation for Duan's GARCH option pricing model
%
% Inputs:
%   M - number of simulations
%   N - number of time steps
%   S0 - initial stock price
%   Z - random normal innovations matrix (N+1 x M)
%   r - risk-free rate
%   omega - GARCH constant term
%   alpha - ARCH parameter
%   beta - GARCH parameter
%   lambda - risk premium parameter
%
% Outputs:
%   S - simulated stock price paths
%   h0 - initial variance

    % Check stationarity condition
    if alpha + beta >= 1
        error('Stationarity condition violated: alpha + beta must be less than 1');
    end

    % Initialize variables
    dt = 1/N;  % Time step size (in years)
    numPoints = N + 1;
    S = zeros(numPoints, M);  % Asset price matrix
    ht = zeros(numPoints, M); % Volatility process matrix

    % Calculate initial variance (long-run variance under Duan's specification)
    % Note: omega is not scaled as it represents daily variance
    h0 = omega/(1 - alpha - beta);

    % Set initial conditions
    ht(1, :) = h0;
    S(1, :) = S0;

    % Main simulation loop
    for i = 2:numPoints
        % Update volatility (GARCH process)
        % GARCH parameters operate on daily variance, so no dt scaling needed here
        ht(i, :) = omega + ...                            % Constant term (daily)
                   beta * ht(i-1, :) + ...                % GARCH effect
                   alpha * ht(i-1, :) .* ...              % ARCH effect
                   (Z(i-1, :) - theta - lambda).^2;       % Innovations
        
        % Compute log-returns
        % Only the drift terms need dt scaling as they are annualized
        Xt = r * dt - ...      % Risk premium (no dt scaling as it works with volatility)
             0.5 * ht(i, :) + ...             % Volatility adjustment (no dt as it's daily variance)
             Z(i, :) * sqrt(ht(i, :));        % Diffusion term (no dt as Z scales with daily volatility)
        
        % Update asset prices
        S(i, :) = S(i-1, :) .* exp(Xt);
    end
end