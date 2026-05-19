function [sig,V,V0] = impVol_Duan(r, lambda, w, beta, alpha, gamma, h0, S0, K, T, N, m_h, m_x, CorP)
% impVol_Duan - Compute implied volatility of American options with underlying following HN-GARCH
% under the Q (risk-neutral) measure using Duan's (1995) framework
%
% Model Specification under Q (risk-neutral) measure:
%   R_t = r - 0.5*h_t + sqrt(h_t)*z_t^Q,  where z_t^Q ~ N(0,1)
%   h_t = w + beta*h_{t-1} + alpha*h_{t-1}*(z_{t-1}^Q - lambda)^2
%
% Inputs:
%   r       - Risk-free interest rate
%   lambda  - Risk premium parameter
%   w       - GARCH constant term
%   beta    - GARCH persistence parameter
%   alpha   - GARCH innovation parameter
%   gamma   - Leverage effect parameter (not used under Q-measure)
%   h0      - Initial conditional variance
%   S0      - Initial stock price
%   K       - Strike price
%   T       - Time to maturity (in years)
%   N       - Number of time steps (typically daily)
%   m_h     - Number of discrete points for conditional variance (h_t)
%   m_x     - Number of discrete points for log-returns (X_t)
%   CorP    - Option type indicator (1 for Call, -1 for Put)
%
% Outputs:
%   sig     - Black-Scholes implied volatility
%   V       - Option price matrix at each node
%   V0      - Initial option price
%
% References:
%   Duan, J.-C. (1995). The GARCH Option Pricing Model
%   Mathematical Finance, 5(1), 13-32.

% Willow tree parameters
gamma_h = 0.6;  % Spacing parameter for variance tree
gamma_x = 0.8;  % Spacing parameter for return tree
tol = 1e-4;    % Convergence tolerance for implied vol
itmax = 60;    % Maximum iterations for implied vol calculation

% Step 1: Construct the willow tree for conditional variance (h_t)
[hd, qhd] = duan(h0, beta, alpha,  w, m_h, gamma_h);
[nodes_ht] = TreeNodes_ht_Duan(m_h, hd, qhd, gamma_h, alpha, beta, w, lambda, N+1);

% Step 2: Construct the willow tree for returns (R_t)
% Using r - 0.5*h_t for the drift term under Q-measure
[nodes_Rt, ~, ~, ~, ~] = TreeNodes_logSt_Duan(m_x,gamma_x,  r, hd, qhd, S0, alpha, beta, lambda, w, N);

% Step 3: Calculate transition probabilities
[q_Xt, P_Xt, ~] = Prob_Xt(nodes_ht, qhd, nodes_Rt, S0, r, alpha, beta, lambda, w);

% Step 4: Convert returns to stock prices
nodes_S = exp(nodes_Xt);

% Step 5: Price the American option using dynamic programming
[V, ~] = American(nodes_S, P_Xt, q_Xt, r, T, S0, K, CorP);

% Step 6: Compute the implied volatility using Black-Scholes model
[sig, V0, ~] = impvol(S0, K, T, r, V, CorP, N, m_x, gamma_x, tol, itmax);

end