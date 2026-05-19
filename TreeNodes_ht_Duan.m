function [nodes_ht, mu, var, k3, k4] = TreeNodes_ht_Duan(m_h, hd, qht, gamma_h, alpha, beta, omega, lambda, N)
% Compute the first four moments of h_t of Duan's GARCH model and
% construct all tree nodes of h_t under Q measure
%
% Duan's GARCH specification under Q measure:
%       R_t = r - 0.5*h_t + sqrt(h_t)*z_t    z_t ~ N(0,1)
%       h_t = omega + beta*h_{t-1} + alpha*h_{t-1}*(z_{t-1} - lambda)^2
%
% Input:
%     m_h    -- number of tree nodes of h_t
%     hd     -- discrete values of h_1
%     qht    -- corresponding probabilities of hd
%     gamma_h -- parameter for generating z_t
%     alpha  -- GARCH innovation parameter
%     beta   -- GARCH persistence parameter
%     omega  -- GARCH constant term
%     lambda -- Risk premium parameter
%     N      -- # of time steps

numPoint = N + 1;

% Initialize arrays for moment derivatives
diffB0 = zeros(numPoint, 1);
diffB1 = zeros(numPoint, 1);
diffB2 = zeros(numPoint, 1);
diffB3 = zeros(numPoint, 1);
diffB4 = zeros(numPoint, 1);

% Initialize first values
diffB0(1) = 0;
diffB1(1) = 1;
diffB2(1) = 0;
diffB3(1) = 0;
diffB4(1) = 0;

% Compute B derivatives under Q-measure specification
for row = 2:numPoint
    % Define terms for Q-measure adjustments
    lambda_term = lambda^2;
    
    % 0th order - adjusted for Q-measure
    diffB0(row) = beta * diffB0(row-1) + alpha * (1 + lambda_term) * diffB0(row-1);
    
    % 1st order - adjusted for Q-measure
    diffB1(row) = beta * diffB1(row-1) + alpha * (1 + lambda_term) * diffB1(row-1);
    
    % 2nd order - adjusted for Q-measure
    diffB2(row) = beta * diffB2(row-1) + alpha * (1 + 2*lambda_term) * diffB2(row-1) + ...
                  2 * alpha * diffB1(row-1)^2;
    
    % 3rd order - adjusted for Q-measure
    diffB3(row) = beta * diffB3(row-1) + alpha * (1 + 3*lambda_term) * diffB3(row-1) + ...
                  6 * alpha * diffB1(row-1) * diffB2(row-1);
    
    % 4th order - adjusted for Q-measure
    diffB4(row) = beta * diffB4(row-1) + alpha * (1 + 4*lambda_term) * diffB4(row-1) + ...
                  8 * alpha * diffB1(row-1) * diffB3(row-1) + ...
                  6 * alpha * diffB2(row-1)^2;
end

% Initialize and compute A derivatives
diffA0 = zeros(numPoint, 1);
diffA1 = zeros(numPoint, 1);
diffA2 = zeros(numPoint, 1);
diffA3 = zeros(numPoint, 1);
diffA4 = zeros(numPoint, 1);

for row = 2:numPoint
    % 0th order - adjusted for Q-measure
    diffA0(row) = diffA0(row-1) + omega * diffB0(row-1) + ...
                  0.5 * log(1 + 2*alpha*lambda_term*diffB0(row-1));
    
    % 1st order - adjusted for Q-measure
    diffA1(row) = diffA1(row-1) + omega * diffB1(row-1) + ...
                  alpha * lambda_term * diffB1(row-1);
    
    % 2nd order - adjusted for Q-measure
    diffA2(row) = diffA2(row-1) + omega * diffB2(row-1) + ...
                  alpha * lambda_term * diffB2(row-1) + ...
                  alpha * diffB1(row-1)^2;
    
    % 3rd order - adjusted for Q-measure
    diffA3(row) = diffA3(row-1) + omega * diffB3(row-1) + ...
                  alpha * lambda_term * diffB3(row-1) + ...
                  3 * alpha * diffB1(row-1) * diffB2(row-1);
    
    % 4th order - adjusted for Q-measure
    diffA4(row) = diffA4(row-1) + omega * diffB4(row-1) + ...
                  alpha * lambda_term * diffB4(row-1) + ...
                  4 * alpha * diffB1(row-1) * diffB3(row-1) + ...
                  3 * alpha * diffB2(row-1)^2;
end

% Compute moment generating function derivatives
tmp1 = ones(m_h,1) .* diffA0' + hd .* diffB0';
tmp2 = ones(m_h,1) .* diffA1' + hd .* diffB1';
tmp3 = ones(m_h,1) .* diffA2' + hd .* diffB2';
tmp4 = ones(m_h,1) .* diffA3' + hd .* diffB3';

% Compute MGF derivatives
diffmgf1 = exp(tmp1) .* tmp2;
diffmgf2 = exp(tmp1) .* (tmp2.^2 + tmp3);
diffmgf3 = exp(tmp1) .* (tmp2.^3 + 3.*tmp2.*tmp3 + tmp4);
diffmgf4 = exp(tmp1) .* (tmp2.^4 + 6.*tmp2.^2.*tmp3 + 3.*tmp3.^2 + ...
    4.*tmp2.*(ones(m_h,1).*diffA4' + hd.*diffB4'));

% Compute moments
mom1 = qht' * diffmgf1(:,2:end);
mom2 = qht' * diffmgf2(:,2:end);
mom3 = qht' * diffmgf3(:,2:end);
mom4 = qht' * diffmgf4(:,2:end);

% Compute standardized moments
mu = mom1;
var = mom2 - mu.^2;
temp = sqrt(var);
k3 = (mom3 - 3*mu.*mom2 + 2*mu.^3) ./ temp.^3;
k4 = (mom4 - 4*mu.*mom3 + 6*(mu.^2).*mom2 - 3*mu.^4) ./ temp.^4;

% Generate tree nodes
G = [mu; var; k3; k4];
[nodes_ht] = Treenodes_JC_h(G, N, m_h, gamma_h);

end