function [nodes_Xt, mu, var, k3, k4] = TreeNodes_logSt_Duan(m_x, gamma_x, r, hd, qht, S0, alpha, beta, lambda, omega, N)
%
% Compute the first four moments of X_t using Duan's GARCH model and
% construct all tree nodes of X_t
%
%       R_t = r - 0.5*h_t + sqrt(h_t)*z_t    z_t ~ N(0,1)
%       h_t = omega + beta*h_{t-1} + alpha*h_{t-1}*(z_{t-1} - lambda)^2
%
%  Input
%     m_x  -- number of tree nodes
%     gamma_x -- parameter for generating z_t
%     r -- interest rate
%     hd -- discrete values of h_1
%     qht -- corresponding probabilities of hd
%     S0 -- initial stock price
%     alpha, beta, lambda, omega -- parameters for Duan GARCH
%     N -- # of time steps
%
%       February 13, 2025
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

numPoint = N+1;
u = 0;

%% Recursively compute derivatives of B at u=0
% 0th order
diffB0 = zeros(numPoint,1);
diffB0(1) = 0;
for row = 2:numPoint
    diffB0(row) = -0.5*u + beta*diffB0(row-1) + ...
                  alpha*diffB0(row-1)*(lambda^2 + u^2/(1 - 2*alpha*lambda^2*diffB0(row-1)));
end

% 1st order
diffB1 = zeros(numPoint,1);
diffB1(1) = 0;
for row = 2:numPoint
    diffB1(row) = -0.5 + beta*diffB1(row-1) + ...
                  alpha*diffB1(row-1)*(lambda^2 + 2*lambda*u/(1 - 2*alpha*lambda^2*diffB0(row-1))) + ...
                  alpha*diffB0(row-1)*u/(1 - 2*alpha*lambda^2*diffB0(row-1));
end

% 2nd order
diffB2 = zeros(numPoint,1);
diffB2(1) = 0;
for row = 2:numPoint
    diffB2(row) = beta*diffB2(row-1) + ...
                  alpha*diffB2(row-1)*(lambda^2 + 2*lambda*u/(1 - 2*alpha*lambda^2*diffB0(row-1))) + ...
                  2*alpha*diffB1(row-1)*(lambda/(1 - 2*alpha*lambda^2*diffB0(row-1))) + ...
                  alpha*diffB0(row-1)/(1 - 2*alpha*lambda^2*diffB0(row-1));
end

% 3rd order
diffB3 = zeros(numPoint,1);
diffB3(1) = 0;
for row = 2:numPoint
    diffB3(row) = beta*diffB3(row-1) + ...
                  alpha*diffB3(row-1)*(lambda^2 + 2*lambda*u/(1 - 2*alpha*lambda^2*diffB0(row-1))) + ...
                  3*alpha*diffB2(row-1)*(lambda/(1 - 2*alpha*lambda^2*diffB0(row-1))) + ...
                  3*alpha*diffB1(row-1)/(1 - 2*alpha*lambda^2*diffB0(row-1));
end

% 4th order
diffB4 = zeros(numPoint,1);
diffB4(1) = 0;
for row = 2:numPoint
    diffB4(row) = beta*diffB4(row-1) + ...
                  alpha*diffB4(row-1)*(lambda^2 + 2*lambda*u/(1 - 2*alpha*lambda^2*diffB0(row-1))) + ...
                  4*alpha*diffB3(row-1)*(lambda/(1 - 2*alpha*lambda^2*diffB0(row-1))) + ...
                  6*alpha*diffB2(row-1)/(1 - 2*alpha*lambda^2*diffB0(row-1));
end

%% Recursively compute derivatives of A at u=0
% 0th order
diffA0 = zeros(numPoint,1);
diffA0(1) = 0;
for row = 2:numPoint
    diffA0(row) = diffA0(row-1) + u*r + omega*diffB0(row-1) - ...
                  0.5*log(1 - 2*alpha*lambda^2*diffB0(row-1));
end

% 1st order
diffA1 = zeros(numPoint,1);
diffA1(1) = 0;
for row = 2:numPoint
    diffA1(row) = r + diffA1(row-1) + omega*diffB1(row-1) + ...
                  alpha*lambda^2*diffB1(row-1)/(1 - 2*alpha*lambda^2*diffB0(row-1));
end

% 2nd order
diffA2 = zeros(numPoint,1);
diffA2(1) = 0;
for row = 2:numPoint
    diffA2(row) = diffA2(row-1) + omega*diffB2(row-1) + ...
                  2*alpha*lambda^2*diffB1(row-1)^2/(1 - 2*alpha*lambda^2*diffB0(row-1))^2 + ...
                  alpha*lambda^2*diffB2(row-1)/(1 - 2*alpha*lambda^2*diffB0(row-1));
end

% 3rd order
diffA3 = zeros(numPoint,1);
diffA3(1) = 0;
for row = 2:numPoint
    diffA3(row) = diffA3(row-1) + omega*diffB3(row-1) + ...
                  6*alpha*lambda^2*diffB1(row-1)*diffB2(row-1)/(1 - 2*alpha*lambda^2*diffB0(row-1))^2 + ...
                  4*alpha*lambda^3*diffB1(row-1)^3/(1 - 2*alpha*lambda^2*diffB0(row-1))^3 + ...
                  alpha*lambda^2*diffB3(row-1)/(1 - 2*alpha*lambda^2*diffB0(row-1));
end

% 4th order
diffA4 = zeros(numPoint,1);
diffA4(1) = 0;
for row = 2:numPoint
    diffA4(row) = diffA4(row-1) + omega*diffB4(row-1) + ...
                  24*alpha*lambda^3*diffB1(row-1)^2*diffB2(row-1)/(1 - 2*alpha*lambda^2*diffB0(row-1))^3 + ...
                  8*alpha*lambda^4*diffB1(row-1)^4/(1 - 2*alpha*lambda^2*diffB0(row-1))^4 + ...
                  (3*alpha*lambda^2*diffB2(row-1)^2 + 4*alpha*lambda^2*diffB1(row-1)*diffB3(row-1))/(1 - 2*alpha*lambda^2*diffB0(row-1))^2 + ...
                  alpha*lambda^2*diffB4(row-1)/(1 - 2*alpha*lambda^2*diffB0(row-1));
end

%% Compute moment generating function derivatives
m_h = length(hd);
tmp1 = ones(m_h,1).*diffA0' + hd.*diffB0';
tmp2 = ones(m_h,1).*diffA1' + hd.*diffB1';
tmp3 = ones(m_h,1).*diffA2' + hd.*diffB2';
tmp4 = ones(m_h,1).*diffA3' + hd.*diffB3';

% 1st to 4th order derivatives of mgf
diffmgf1 = exp(tmp1).*S0.^u.*(log(S0) + tmp2);
diffmgf2 = exp(tmp1).*S0.^u.*(log(S0).^2 + 2*(tmp2).*log(S0) + (tmp2).^2 + tmp3);
diffmgf3 = exp(tmp1).*S0.^u.*((log(S0)+tmp2).^3 + 3*(log(S0)+tmp2).*tmp3 + tmp4);
diffmgf4 = exp(tmp1).*S0.^u.*((log(S0)+tmp2).^4 + 6*(log(S0)+tmp2).^2.*tmp3 + ...
           3.*tmp3.^2 + 4*(log(S0)+tmp2).*tmp4 + ones(m_h,1).*diffA4' + hd.*diffB4');

% Calculate moments
mom1 = qht'*diffmgf1(:,2:end);
mom2 = qht'*diffmgf2(:,2:end);
mom3 = qht'*diffmgf3(:,2:end);
mom4 = qht'*diffmgf4(:,2:end);

% Calculate standardized moments
mu = mom1;
var = (mom2-mu.^2);
temp = sqrt(var);
k3 = 1./(temp.^3).*(mom3-3.*mu.*mom2+2.*mu.^3);
k4 = 1./(temp.^4).*(mom4-4.*mu.*mom3+6.*(mu.^2).*mom2-3.*mu.^4);

% Replicate moments for all time steps
k33 = repmat(k3(:,1), 1, N);
k44 = repmat(k4(:,1), 1, N);

% Combine moments
G = [mu; var; k33; k44];

% Generate tree nodes
nodes_Xt = Treenodes_JC_X(G, N, m_x, gamma_x);

end