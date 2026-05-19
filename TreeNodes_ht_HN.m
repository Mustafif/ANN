% function [nodes_ht, mu,var, k3,k4] = TreeNodes_ht_HN(m_h, hd, qht, gamma_h,alpha,beta,gamma,omega,N)
% %
% % compute the first four moments of h_t of the Heston-Nadi GARCH model and
% % construct all tree nodes of h_t
% %
% %       X_t = r-0.5*h_t + sqrt(h_t)z_t    z_t~~N(0,1)
% %       h_{t+1} = omega +alpha(z_t-gamma*sqrt(h_t))^2+beta h_t
% %
% %
% %  Input
% %     m_h  -- number of tree nodes of h_t
% %     hd -- discrete values of h_1
% %     qht -- corresponding probabilities of hd
% %     gamma_h -- parameter for generating z_t
% %     alpha, beta, gamma, omega -- parameters for Heston-Nadi GARCH
% %     N -- # of time steps.
% %
% %       April 22, 2022
% %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% numPoint = N+1;
% u =0;
% %% Recursivesly compute the derivatives of B at u=0
% % 0th order
% 
% diffB0 = zeros(numPoint,1);
% diffB0(1) = 0; % for u=0
% 
% % diffB0(1) = 0; 
% % for row = 2:numPoint
% % diffB0(row) = -0.5*gamma^2 + (-0.5 + gamma)*u + beta*diffB0(row-1) + (-gamma + u)^2/(2*(1 - 2*alpha*diffB0(row-1)));
% % end
% 
% % 1st order
% for row = 2:numPoint
%     diffB0(row) = beta*diffB0(row-1)+(alpha*gamma^2*diffB0(row-1))/(1-2*alpha*diffB0(row-1));
% end
% 
% % 1st order
% diffB1 = zeros(numPoint,1);
% diffB1(1) = 1; % 1st order derivative equals 1
% for row = 2:numPoint
%     diffB1(row) = ((beta + alpha*gamma^2 + 4*alpha*beta*diffB0(row-1)*(-1 + alpha*diffB0(row-1)))*diffB1(row-1))/(1 - 2*alpha*diffB0(row-1))^2;
% end
% 
% % 2nd order
% diffB2 = zeros(numPoint,1);
% diffB2(1) = 0; % 2nd order derivative equals 0
% for row = 2:numPoint
%     diffB2(row) = beta*diffB2(row-1) + (8*alpha^3*gamma^2*diffB0(row-1)*diffB1(row-1).^2)/(1 - 2*alpha*diffB0(row-1)).^3+ ...
%         (2*alpha^2*gamma^2*(2*diffB1(row-1).^2+diffB0(row-1)*diffB2(row-1)))/(1 - 2*alpha*diffB0(row-1)).^2 + ...
%         alpha*gamma^2*diffB2(row-1)/(1 - 2*alpha*diffB0(row-1));
% end
% 
% % 3rd order
% diffB3 = zeros(numPoint,1);
% diffB3(1) = 0; %3rd order derivative equals 0
% for row = 2:numPoint
%     diffB3(row) = beta*diffB3(row-1) + 48*alpha^4*gamma^2*diffB0(row-1)*diffB1(row-1).^3/(1 - 2*alpha*diffB0(row-1))^4 + ...
%         + (24*alpha^3*gamma^2*diffB1(row-1).^3+24*alpha^3*gamma^2*diffB0(row-1)*diffB1(row-1)*diffB2(row-1))/(1 - 2*alpha*diffB0(row-1))^3 + ...
%         +(12*alpha^2*gamma^2*diffB1(row-1)*diffB2(row-1) +2*alpha^2*gamma^2*diffB0(row-1)*diffB3(row-1))/(1 - 2*alpha*diffB0(row-1))^2 + ...
%         alpha*gamma^2*diffB3(row-1)/(1 - 2*alpha*diffB0(row-1));
% end
% 
% % 4th order
% diffB4 = zeros(numPoint,1);
% diffB4(1) = 0; %4th order derivative equals 0
% for row = 2:numPoint
% 
%     diffB4(row) = beta*diffB4(row-1) + 384*alpha^5*gamma^2*diffB0(row-1)*diffB1(row-1)^4/(1 - 2*alpha*diffB0(row-1))^5 + ...
%         (192*alpha^4*gamma^2*diffB1(row-1)^4+288*alpha^4*gamma^2*diffB0(row-1)*diffB1(row-1)^2*diffB2(row-1))/(1 - 2*alpha*diffB0(row-1))^4 + ...
%         (144*alpha^3*gamma^2*diffB1(row-1)^2*diffB2(row-1)+24*alpha^3*gamma^2*diffB0(row-1)*diffB2(row-1)^2+32*alpha^3*gamma^2*diffB0(row-1)*diffB1(row-1)*diffB3(row-1))/(1 - 2*alpha*diffB0(row-1))^3 +...
%         (12*alpha^2*gamma^2*diffB2(row-1)^2+16*alpha^2*gamma^2*diffB1(row-1)*diffB3(row-1)+2*alpha^2*gamma^2*diffB0(row-1)*diffB4(row-1))/(1 - 2*alpha*diffB0(row-1))^2 +...
%         alpha*gamma^2*diffB4(row-1)/(1 - 2*alpha*diffB0(row-1));
% end
% 
% %% Recursively compute the derivatives of A at u=0
% % 0th order
% diffA0 = zeros(numPoint,1);
% diffA0(1) = 0;
% for row = 2:numPoint
%     diffA0(row) =  diffA0(row-1)+omega*diffB0(row-1)-1/2*log(1-2*alpha*diffB0(row-1));
% end
% 
% % 1st order
% diffA1 = zeros(numPoint,1);
% diffA1(1) = 0;
% for row = 2:numPoint
%     diffA1(row) =  diffA1(row-1) + (omega + (1.*alpha)/(1 - 2*alpha*diffB0(row-1)))*diffB1(row-1);
% end
% 
% % 2nd order
% diffA2 = zeros(numPoint,1);
% diffA2(1) = 0;
% for row = 2:numPoint
%     diffA2(row) =  (2.*alpha^2*diffB1(row-1)^2)/(1 - 2*alpha*diffB0(row-1))^2 + diffA2(row-1) + ...
%         (omega + (1.*alpha)/(1 - 2*alpha*diffB0(row-1)))*diffB2(row-1);
% end
% 
% % 3rd order
% diffA3 = zeros(numPoint,1);
% diffA3(1) = 0;
% for row = 2:numPoint
%     diffA3(row) =  (8.*alpha^3*diffB1(row-1)^3)/(1 - 2*alpha*diffB0(row-1))^3 +...
%         (6.*alpha^2*diffB1(row-1)*diffB2(row-1))/(1 - 2*alpha*diffB0(row-1))^2 + ...
%         diffA3(row-1) + (omega + (1.*alpha)/(1 - 2*alpha*diffB0(row-1)))*diffB3(row-1);
% end
% 
% % 4th order
% diffA4 = zeros(numPoint,1);
% diffA4(1) = 0;
% for row = 2:numPoint
% diffA4(row) = diffA4(row-1) + omega*diffB4(row-1)+(48*alpha^4*diffB1(row-1).^4)/(1-2*alpha*diffB0(row-1)).^4 ...
%     +(48*alpha.^3*diffB1(row-1).^2.*diffB2(row-1))/(1-2*alpha.*diffB0(row-1)).^3 + ...
%     (6*alpha^2*diffB2(row-1).^2+8*alpha.^2.*diffB1(row-1).*diffB3(row-1))/(1-2*alpha.*diffB0(row-1)).^2 +...
%     alpha.*diffB4(row-1)/(1-2*alpha.*diffB0(row-1));
% end
% 
% %%Recursively compute the derivatives of m.g.f at u=0
% 
% tmp1 = ones(m_h,1).*diffA0' + hd.*diffB0';
% tmp2 = ones(m_h,1).*diffA1' + hd.*diffB1';
% tmp3 = ones(m_h,1).*diffA2' + hd.*diffB2';
% tmp4 = ones(m_h,1).*diffA3' + hd.*diffB3';
% % 0th order
% % diffmgf0 = exp(diffA0+hd.*diffB0);
% 
% % 1st order
% diffmgf1 = exp(tmp1).*tmp2;
% 
% % 2nd order
% diffmgf2 = exp(tmp1).*((tmp2).^2 + tmp3) ;
% 
% % 3rd order
% diffmgf3 = exp(tmp1).*(tmp2.^3 + 3.*(tmp2).*(tmp3) + tmp4);
% 
% % 4th order
% diffmgf4 = exp(tmp1).*((tmp2).^4 + ...
%     6.*(tmp2).^2.*(tmp3) + 3.*(tmp3).^2 + ...
%     4.*(tmp2).*(tmp4) + ones(m_h,1).*diffA4' + hd.*diffB4');
% 
% mom1 = qht'*diffmgf1(:,2:end);
% mom2 = qht'*diffmgf2(:,2:end);
% mom3 = qht'*diffmgf3(:,2:end);
% mom4 = qht'*diffmgf4(:,2:end);
% 
% %% generate nodes by 4th order moments
% mu = mom1;
% var = mom2-mu.^2;
% temp = sqrt(var);
% k3 = 1./(temp.^3).*(mom3-3*mu.*mom2+2*mu.^3);
% k4 = 1./(temp.^4).*(mom4-4*mu.*mom3+6*(mu.^2).*mom2-3*mu.^4);
% 
% G= [mu;var;k3;k4];
% [nodes_ht] = Treenodes_JC_h(G,N,m_h,gamma_h);
% 
% end
% 
% 
% 
% 

function [nodes_ht, mu, var, k3, k4] = TreeNodes_ht_HN(m_h, hd, qht, gamma_h, alpha, beta, gamma, omega, N)
%
% compute the first four moments of h_t of the Heston-Nadi GARCH model and
% construct all tree nodes of h_t
%
%       X_t = r-0.5*h_t + sqrt(h_t)z_t    z_t~~N(0,1)
%       h_{t+1} = omega +alpha(z_t-gamma*sqrt(h_t))^2+beta h_t
%
%
%  Input
%     m_h  -- number of tree nodes of h_t
%     hd -- discrete values of h_1
%     qht -- corresponding probabilities of hd
%     gamma_h -- parameter for generating z_t
%     alpha, beta, gamma, omega -- parameters for Heston-Nadi GARCH
%     N -- # of time steps.
%
%       April 22, 2022
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Input validation
if length(hd) ~= length(qht)
    error('hd and qht must have the same length');
end
if abs(sum(qht) - 1) > 1e-10
    error('Probabilities qht must sum to 1');
end

numPoint = N + 1;
u = 0;

%% Recursively compute the derivatives of B at u=0
% 0th order
diffB0 = zeros(numPoint, 1);
diffB0(1) = 0; % for u=0

% 1st order through recursive calculation
for row = 2:numPoint
    denominator = 1 - 2*alpha*diffB0(row-1);
    if abs(denominator) < 1e-12
        error('Division by zero in diffB0 calculation at step %d', row);
    end
    diffB0(row) = beta*diffB0(row-1) + (alpha*gamma^2*diffB0(row-1))/denominator;
end

% 1st order derivative
diffB1 = zeros(numPoint, 1);
diffB1(1) = 1; % 1st order derivative equals 1
for row = 2:numPoint
    denominator = 1 - 2*alpha*diffB0(row-1);
    if abs(denominator) < 1e-12
        error('Division by zero in diffB1 calculation at step %d', row);
    end
    numerator = (beta + alpha*gamma^2 + 4*alpha*beta*diffB0(row-1)*(-1 + alpha*diffB0(row-1)))*diffB1(row-1);
    diffB1(row) = numerator / (denominator^2);
end

% 2nd order derivative
diffB2 = zeros(numPoint, 1);
diffB2(1) = 0; % 2nd order derivative equals 0
for row = 2:numPoint
    denominator = 1 - 2*alpha*diffB0(row-1);
    if abs(denominator) < 1e-12
        error('Division by zero in diffB2 calculation at step %d', row);
    end
    
    term1 = beta*diffB2(row-1);
    term2 = (8*alpha^3*gamma^2*diffB0(row-1)*diffB1(row-1)^2) / (denominator^3);
    term3 = (2*alpha^2*gamma^2*(2*diffB1(row-1)^2 + diffB0(row-1)*diffB2(row-1))) / (denominator^2);
    term4 = (alpha*gamma^2*diffB2(row-1)) / denominator;
    
    diffB2(row) = term1 + term2 + term3 + term4;
end

% 3rd order derivative
diffB3 = zeros(numPoint, 1);
diffB3(1) = 0; % 3rd order derivative equals 0
for row = 2:numPoint
    denominator = 1 - 2*alpha*diffB0(row-1);
    if abs(denominator) < 1e-12
        error('Division by zero in diffB3 calculation at step %d', row);
    end
    
    term1 = beta*diffB3(row-1);
    term2 = 48*alpha^4*gamma^2*diffB0(row-1)*diffB1(row-1)^3 / (denominator^4);
    term3 = (24*alpha^3*gamma^2*diffB1(row-1)^3 + 24*alpha^3*gamma^2*diffB0(row-1)*diffB1(row-1)*diffB2(row-1)) / (denominator^3);
    term4 = (12*alpha^2*gamma^2*diffB1(row-1)*diffB2(row-1) + 2*alpha^2*gamma^2*diffB0(row-1)*diffB3(row-1)) / (denominator^2);
    term5 = alpha*gamma^2*diffB3(row-1) / denominator;
    
    diffB3(row) = term1 + term2 + term3 + term4 + term5;
end

% 4th order derivative
diffB4 = zeros(numPoint, 1);
diffB4(1) = 0; % 4th order derivative equals 0
for row = 2:numPoint
    denominator = 1 - 2*alpha*diffB0(row-1);
    if abs(denominator) < 1e-12
        error('Division by zero in diffB4 calculation at step %d', row);
    end
    
    term1 = beta*diffB4(row-1);
    term2 = 384*alpha^5*gamma^2*diffB0(row-1)*diffB1(row-1)^4 / (denominator^5);
    term3 = (192*alpha^4*gamma^2*diffB1(row-1)^4 + 288*alpha^4*gamma^2*diffB0(row-1)*diffB1(row-1)^2*diffB2(row-1)) / (denominator^4);
    term4 = (144*alpha^3*gamma^2*diffB1(row-1)^2*diffB2(row-1) + 24*alpha^3*gamma^2*diffB0(row-1)*diffB2(row-1)^2 + 32*alpha^3*gamma^2*diffB0(row-1)*diffB1(row-1)*diffB3(row-1)) / (denominator^3);
    term5 = (12*alpha^2*gamma^2*diffB2(row-1)^2 + 16*alpha^2*gamma^2*diffB1(row-1)*diffB3(row-1) + 2*alpha^2*gamma^2*diffB0(row-1)*diffB4(row-1)) / (denominator^2);
    term6 = alpha*gamma^2*diffB4(row-1) / denominator;
    
    diffB4(row) = term1 + term2 + term3 + term4 + term5 + term6;
end

%% Recursively compute the derivatives of A at u=0
% 0th order
diffA0 = zeros(numPoint, 1);
diffA0(1) = 0;
for row = 2:numPoint
    denominator = 1 - 2*alpha*diffB0(row-1);
    if abs(denominator) < 1e-12
        error('Division by zero in diffA0 calculation at step %d', row);
    end
    if denominator <= 0
        error('Logarithm of non-positive number in diffA0 calculation at step %d', row);
    end
    diffA0(row) = diffA0(row-1) + omega*diffB0(row-1) - 0.5*log(denominator);
end

% 1st order
diffA1 = zeros(numPoint, 1);
diffA1(1) = 0;
for row = 2:numPoint
    denominator = 1 - 2*alpha*diffB0(row-1);
    if abs(denominator) < 1e-12
        error('Division by zero in diffA1 calculation at step %d', row);
    end
    diffA1(row) = diffA1(row-1) + (omega + alpha/denominator)*diffB1(row-1);
end

% 2nd order
diffA2 = zeros(numPoint, 1);
diffA2(1) = 0;
for row = 2:numPoint
    denominator = 1 - 2*alpha*diffB0(row-1);
    if abs(denominator) < 1e-12
        error('Division by zero in diffA2 calculation at step %d', row);
    end
    term1 = (2*alpha^2*diffB1(row-1)^2) / (denominator^2);
    term2 = diffA2(row-1);
    term3 = (omega + alpha/denominator)*diffB2(row-1);
    diffA2(row) = term1 + term2 + term3;
end

% 3rd order
diffA3 = zeros(numPoint, 1);
diffA3(1) = 0;
for row = 2:numPoint
    denominator = 1 - 2*alpha*diffB0(row-1);
    if abs(denominator) < 1e-12
        error('Division by zero in diffA3 calculation at step %d', row);
    end
    term1 = (8*alpha^3*diffB1(row-1)^3) / (denominator^3);
    term2 = (6*alpha^2*diffB1(row-1)*diffB2(row-1)) / (denominator^2);
    term3 = diffA3(row-1);
    term4 = (omega + alpha/denominator)*diffB3(row-1);
    diffA3(row) = term1 + term2 + term3 + term4;
end

% 4th order
diffA4 = zeros(numPoint, 1);
diffA4(1) = 0;
for row = 2:numPoint
    denominator = 1 - 2*alpha*diffB0(row-1);
    if abs(denominator) < 1e-12
        error('Division by zero in diffA4 calculation at step %d', row);
    end
    term1 = diffA4(row-1);
    term2 = omega*diffB4(row-1);
    term3 = (48*alpha^4*diffB1(row-1)^4) / (denominator^4);
    term4 = (48*alpha^3*diffB1(row-1)^2*diffB2(row-1)) / (denominator^3);
    term5 = (6*alpha^2*diffB2(row-1)^2 + 8*alpha^2*diffB1(row-1)*diffB3(row-1)) / (denominator^2);
    term6 = alpha*diffB4(row-1) / denominator;
    diffA4(row) = term1 + term2 + term3 + term4 + term5 + term6;
end

%% Recursively compute the derivatives of m.g.f at u=0
% Ensure hd is a column vector
if size(hd, 1) == 1
    hd = hd';
end

% Create matrices for computation
tmp1 = repmat(diffA0', m_h, 1) + hd * diffB0';
tmp2 = repmat(diffA1', m_h, 1) + hd * diffB1';
tmp3 = repmat(diffA2', m_h, 1) + hd * diffB2';
tmp4 = repmat(diffA3', m_h, 1) + hd * diffB3';

% Check for numerical stability
if any(tmp1(:) > 700) || any(tmp1(:) < -700)
    warning('Large exponential arguments detected, results may be unstable');
end

% Compute derivatives of moment generating function
exp_tmp1 = exp(tmp1);

% 1st order
diffmgf1 = exp_tmp1 .* tmp2;

% 2nd order
diffmgf2 = exp_tmp1 .* (tmp2.^2 + tmp3);

% 3rd order
diffmgf3 = exp_tmp1 .* (tmp2.^3 + 3*tmp2.*tmp3 + tmp4);

% 4th order
tmp5 = repmat(diffA4', m_h, 1) + hd * diffB4';
diffmgf4 = exp_tmp1 .* (tmp2.^4 + 6*tmp2.^2.*tmp3 + 3*tmp3.^2 + 4*tmp2.*tmp4 + tmp5);

% Compute moments (skip the first column which corresponds to t=0)
mom1 = qht' * diffmgf1(:, 2:end);
mom2 = qht' * diffmgf2(:, 2:end);
mom3 = qht' * diffmgf3(:, 2:end);
mom4 = qht' * diffmgf4(:, 2:end);

%% Generate nodes by 4th order moments
mu = mom1;
var = mom2 - mu.^2;

% Check for negative variance
if any(var <= 0)
    warning('Negative or zero variance detected, setting to small positive value');
    var = max(var, 1e-10);
end

temp = sqrt(var);
k3 = (mom3 - 3*mu.*mom2 + 2*mu.^3) ./ (temp.^3);
k4 = (mom4 - 4*mu.*mom3 + 6*(mu.^2).*mom2 - 3*mu.^4) ./ (temp.^4);

% Check for NaN or Inf values
if any(~isfinite([mu(:); var(:); k3(:); k4(:)]))
    error('Non-finite values detected in moment calculations');
end

G = [mu; var; k3; k4];
[nodes_ht] = Treenodes_JC_h(G, N, m_h, gamma_h);

end