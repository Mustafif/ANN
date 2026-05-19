function [P_ht_N, P_ht] = Prob_ht(nodes_ht,h0,alpha,beta,gamma,omega)
%
% compute the transition probabilities and probability of h_t of the Heston-Nadi GARCH model and
% construct all tree nodes of h_t
%
%       X_t = r-0.5*h_t + sqrt(h_t)z_t    z_t~~N(0,1)
%       h_{t+1} = omega +alpha(z_t-gamma*sqrt(h_t))^2+beta h_t
%
%
%  Input
%     nodes_ht  -- tree nodes of ht
%     h0 -- initial value of ht
%     alpha, beta, gamma, omega -- parameters for Heston-Nadi GARCH
% 
%  Output
%     P_ht_N -- transition probability matrix of ht, 3-d array
%     P_ht -- probability of ht given h0
%
%       April 22, 2022
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[M,N] = size(nodes_ht);

P_ht = zeros(M,N); % transition probability matrix between two time steps
P_ht_N = zeros(M,M,N-1);
% probability for hd->h2d

curr_h = h0;
next_h = nodes_ht(:, 1);
p = Prob(curr_h,next_h, alpha, beta, gamma, omega);
P_ht(:,1) = p(:);


for n = 2:N
    next_h = nodes_ht(:,n);
    for i = 1:M
        curr_h = nodes_ht(i,n-1);
        p = Prob(curr_h,next_h, alpha, beta, gamma, omega);
        P_ht_N(i,:,n-1) = p(:)';
    end  
    P_ht(:,n) = (P_ht(:,n-1)'*P_ht_N(:,:,n-1))';
end




