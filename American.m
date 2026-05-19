% function [price,V] = American(nodes, P, q, r, T, S0, K, index)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %
% % Introduction
% % Given the willow tree including tree-nodes matrix and transition
% % probability matrices, this function calculates the price of an
% % American option (put or call).
% %
% % Input
% % nodes (M*N matrix) willow tree-nodes matrix of stock prices
% % P (M*M*(N-1) matrix) P(i,j,k) represents the transition probability from i-th node
% %                      at time t_k to j-th node at t_{k+1}
% % q (M*1 vector) q(i) represents the initial transition probability
% % r (scalar) risk-free rate
% % T (scalar) time to expiration
% % S0 (scalar) initial stock price
% % K (scalar) strike price
% % index (scalar) +1 for call and -1 for put
% %
% % Output
% % price: the computed price of the option
% % V: option values at each node
% %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% % Get dimensions of the tree
% [M, N] = size(nodes);
% 
% % Initialize option values matrix
% V = zeros(M, N);
% 
% % Terminal condition at maturity
% if index == 1  % Call option
%     V(:,N) = max(nodes(:,N) - K, 0);
% else  % Put option
%     V(:,N) = max(K - nodes(:,N), 0);
% end
% 
% % Backward induction
% dt = T/N;
% for i = N-1:-1:1
%     % Calculate continuation values explicitly
%     cont_values = zeros(M, 1);
%     for j = 1:M
%         for k = 1:M
%             cont_values(j) = cont_values(j) + P(j,k,i) * V(k,i+1);
%         end
%     end
%     cont_values = exp(-r*dt) * cont_values;
% 
%     % Calculate exercise values
%     if index == 1  % Call option
%         exercise_values = max(nodes(:,i) - K, 0);
%     else  % Put option
%         exercise_values = max(K - nodes(:,i), 0);
%     end
% 
%     % Take maximum
%     V(:,i) = max(exercise_values, cont_values);
% end
% 
% % Calculate final price
% if index == 1  % Call option
%     immediate_exercise = max(S0 - K, 0);
% else  % Put option
%     immediate_exercise = max(K - S0, 0);
% end
% 
% continuation_value = exp(-r*dt) * q' * V(:,1);
% price = max(immediate_exercise, continuation_value);
% 
% end

function [price,V] = American(nodes, P, q, r, T, S0, K, index)
% [Previous comments remain the same]

% Get dimensions of the tree
[M, N] = size(nodes);

% Initialize option values matrix
V = zeros(M, N);

% Terminal condition at maturity
if index == 1 % Call option
    V(:,N) = max(nodes(:,N) - K, 0);
else % Put option
    V(:,N) = max(K - nodes(:,N), 0);
end

% Correct time step calculation
dt = T/(N-1);

% Backward induction
for i = N-1:-1:1
    % Calculate continuation values
    cont_values = zeros(M, 1);
    for j = 1:M
        for k = 1:M
            cont_values(j) = cont_values(j) + P(j,k,i) * V(k,i+1);
        end
    end
    cont_values = exp(-r*dt) * cont_values;
    
    % Calculate exercise values
    if index == 1 % Call option
        exercise_values = max(nodes(:,i) - K, 0);
    else % Put option
        exercise_values = max(K - nodes(:,i), 0);
    end
    
    % Take maximum (American feature)
    V(:,i) = max(exercise_values, cont_values);
end

% Calculate final price
if index == 1 % Call option
    immediate_exercise = max(S0 - K, 0);
else % Put option
    immediate_exercise = max(K - S0, 0);
end

% Ensure q is treated as row vector for matrix multiplication
continuation_value = exp(-r*dt) * q(:)' * V(:,1);
price = max(immediate_exercise, continuation_value);

end