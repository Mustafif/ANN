function [q_Xt, P_Xt_N, tmpHt] = Prob_Xt_Duan(nodes_ht, q_ht, nodes_Xt, S0, r, alpha, beta, lambda, omega)
%
% compute the transition probabilities X_t of the Duan GARCH model and
% construct all tree nodes of X_t and h_t
%
% Duan's GARCH specification:
%       X_t = r + lambda*h_t + sqrt(h_t)z_t    z_t~~N(0,1)
%       h_{t+1} = omega + alpha*(r_t - r - lambda*h_t)^2 + beta*h_t
%
%  Input
%     nodes_ht  -- tree nodes of ht
%     q_ht -- probabilities of h_1 given h0
%     nodes_Xt -- tree nodes of Xt
%     S0 -- initial value of St
%     r -- interest rate
%     alpha, beta -- GARCH parameters
%     lambda -- risk premium parameter
%     omega -- constant term in variance equation
%
%  Output
%     q_Xt -- transition probabilities from X1 to X0
%     P_Xt_N -- transition probability matrix of Xt, 3-d array
%     tmpHt -- temporary storage for h_t values
%

X0 = log(S0);
m_h = size(nodes_ht, 1);
[X_len, N] = size(nodes_Xt);

% Compute q_Xt for t=1
Xt = nodes_Xt(:, 1);
m_x = X_len;
cur_ht = nodes_ht(:, 1);

q_Xt = zeros(m_x, 1);
mu = r + lambda * cur_ht; % Changed from H-N to Duan specification
std = sqrt(cur_ht);
dx = Xt - X0;
intX = [-inf; (dx(1:end-1) + dx(2:end)) / 2; inf];
tmpP = zeros(m_h, m_x);
Ph = zeros(m_h, m_x, N);
Ph_XXh_h = zeros(m_h, m_x, m_x);
next_ht = nodes_ht(:, 2);

for i = 1:m_h
    % compute P(X_i^1|X^0)
    p = normcdf(intX, mu(i), std(i));
    p = diff(p);
    tmpP(i, :) = p(:)';
    
    % compute P(h_j^2|X_i^1,X^0)
    ht = cur_ht(i);
    % Duan's innovation calculation
    returns = dx - r - lambda * ht;
    
    % Calculate next period's variance bounds
    intH = [omega + beta * ht + 1e-16; (next_ht(1:end-1) + next_ht(2:end)) / 2; 1];
    
    % Calculate innovation bounds for variance transition
    z = (returns) ./ sqrt(ht);
    
    for j = 1:m_x
        % Compute variance transition probabilities under Duan's specification
        next_h = omega + alpha * (returns(j))^2 + beta * ht;
        
        % Find probability regions that satisfy variance transitions
        ph1 = zeros(m_h, 1);
        for k = 1:m_h
            if next_h >= intH(k) && next_h < intH(k+1)
                ph1(k) = 1;
            end
        end
        
        Ph_XXh_h(:, j, i) = ph1 ./ sum(ph1);
    end
end

q_Xt = q_ht' * tmpP;
q_Xt = q_Xt(:);

for i = 1:m_h
    Ph_YY = tmpP(i, :) .* q_ht(i) ./ (q_Xt');
    Ph(:, :, 1) = Ph(:, :, 1) + Ph_XXh_h(:, :, i) .* repmat(Ph_YY, m_h, 1);
end

for i = 1:m_x
    Ph(:, i, 1) = Ph(:, i, 1) / sum(Ph(:, i, 1));
end

P_Xt = zeros(m_x, N);
P_Xt(:, 1) = q_Xt;
tmpHt(:, 1) = Ph(:, :, 1) * P_Xt(:, 1);

% compute transition probability matrices [p_ij]^n
P_Xt_N = zeros(m_x, m_x, N - 1);

for n = 1:N-1
    next_ht = nodes_ht(:, n + 2);
    cur_ht = nodes_ht(:, n + 1);
    Xt = nodes_Xt(:, n + 1);
    mu = r + lambda * cur_ht; % Duan specification
    std = sqrt(cur_ht);
    Ph_XXX = zeros(m_h, m_x, m_x, m_h);
    tmpP = zeros(m_h, m_x, m_x);

    for i = 1:m_x  % X_i^n
        cur_Xt = nodes_Xt(i, n);
        dx = Xt - cur_Xt;
        intX = [-1000; (dx(1:end-1) + dx(2:end)) / 2; 1000];

        for j = 1:m_h  % h_j^n+1
            % compute P(X^n+1|X_i^n, h_j^n+1)
            p = normcdf(intX, mu(j), std(j));
            p = diff(p);
            tmpP(j, :, i) = p(:)';

            % Compute variance transitions under Duan's specification
            ht = cur_ht(j);
            returns = dx - r - lambda * ht;
            
            for k = 1:m_x  % X_j^n+1
                next_h = omega + alpha * (returns(k))^2 + beta * ht;
                
                % Find probability regions that satisfy variance transitions
                ph1 = zeros(m_h, 1);
                for m = 1:m_h
                    if next_h >= intH(m) && next_h < intH(m+1)
                        ph1(m) = 1;
                    end
                end
                
                if tmpP(j, k, i) < 1e-4
                    Ph_XXX(:, k, i, j) = ones(m_h, 1) / m_h;
                else
                    Ph_XXX(:, k, i, j) = ph1 / sum(ph1);
                end
            end
        end

        tmp = Ph(:, i, n)' * tmpP(:, :, i);
        P_Xt_N(i, :, n) = tmp / sum(tmp);
    end

    P_Xt(:, n + 1) = (P_Xt(:, n)' * P_Xt_N(:, :, n))';

    Ph_XXh_h = zeros(m_h, m_x, m_x);
    for e = 1:m_x
        tmpP(:, :, e) = tmpP(:, :, e) .* repmat(Ph(:, e, n), 1, m_x);
    end

    for e = 1:m_h
        Ph_XXh_h = Ph_XXh_h + Ph_XXX(:, :, :, e) .* repmat(tmpP(e, :, :), m_h, 1, 1);
    end

    for d = 1:m_x
        tmp = P_Xt(:, n) / P_Xt(d, n + 1);
        sumtmp = zeros(m_h, 1);
        for e = 1:m_x
            sumtmp = sumtmp + Ph_XXh_h(:, d, e) .* tmp(e);
        end
        Ph(:, d, n + 1) = sumtmp;
    end

    tmpHt(:, n + 1) = Ph(:, :, n + 1) * P_Xt(:, n + 1);
end

end