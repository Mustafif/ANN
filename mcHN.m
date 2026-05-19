function [S, h0] = mcHN(M, N,S0, Z, r, omega, alpha, beta, gamma, lambda)
dt = 1/N; 
numPoint = N+1;
S = zeros(N+1, M);
ht = zeros(N+1, M);
h0 = (omega + alpha)/(1-beta-alpha*gamma^2);
%h0 = (0.2^2)/N;
ht(1, :) = h0;
S(1, :) = S0;
for i = 2:numPoint
    ht(i,:) = omega + alpha * (Z(i-1,:) - gamma * sqrt(ht(i-1,:))).^2 + beta * ht(i-1,:);
    Xt = r * dt + lambda * ht(i, :) + sqrt(ht(i, :)) * Z(i, :);
    S(i, :) = S(i-1, :) * exp(Xt);
end

end