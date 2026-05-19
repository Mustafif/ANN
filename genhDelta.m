function [hd,q] = genhDelta(h0, beta, alpha, gamma, omega, mh, gamma_h)

% generate nodes of standard normal distribution
[z,q] = zq(mh, gamma_h);
z = z';
hd = omega+beta*h0+alpha*(z-gamma*sqrt(h0)).^2;
hd = sort(hd);
intPoints = (hd(1:end-1)+hd(2:end))/2;

numHt = length(hd);

upBound = gamma*sqrt(h0)+sqrt((intPoints-omega-beta*h0)/alpha);
lowBound = gamma*sqrt(h0)-sqrt((intPoints-omega-beta*h0)/alpha);

if alpha>0
    prob = normcdf(real(upBound))-normcdf(real(lowBound));
    prob = [0;prob;1];
    q = (diff(prob))';
elseif alpha<0  % 
    prob = 1-(normcdf(real(upBound))-normcdf(real(lowBound)));
    prob = [ones(1,numHt);prob;zeroes(1,numHt)];
    q = (prob(1:end-1,:)-prob(2:end,:))';
end
q = q(:);
% adjust the probability 
