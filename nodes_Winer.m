function nodes = nodes_Winer(T,N,z,r, sigma)
%
% construct a willow tree for standard Brownian motion with maturity T in N
% time steps
%
% gennerate the transition probabilities of a standard Winner process
%
% Input
%    T -- matuirty 
%    N -- number of time steps
%    z -- m*1-vector of normal distribution poin [z1,z2,....,zm], z1<z2...<zm
%    r -- interest rate
%   sigma -- volatility of stock price
%
%  Output
%    nodes -- tree nodes of the standard Brownian motion  m*N
%
m=length(z);
dt=T/N;
tt=linspace(dt,T,N);
z = z(:);

t = repmat(tt,m,1);
nodes = (r-sigma^2/2).*t + sigma*sqrt(t).*repmat(z,1,N);

end

