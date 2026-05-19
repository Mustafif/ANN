function [z,q,vzq,kzq] = zq(M,gamma)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% [z,q,varz,kurtosis] = zq(M,gamma)
%
%   Introduction
%                 Given an even number M and a gamma belonging to (0,1), 
%                 this function generates the discrete density distribution function {z,q}.
%
%  Input
%         M (an even scalar)  : number of spatial nodes at each time step;
%         gamma (a scalar)    : an adjustable factor between 0 and 1.
%
%   Output
%         z  (1*N vector)    :  a row vactor with M entities , z of the function {z,q};
%         q  (1*N vector)   :  also a row vactor with M entities, the probablities of z;
%         vzq (a scalar)       :  the variance of {z,q};
%         kzq  (a scalar)      :  the kurtosis of {z,q}.
%
%   References
%             1. W.Xu, Z.Hong, and C.Qin, A new sampling strategy willow tree method with application
%                  to path-dependent option pricing, 2013.
%
%   Implemented by
%           G.Wang  2016.12.15.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin < 2
    error('Pay attention to your input arguments!');
end
if mod(M,2) ~= 0 || M < 0
    error('M should be positive and even');
end
% if gamma < 0 || gamma > 1;
%     error('gamma should be a number between 0 and 1');
% end
I = 1:M/2;
q = zeros(1,M);
q(1:M/2) = (I-0.5).^gamma/M;
qsum = sum(q(1:M/2));
q(1:M/2) = q(1:M/2)/qsum/2;
q(end:-1:M/2+1) = q(1:M/2);
z0 = zeros(1,M);
z0(1) = norminv(q(1)/2);
for i = 2:M
    z0(i) = norminv(sum(q(1:i))-(sum(q(1:i))-sum(q(1:i-1)))/2);
end
z = z0;
a = q(1) + q(end);
b = 2*(q(end)*z0(end) - q(1)*z0(1));
c = q*z0.^2' - 1;
x = (-b+sqrt(b^2-4*a*c))/2/a;
z(1) = z0(1) - x;
z(2:M-1) = z0(2:M-1);
z(M) = z0(M) + x;

% a = (3-q*z.^4')/2/(z(1)^2 - z(M/2)^2)/(z(1)^2-z(2)^2);
% b = (3-q*z.^4')/2/(z(2)^2 - z(M/2)^2)/(z(2)^2-z(1)^2);
tmp = 1.5 - sum(q(1: M / 2) .* z(1: M / 2) .^ 4);
a = tmp / (z(1) ^ 2 - z(2) ^ 2) / (z(1) ^ 2 - z(M / 2) ^ 2);
b = tmp / (z(2) ^ 2 - z(1) ^ 2) / (...
    z(2) ^ 2 - z(M / 2) ^ 2);

q(1) = q(1) + a;
q(2) = q(2) + b;
q(M/2) = q(M/2) - a - b;
q(M:-1:M/2+1) = q(1:M/2);
vzq = q*z.^2';
kzq = q*z.^4';