function [P,q]=gen_PoWiner(T,N,z)
% gennerate the transition probabilities of a standard Winner process
%
% Input
%    T -- matuirty
%    N -- number of time steps
%    z -- m*1-vector of normal distribution poin [z1,z2,....,zm], z1<z2...<zm
%
%  Output
%    P -- transition probability matrices from t1 to tN, m*m*N
%    q -- transition probability vector from t0 to t1
%
m=length(z);
dt=T/N;
tt=linspace(dt,T,N);
Yt=z'*sqrt(tt);% (n))*z(i);Yt size m*N
P=zeros(m,m,N-1);

sigma=sqrt(dt);
C=zeros(m+1,1);

for i = 2:m
    C(i) = (Yt(i-1,1)+Yt(i,1))/2;
end

C(1)=-Inf;
C(m+1)=Inf;
NF=normcdf(C,0,sigma);
q=NF(2:m+1)-NF(1:m);


for n=1:N-1

    for i = 2:m
        C(i) = (Yt(i-1,n+1)+Yt(i,n+1))/2;
    end

    for i=1:m
        mu=Yt(i,n);

        NF=normcdf(C,mu,sigma);
        P(i,:,n)=NF(2:m+1)-NF(1:m);

        % P(i,:,n)=probcali(P(i,:,n),Yt(:,n+1),Yt(i,n),sigma);
        P(i,:,n)=P(i,:,n)/sum(P(i,:,n));

    end
end
