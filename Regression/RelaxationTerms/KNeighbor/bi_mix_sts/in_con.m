function Y = in_con

global k xc m v0 T0

syms n v t

xm = xc(1);
xa = xc(2);

C1 = sum(m.*xc)*v0^2/(k*T0);
C2 = 0.5*C1;

S = [n*v == 1,...
     n*t+n*v^2*C1 == 1+C1,...
     3.5*xm*t+2.5*xa*t+v^2*C2 == 3.5*xm+2.5*xa+C2];

N = vpasolve(S,[n,t,v],[10,100,0.1]);
X = [N.n;N.t;N.v];
sol = double(X);

sol1 = sol([1 3 5]);
sol2 = sol([2 4 6]);
if any(sol1-1)
    Y = sol1;
elseif any(sol2-1)
    Y = sol2;
end