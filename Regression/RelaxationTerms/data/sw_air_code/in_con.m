function [Y] = in_con
% ������� ������� ��������� ���������� ����� O2/N2 �� ��
% ����������� wurster (1991)

global k m v0 T0 xc; 
% n0, v0, T0 -- ��������� ����� ����� ��
% xc -- ������ �����

% ������� ��������� �� ������� ��
syms n v t

C1 = v0^2*sum(xc.*m)/(k*T0);
C2 = 0.5*C1;

S = [n*v == 1,...
     n*t+n*v^2*C1 == 1+C1,...
     3.5*sum(xc(1:3))*t+2.5*sum(xc(4:5))*t+C2*v^2 == ...
     3.5*sum(xc(1:3))+2.5*sum(xc(4:5))+C2];

N = vpasolve(S,[n,v,t],[10,0.1,100]);
X = [N.n;N.v;N.t];
sol = double(X);

% ����� �������������� �������
sol1 = sol([1 3 5]);
sol2 = sol([2 4 6]);
if any(sol1-1)
    Y = sol1;
elseif any(sol2-1)
    Y = sol2;
end
