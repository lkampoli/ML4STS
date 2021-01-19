function kd = kdis(t)
% calculation of dissociation rates, collision w mol/at 
% Treanor-Marrone model
% t - temperature
% sw - switch of osc-or model (1 - anh.o./ 2 - har.o.)

global k e_i CA nA D

% equil. coef-s
kd_eq = CA.*t.^nA*exp(-D/t);% m^3/sec
% equil. vibr. partition function
ZvT = sum(exp(-e_i/(t*k)));
% parameter of TM model
U = D/6;
% U = 3*t;
% U = Inf;
ZvU = sum(exp(e_i/(U*k)));

% non-equilibrium factor
Z = ZvT / ZvU * exp(e_i/k*(1/t + 1/U));

% dis. rates
kd = kd_eq' * Z'; % m^3/sec
end