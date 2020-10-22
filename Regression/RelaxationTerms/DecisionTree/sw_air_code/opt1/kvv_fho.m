function kvv = kvv_fho(AB, CD, t, i1, i2, f1, f2)
% программа расчета коэффициентов скорости VV обмена
% на основе кода fho.f
% AB(i1) + CD(i2) = AB(f1) + CD(f2)

global k h_bar om_e om_x_e m r0

p1 = AB;
p2 = CD;

% спектроскопические постоянные AB, K
ome_ab = om_e(p1) * 1.4388e-2;
omexe_ab = om_x_e(p1) * 1.4388e-2;
ome_cd = om_e(p2) * 1.4388e-2;
omexe_cd = om_x_e(p2) * 1.4388e-2;

% колебательная энергия частиц до и после столкновения, К
Ei1 = ome_ab*i1 - omexe_ab*i1*(i1+1);
Ei2 = ome_cd*i2 - omexe_cd*i2*(i2+1);
Ef1 = ome_ab*f1 - omexe_ab*f1*(f1+1);
Ef2 = ome_cd*f2 - omexe_cd*f2*(f2+1);

% масса атомов в AB и партнера, кг
mab = m(p1); mcd = m(p2);

% приведенная масса AB-CD, кг
mu = mab*mcd/(mab+mcd);

% газокинетический радиус упругого столкновения, м
R0 = r0(p1,p2);

% сечение столкновения, м^2
sigma = pi*R0^2;

% частота столкновений, м^3/сек
Z = sigma*sqrt(8*k*t/(pi*mu));

% параметр потенциала Морзе, м^-1
alpha = 4e10;

Svv = 1/16;

s = abs(i1-f1);
sf = factorial(s);
ns1 = (factorial(max(i1,f1))/factorial(min(i1,f1)))^(1/s);
ns2 = (factorial(max(i2,f2))/factorial(min(i2,f2)))^(1/s);

delE1 = Ei1-Ef1;
delE2 = Ei2-Ef2;

if i1 == f1
    th1 = ome_ab*(1.-2.*(omexe_ab/ome_ab)*i1);  % K
    z1 = 0;
 else
    th1 = abs(delE1/(i1-f1));                   % K
    z1 = 1;
end
if i2 == f2
    th2 = ome_cd*(1.-2.*(omexe_cd/ome_cd)*i2);  % K
    z2 = 0;
 else
    th2 = abs(delE2/(i2-f2));                   % K
    z2 = 1;
end

if (z1 ~= 0) || (z2 ~= 0)
    Evib = (th1*z1+th2*z2)/(z1+z2);             % K
end
if (z1 == 0) && (z2 == 0) 
    Evib = 0.5*(th1+th2);                       % K
end

Omega = Evib*k/h_bar; % сек^-1

% вероятность перехода (1,0)->(0,1)
P1001 = Svv*alpha^2*k*t/(2*Omega^2*mu);

% Resonance defect correction (Keck, Carrier, Adamovich)
delE = delE1+delE2;
lambda = 2/3*sqrt(4*pi^2*mu*Omega^2/(alpha^2*k*t)) * ...
    abs(delE)/(Evib*sqrt(8)*s);
G = 0.5*(3-exp(-2/3 * lambda))*exp(-2/3 * lambda);

kvv = Z * (ns1*ns2*P1001)^s/sf/(1+2*ns1*ns2*P1001/(s+1))^(s+1) * G; % м^3/сек