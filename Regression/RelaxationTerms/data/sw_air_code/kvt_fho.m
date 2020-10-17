function kvt = kvt_fho(AB, C, t, i, f)
% программа расчета коэффициентов скорости VT обмена
% соотношения осредненных к.с. Adamovich et al. 1998 (упрощенные формулы)
% внесены поправки согласно коду fho.f
% AB(i) + C = AB(f) + C

global k h_bar om_e om_x_e m ram_masses r0;

if i == f
   error('Error. The same states.')
end

p1 = AB;
p2 = C;

% спектроскопические постоянные AB, K
ome = om_e(p1) * 1.4388e-2;
omexe = om_x_e(p1) * 1.4388e-2;

Ei = ome*i - omexe*i*(i+1);
Ef = ome*f - omexe*f*(f+1);

% масса атомов в AB и партнера, кг
mab = m(p1); mc = m(p2);
ma = mab * ram_masses(p1,1);
mb = mab * ram_masses(p1,2);

% приведенная масса AB-C, кгs
mu = (ma+mb)*mc/(ma+mb+mc);

% параметры модели
alpha = 4e10;               % м^-1
depth = 200.0*k;            % Дж
mpar = ma*mc/mb/(ma+mb+mc); 
Svt = 2*mpar/(1+mpar)^2; % Svt = 1/pi;

% газокинетический радиус упругого столкновения AB-C
R0 = r0(p1,p2);     % m

% сечение столкновения
sigma = pi*R0^2;	% модель твердых сфер

% частота столкновений, m^3/sec
Z = sigma*sqrt(8*k*t/(pi*mu));

if p2 <= 3
    % столкновение молекула-молекула
    nu = 1;
else
    % столкновение молекула-атом
    nu = 0; 
end

delE = Ei-Ef;
theta = abs(delE/(i-f));            % K
om = theta*k/h_bar;                  % sec^-1
theta1 = 4*pi^2*om^2*mu/alpha^2/k;  % K

s = abs(i-f);
sf = factorial(s);
ns = (factorial(max(i,f))/factorial(min(i,f)))^(1/s);

% эффективная скорость ЛТ
vm0 = (2.*pi*om*s*k*t/alpha/mu)^(1./3.);
%
const = 1/s * (nu+2*ns^(1/s)/(s+1))*Svt*theta1/theta;
const2 = (ns^(1/s)/(s+1)*Svt*theta1/theta)^2;
% fun = @(x) x-nthroot(1-const*exp(-2*pi*om./(alpha*vm0*x)), 3);
fun = @(x) x-nthroot(1-const*exp(-2*pi*om./(alpha*vm0*x))-...
    const2*exp(-4*pi*om./(alpha*vm0*x)), 3);
x0 = 2;
Cvt = fzero(fun,x0);

delta = (1-Cvt^3)/Cvt^3 * 2*pi*om/alpha/vm0/Cvt;
phi = 2/pi * atan(sqrt(2*depth/mu)/vm0);

rate = ns*sqrt(2*pi/(3+delta))*s^(1/3)/sf^2; 
rate = rate * Cvt * (Svt*theta1/theta)^s * (theta1/t)^(1/6);
rate = rate*exp(-s^(2/3)*(theta1/t)^(1/3)*(0.5*Cvt^2+1/Cvt)*...
    (1-phi)^(2/3)-s*(1-Cvt^3));
rate = rate * exp(theta*s/2/t);
kvt = rate * Z;     % m^3/sec

