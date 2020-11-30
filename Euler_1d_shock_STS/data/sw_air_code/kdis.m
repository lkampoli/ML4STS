function kd = kdis(t,sp)
% calculation of dissociation rates, collision w mol/at 
% Treanor-Marrone model
% t - temperature
% sp - molecule species

global k en2_i eo2_i eno_i CA nA D sw_u

% equil. coef-s
kd_eq = CA(sp,:).*t.^nA(sp,:)*exp(-D(sp)/t);% m^3/sec
% parameter of TM model
if  strcmp(sw_u,'D/6k')
    U = D(sp)/6;
elseif strcmp(sw_u,'3T')
    U = 3*t;
elseif strcmp(sw_u,'Inf')
    U = Inf;
else
    disp('Error! Check switch on parameter U.');
    return;
end
% U = Inf;
% equil. vibr. partition function
if sp == 1
    ZvT = sum(exp(-en2_i/(t*k)));
    ZvU = sum(exp(en2_i/(U*k)));
    % non-equilibrium factor
    Z = ZvT / ZvU * exp(en2_i/k*(1/t + 1/U));
elseif sp == 2
    ZvT = sum(exp(-eo2_i/(t*k)));
    ZvU = sum(exp(eo2_i/(U*k)));
    % non-equilibrium factor
    Z = ZvT / ZvU * exp(eo2_i/k*(1/t + 1/U));
elseif sp == 3
    ZvT = sum(exp(-eno_i/(t*k)));
    ZvU = sum(exp(eno_i/(U*k)));
    % non-equilibrium factor
    Z = ZvT / ZvU * exp(eno_i/k*(1/t + 1/U));
end

% dis. rates
kd = kd_eq' * Z'; % m^3/sec
% kd = Z';

end