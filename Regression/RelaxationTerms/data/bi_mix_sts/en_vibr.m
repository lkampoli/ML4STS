function e = en_vibr
% функция расчета колебательной энергии молекулы
% в расчете от нулевого уровня
format long e
global c h om_e om_x_e l sw_o

om_0 = om_e-om_x_e;
om_x_e_0 = om_x_e;
if sw_o == 1
    e = h*c*(om_0*(0:l-1)'-om_x_e_0*((0:l-1)').^2); % анг.о.
else
    e = h*c*om_e*(0:l-1)'; % г.о.
end
