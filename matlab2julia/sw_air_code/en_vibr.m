function e = en_vibr(sp)
% функция расчета колебательной энергии молекулы
% в расчете от нулевого уровня
format long e
global c h om_e om_x_e l sw_o

if sw_o == 1
    om_0 = om_e(sp)-om_x_e(sp);
    om_x_e_0 = om_x_e(sp);
    e = h*c*(om_0*(0:l(sp)-1)'-om_x_e_0*((0:l(sp)-1)').^2); % анг.о.
else
    e = h*c*om_e(sp)*(0:l(sp)-1)'; % г.о.
end
