function e = en_vibr_0

format long e
global c h om_e om_x_e sw_o

if sw_o == 1
    e = h*c*(0.5*om_e-0.25*om_x_e);
else
    e = h*c*0.5*om_e; 
end
