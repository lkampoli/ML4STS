function en_vibr()

om_0     = om_e-om_x_e
om_x_e_0 = om_x_e

if sw_o == 1
  #e = h*c*(om_0*(0:l-1)'-om_x_e_0*((0:l-1)').^2)
  e = h*c*(om_0*(0:l-1)-om_x_e_0*((0:l-1)).^2)
else
  #e = h*c*om_e*(0:l-1)'
  e = h*c*om_e*(0:l-1)
end
return e
end
