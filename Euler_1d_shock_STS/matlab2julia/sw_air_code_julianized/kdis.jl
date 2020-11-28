function kdis(t,sp)

# equil. coef-s
kd_eq = CA[sp,:].*t.^nA[sp,:]*exp(-D[sp]/t); # m^3/sec

# parameter of TM model
if sw_u == "D/6k"
  U = D[sp]/6;
elseif sw_u == "3T"
  U = 3*t;
elseif sw_u == "Inf"
  U = Inf;
else
  disp("Error! Check switch on parameter U.");
  return;
end

# U = Inf;

#equil. vibr. partition function
if sp == 1

  ZvT = sum(exp.(-en2_i./(t*k)));
  ZvU = sum(exp.(en2_i./(U*k)));

  # non-equilibrium factor
  Z = ZvT / ZvU * exp.(en2_i./k*(1/t + 1/U));

elseif sp == 2

    ZvT = sum(exp.(-eo2_i./(t*k)));
    ZvU = sum(exp.(eo2_i./(U*k)));

    #non-equilibrium factor
    Z = ZvT / ZvU * exp.(eo2_i./k*(1/t + 1/U));

elseif sp == 3

    ZvT = sum(exp.(-eno_i./(t*k)));
    ZvU = sum(exp.(eno_i./(U*k)));

    # non-equilibrium factor
    Z = ZvT / ZvU * exp.(eno_i./k*(1/t + 1/U));

end

# dis. rates
kd = kd_eq .* Z'; # m^3/sec
#kd = kd_eq' .* Z'; # m^3/sec
# kd = Z';
#
end
