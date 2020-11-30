function k_ex_savelev_st(T)

# constants for N2,O2
# A = [3e-17 4e-16]; % m^3/sec
A = [8e-17 4e-16]; # m^3/sec
b = [0 -0.39];
U = Inf; # 3*T;

ln2 = length(en2_i);
lo2 = length(eo2_i);
lno = length(eno_i);

ear_n2   = zeros(lno)
ear_n2_J = zeros(lno)
Ear_o2   = zeros(lno)
Ear_o2_J = zeros(lno)

# vibr. energy, J
en2 = en2_i .+ en2_0;
eo2 = eo2_i .+ eo2_0;
eno = eno_i .+ eno_0;
eno_eV = eno .* 6.242e18; # eV

Ear_n2 = 2.8793 .+ 1.02227 .* eno_eV; # eV
Ear_n2_J = Ear_n2 ./ 6.242e18; # J

# TODO: write as loop
for i = 1:lno
  if eno_eV[i] < 1.3706
    Ear_o2[i] = 0.098
  elseif eno_eV[i] > 1.3706 && eno_eV[i] < 2.4121
    Ear_o2[i] = -0.6521+0.54736*eno_eV[i]
  elseif eno_eV[i] > 2.4121
    Ear_o2[i] = -1.8451+1.04189*eno_eV[i]
  else
    println("Something wrong with energy threshold level!" )
  end
end
#Ear_o2 = 0.098.^(eno_eV .< 1.3706) .* (-0.6521 .+0.54736 .*eno_eV).^((1.3706 .< eno_eV) & (eno_eV .< 2.4121)) .*
#                                                         (-1.8451 .+ 1.04189 .*eno_eV).^(eno_eV .> 2.4121); # eV
Ear_o2_J = Ear_o2./6.242e18; # J

# equilibrium coefficient
k_eq_n2 = A[1] .* (1 .+eno_eV./3) .* T^b[1] .* exp.(-Ear_n2_J ./ (k*T)); # m^3/sec
k_eq_o2 = A[2] .* (Ear_o2 .+ 0.8) .* T^b[2] .* exp.(-Ear_o2_J ./ (k*T)); # m^3/sec

# vibr. partial function
Zv_n2 = sum(exp.(-en2./(k*T)));
Zv_o2 = sum(exp.(-eo2./(k*T)));

# energy threshold, for each k -> e_i*
sum1_n2 = zeros(lno);
sum2_n2 = zeros(lno);
sum1_o2 = zeros(lno);
sum2_o2 = zeros(lno);
#for ix = 1:lno
#  i_sw_n2 = find(en2 < Ear_n2_J[ix],1,"last");
#  i_sw_o2 = find(eo2 < Ear_o2_J[ix],1,"last");
#  sum1_n2[ix] = sum(exp(-(Ear_n2_J[ix]-en2[1:i_sw_n2])/(k*U)));
#  sum2_n2[ix] = sum(exp((Ear_n2_J[ix]-en2[i_sw_n2+1:end])/(k*T)));
#  sum1_o2[ix] = sum(exp(-(Ear_o2_J[ix]-eo2[1:i_sw_o2])/(k*U)));
#  sum2_o2[ix] = sum(exp((Ear_o2_J[ix]-eo2[i_sw_o2+1:end])/(k*T)));
#end
for i = 1:lno
  for j = 1:ln2
    if en2[j] < Ear_n2_J[i]
      global i_sw_n2 = j
    end
  end
  sum1_n2[i] = sum(exp.(-(Ear_n2_J[i].-en2[1:i_sw_n2]) ./ (k*U)))
  sum2_n2[i] = sum(exp.( (Ear_n2_J[i].-en2[i_sw_n2+1:end])./(k*U)))
end
for i = 1:lno
  for j = 1:lo2
    if eo2[j] < Ear_o2_J[i]
      global i_sw_o2 = j
    end
  end
  sum1_o2[i] = sum(exp.(-(Ear_o2_J[i].-eo2[1:i_sw_o2]) ./ (k*U)))
  sum2_o2[i] = sum(exp.( (Ear_o2_J[i].-eo2[i_sw_o2+1:end])./(k*U)))
end

# normalizing coefficients
C_n2 = Zv_n2 * (sum1_n2.+sum2_n2).^(-1);
C_o2 = Zv_o2 * (sum1_o2.+sum2_o2).^(-1);
B1_n2 = C_n2 .* k_eq_n2 .* exp.(-Ear_n2_J./(k*U));
B2_n2 = C_n2 .* k_eq_n2 .* exp.(Ear_n2_J ./(k*T));
B1_o2 = C_o2 .* k_eq_o2 .* exp.(-Ear_o2_J./(k*U));
B2_o2 = C_o2 .* k_eq_o2 .* exp.(Ear_o2_J ./(k*T));

kf_n2 = zeros(ln2,lno);
kf_o2 = zeros(lo2,lno);
#for ix = 1:ln2
#  kf_n2[ix,:] = (B1_n2'*exp(en2[ix]/k*(1/T+1/U))).^(en2[ix] < Ear_n2_J').* B2_n2'.^(en2[ix] > Ear_n2_J');
#end
#for iy = 1:lo2
#  kf_o2[iy,:] = (B1_o2'*exp(eo2[iy]/k*(1/T+1/U))).^(eo2[iy] < Ear_o2_J').* B2_o2'.^(eo2[iy] > Ear_o2_J');
#end
for i = 1:ln2
  for j = 1:lno
    if en2[i] < Ear_n2_J[j]
      kf_n2[i,j] = (B1_n2[j]*exp(en2[i]/k*(1/T+1/U)))
    elseif en2[i] > Ear_n2_J[j]
      kf_n2[i,j] = B2_n2[j]
    else
      println("As usual, shit happens!")
    end
  end
end
for i = 1:lo2
  for j = 1:lno
    if eo2[i] < Ear_o2_J[j]
      kf_o2[i,j] = (B1_o2[j]*exp(eo2[i]/k*(1/T+1/U)))
    elseif eo2[i] > Ear_o2_J[j]
      kf_o2[i,j] = B2_o2[j]
    else
      println("As usual, shit happens!")
    end
  end
end

return [kf_n2, kf_o2]

end
