function kvt_ssh(t)

if sw_o == 1
  om10 = om_e-2*om_x_e;          println("om10 = ", om10, "\n")
elseif sw_o == 2
  om10 = om_e;                   println("om10 = ", om10, "\n")
end

Lmax = l-1;                      println("Lmax = ", Lmax, "\n")
i    = [0:Lmax-1];               println("i = ", i, "\n")
om0  = 2*pi*c*om10;              println("om0 = ", om0, "\n")
mu   = m[1] .* m ./ (m[1] .+ m); println("mu = ", mu, "\n")
a    = 17.5 ./ r0;               println("a = ", a, "\n")

Z        = r0.^2 / sqrt.(mu) * sqrt(8*pi*k*t);                     println("Z = ", Z, "\n")
chi      = (pi^2 * om0^2 /2/k*(mu ./ a.^2)*t^(-1)).^(1/3);         println("chi = ", chi, "\n")
#@. r     = r0 .* (0.5 * sqrt.(1 .+ (chi.*t) ./ em) + 0.5).^(-1/6); println("r = ", r, "\n")
#Z_0      = (a .* re).^2 .* exp.(-((3*a .* re.^2) ./ (8*r)));       println("Z_0 = ", Z_0, "\n")
#Const    = 1.294 ./ Z_0*4*pi^2*om0/h_bar*sqrt(4*pi/3);             println("Const = ", Const, "\n")
#r_c_r_02 = (0.5*sqrt(1+(chi*t) ./ em)+0.5).^(-1/3);                println("r_c_r_02 = ", r_c_r_02, "\n")
#P10      = Const .* r_c_r_02 .* (1+1.1*em/t).^(-1) .* mu ./ a.^2 .* chi.^0.5 .* exp.(-3*chi+h_bar*om0/(2*k*t)+em/t)
#k10      = Z .* P10;                                               println("k10 = ", k10, "\n")

if sw_o == 2

#  k_down = k10*(i+1)

elseif sw_o == 1

  aA      = a * 1e-10
# mu_amu  = m[1]*m ./ (m[1].+m)/1.6605e-27
# diffE   = (om_e-2*(i+1)*om_x_e)*h*c
# E1      = (om_e-2*om_x_e)*1.4388e-2
# dE      = om_x_e*1.4388e-2
# gamma_n = pi ./ a / h_bar .* sqrt.(mu./(2*k*t))
# gamma_n = gamma_n * diffE
# gamma_0 = 0.32 ./ aA .* sqrt.(mu_amu/t)*E1
# delta   = zeros(2,length(i))
# delta[1,:] = (4/3*gamma_0[1] * dE/E1).^(gamma_n[1,:] < 20) .* (4*(gamma_0[1])^(2/3) * dE/E1).^(gamma_n[1,:] >= 20)
# delta[2,:] = (4/3*gamma_0[1] * dE/E1).^(gamma_n[1,:] < 20) .* (4*(gamma_0[1])^(2/3) * dE/E1).^(gamma_n[1,:] >= 20)
# k_down[1,:] = (i+1) * k10[1] .* exp(i .* delta[1,:]) .* exp(-i*h*c*om_x_e / (k*t))
# k_down[2,:] = (i+1) * k10[2] .* exp(i .* delta[2,:]) .* exp(-i*h*c*om_x_e / (k*t))

end

k_down = ones(2,46)
return k_down
end