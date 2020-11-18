function kvt_ssh!(t)

if sw_o == 1
  om10 = om_e-2*om_x_e;          #println("om10 = ", om10, "\n")
elseif sw_o == 2
  om10 = om_e;                   #println("om10 = ", om10, "\n")
end

Lmax = l-1;                      #println("Lmax = ", Lmax, "\n")
i    = collect(0:Lmax-1);        #println("i = ", i, "\n", typeof(i), "\n", length(i), "\n")
om0  = 2*pi*c*om10;              #println("om0 = ", om0, "\n")
mu   = m[1] .* m ./ (m[1] .+ m); #println("mu = ", mu, "\n")
a    = 17.5 ./ r0;               #println("a = ", a, "\n")

Z        = r0.^2 ./ sqrt.(mu) * sqrt(8*pi*k*t);                        #println("Z = ", Z, "\n")
chi      = ((pi^2 * om0^2 / 2 / k * t^(-1)) .* (mu ./ a.^2)).^(1/3);   #println("chi = ", chi, "\n")
r        = r0 .* (0.5 .* sqrt.(1 .+ (chi .* t) ./ em) .+ 0.5).^(-1/6); #println("r = ", r, "\n")
Z_0      = (a .* re).^2 .* exp.(-((3 .* a .* re.^2) ./ (8 .* r)));     #println("Z_0 = ", Z_0, "\n")
Const    = 1.294 ./ Z_0 .* 4*pi^2*om0/h_bar*sqrt(4*pi/3);              #println("Const = ", Const, "\n")
r_c_r_02 = (0.5 .* sqrt.(1 .+ (chi.*t) ./ em) .+ 0.5).^(-1/3);         #println("r_c_r_02 = ", r_c_r_02, "\n")
P10      = Const .* r_c_r_02 .* (1 .+ 1.1 .* em ./ t).^(-1) .* mu ./ a.^2 .* chi.^0.5 .* exp.(-3 .* chi .+ h_bar*om0/(2*k*t) .+ em ./ t)
k10      = Z .* P10;                                                   #println("k10 = ", k10, "\n")

if sw_o == 2
  k_down = k10.*(i.+1);                             #println("k_down = ", k_down, "\n")
elseif sw_o == 1
  aA      = a .* 1e-10;                             #println("aA = ", aA, "\n")
  mu_amu  = m[1].*m ./ (m[1].+m)./1.6605e-27;       #println("mu_amu = ", mu_amu, "\n")
  E1      = (om_e-2*om_x_e)*1.4388e-2;              #println("E1 = ", E1, "\n")
  dE      = om_x_e*1.4388e-2;                       #println("dE = ", dE, "\n")
  # https://julialang.org/blog/2017/01/moredots/
  diffE = zeros(length(i))
  for ii in 1:length(i)
    println("i = ", ii, "\n")
    diffE[ii] = (om_e - 2 * om_x_e * (i[ii] + 1)) * h * c
  end
  println("diffE = ", diffE, "\n", size(diffE), "\n")
  gamma_n = pi ./ a ./ h_bar .* sqrt.(mu./(2*k*t));
  gamma_n = gamma_n .* diffE;                       println("gamma_n = ", gamma_n, "\n", size(gamma_n), "\n")
  gamma_n = gamma_n';                               println("gamma_n = ", gamma_n, "\n", size(gamma_n), "\n")
  gamma_0 = 0.32 ./ aA .* sqrt.(mu_amu./t).*E1;     println("gamma_0 = ", gamma_0, "\n")
  delta   = zeros(2,length(i));
  k_down  = zeros(2,length(i))
  for ii in 1:length(i)
    if gamma_n[1,ii] < 20
      delta[1,ii] = 4/3*gamma_0[1] * dE/E1
      delta[2,ii] = 4/3*gamma_0[1] * dE/E1
    elseif gamma_n[1,ii] >= 20
      delta[1,ii] = 4*(gamma_0[1])^(2/3) * dE/E1
      delta[2,ii] = 4*(gamma_0[1])^(2/3) * dE/E1
    end
  end
 println("delta = ", delta, "\n", size(delta), "\n")
  for ii in 1:length(i)
    k_down[1,ii] = (i[ii]+1) * k10[1] * exp(i[ii] * delta[1,ii]) * exp(-i[ii]*h*c*om_x_e / (k*t))
    k_down[2,ii] = (i[ii]+1) * k10[2] * exp(i[ii] * delta[2,ii]) * exp(-i[ii]*h*c*om_x_e / (k*t))
  end
end
println("k_down = ", k_down, "\n", size(k_down), "\n")
#return k_down
end
