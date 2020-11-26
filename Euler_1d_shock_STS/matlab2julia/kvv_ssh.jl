function kvv_ssh(t)

mu    = 0.5  * m[1]
m_osc = 0.25 * m[1]

if sw_o == 1
  om10 = om_e-2*om_x_e
elseif sw_o == 2
  om10 = om_e
end

Lmax = l-1
om0  = 2*pi*c*om10

a = 17.5 / r0[1]
Z = r0[1]^2 / sqrt(mu) * sqrt(8*pi*k*t)

Q10 = 0.5^4 / m_osc * a^2 / om0^2 * 4 * k * t
k10 = Q10 * Z

kdown = zeros(Lmax,Lmax)
j_up  = 0:Lmax-1
if sw_o == 2
  for i_down = 1:Lmax
    kdown[i_down,:] = i_down * (j_up+1) * k10
  end
#println("kdown = ", kdown, "\n", size(kdown), "\n")
elseif sw_o == 1
  aA     = a * 1e-10
  mu_amu = mu / 1.6605e-27
  dE     = om_x_e * 1.4388e-2
  delta  = 0.427 / aA * sqrt(mu_amu / t) * dE; println("delta = ", delta, "\n")
  for i_down = 1:Lmax
  @. kdown[i_down,:] = i_down * (j_up+1) * k10 .* exp.(-delta .* abs(i_down-1-j_up)) .*
    (1.5 - 0.5 .* exp.(-delta .* abs(i_down-1-j_up))) .* exp.((j_up-i_down+1) * h * c * om_x_e / (k * t))
  end
end
println("kdown = ", kdown, "\n", size(kdown), "\n")
#return kdown
end
