function kdis(temp)

kd_eq = CA.*temp.^nA*exp(-D/temp); print("kd_eq = ", kd_eq, "\n")
ZvT   = sum(exp.(-e_i./(temp*k))); print("ZvT = ",   ZvT,   "\n")

U = D/6; print("U = ", U, "\n")
# U = 3*temp
# U = Inf

ZvU = sum(exp.(e_i./(U*k)))                  ; print("ZvU = ", ZvU, "\n")
Z   = ZvT / ZvU * exp.(e_i./k*(1/temp + 1/U)); print("Z = ",   Z,   "\n")
kd  = kd_eq .* Z'                            ; print("kd = ",  kd,  "\n")

return kd
end
