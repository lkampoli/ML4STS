function kdis!(temp)
kd_eq = CA .* temp.^nA*exp(-D/temp);             #print("kd_eq = ", kd_eq, "\n")
ZvT   = sum(exp.(-e_i ./ (temp*k)));             #print("ZvT = ",   ZvT,   "\n")
U = D/6.;                                        # print("U = ", U, "\n")
# U = 3*temp
# U = Inf
ZvU = sum(exp.(e_i ./ (U*k)));                   #print("ZvU = ", ZvU, "\n")
Z   = ZvT / ZvU * exp.(e_i ./ k*(1/temp + 1/U)); #print("Z = ",   Z,   "\n")
kvt_down = kd_eq .* Z';                               #print("kd = ",  kd,  "\n")
end


function kdis1!(temp)
kd_eq = CA .* temp.^nA*exp(-D/temp);
ZvT   = sum(exp.(-e_i ./ (temp*k)));
U = D/6.;
ZvU = sum(exp.(e_i ./ (U*k)));
Z   = ZvT / ZvU * exp.(e_i ./ k*(1/temp + 1/U));
kd  = kd_eq .* Z';
end


function kdis2!(temp)
@fastmath @inbounds kd_eq = CA .* temp.^nA*exp(-D/temp);
@fastmath @inbounds ZvT   = sum(exp.(-e_i ./ (temp*k)));
U = D/6.;
@fastmath @inbounds ZvU = sum(exp.(e_i ./ (U*k)));
@fastmath @inbounds Z   = ZvT / ZvU * exp.(e_i ./ k*(1/temp + 1/U));
@fastmath @inbounds kd  = kd_eq .* Z';
end


function kdis3!(temp::Float64, kd::Array{Float64})
kd_eq = CA .* temp.^nA*exp(-D/temp);
ZvT   = sum(exp.(-e_i ./ (temp*k)));
U = D/6.;
ZvU = sum(exp.(e_i ./ (U*k)));
Z   = ZvT / ZvU * exp.(e_i ./ k*(1/temp + 1/U));
kd  = kd_eq .* Z';
end


function kdis4(temp)
kd = zeros(2,l)
Kdr = (m[1]*h^2/(m[2]*m[2]*2*pi*k*temp))^(3/2)*Z_rot*exp.(-e_i/(k*temp))*exp(D/temp);
kd_eq = CA .* temp.^nA*exp(-D/temp);
ZvT   = sum(exp.(-e_i ./ (temp*k)));
U = D/6.
ZvU = sum(exp.(e_i ./ (U*k)));
Z   = ZvT / ZvU * exp.(e_i ./ k*(1/temp + 1/U));
kd  = kd_eq .* Z';
end
