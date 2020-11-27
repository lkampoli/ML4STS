function in_con()

n = Sym("n")
v = Sym("v")
t = Sym("t")
n, v, t = symbols("n, v, t", real=true)

xm = xc[1]
xa = xc[2]

C1 = sum(m.*xc)*v0^2/(k*T0)
C2 = 0.5*C1

S = [n*v ⩵ 1, n*t+n*v^2*C1 ⩵ (1+C1), 3.5*xm*t+2.5*xa*t+v^2*C2 ⩵ (3.5*xm+2.5*xa+C2)]
N = SymPy.solve(S, n, v, t)

sol1 = collect(values(N[2]))
sol2 = collect(values(N[1]))

#FIXME
#if any(sol1 != 0)
#  Y = sol1
#elseif any(sol2 != 0)
#  Y = sol2
#end

return sol1

end
