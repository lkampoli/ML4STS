function in_con()

n = Sym("n")
v = Sym("v")
t = Sym("t")
n, v, t = symbols("n, v, t", real=true)

C1 = v0^2*sum(xc.*m)/(k*T0)
C2 = 0.5*C1;

S = [n*v == 1, n*t+n*v^2*C1 == 1+C1, 3.5*sum(xc[1:3])*t+2.5*sum(xc[4:5])*t+C2*v^2 == 3.5*sum(xc[1:3])+2.5*sum(xc[4:5])+C2]
N = SymPy.solve(S, n, v, t)

sol1 = collect(values(N[2]))
sol2 = collect(values(N[1]))

#if any(sol1-1)
#    Y = sol1;
#elseif any(sol2-1)
#    Y = sol2;
#end

return sol1

end
