function rpart!(du,u,p,t)
# https://flow.byu.edu/posts/julia-c++
# https://docs.julialang.org/en/v1/manual/integers-and-floating-point-numbers/
  ni_b = zeros(l);
  ni_b = u[1:l]; #print("ni_b = ", ni_b, "\n")
  na_b = u[l+1]; #print("na_b = ", na_b, "\n")
  v_b  = u[l+2]; #print("v_b = ",  v_b,  "\n")
  T_b  = u[l+3]; #print("T_b = ",  T_b,  "\n")

  nm_b = sum(ni_b);   #print("nm_b = ", nm_b, "\n")
  Lmax = l-1;         #println("Lmax = ", Lmax, "\n")
  temp = T_b*T0;      #print("T = ", temp, "\n")

  ef_b = 0.5*D/T0;    #println("ef_b = ", ef_b, "\n")

  ei_b = e_i./(k*T0); #println("ei_b = ", ei_b, "\n")
  e0_b = e_0/(k*T0);  #println("e0_b = ", e0_b, "\n")

  sigma   = 2.;                     #println("sigma = ", sigma, "\n")
  Theta_r = Be*h*c/k;               #println("Theta_r = ", Theta_r, "\n")
  Z_rot   = temp./(sigma.*Theta_r); #println("Z_rot = ", Z_rot, "\n")

  M  = sum(m); #println("M = ", M, "\n")
  mb = m/M;    #println("mb = ", mb, "\n")

  A = zeros(l+3,l+3)
  #spy(A, markersize=1, colorbar=false, color=:deep)

  for i = 1:l
    A[i,i]   = v_b
    A[i,l+2] = ni_b[i]
  end

  A[l+1,l+1] = v_b
  A[l+1,l+2] = na_b

  for i = 1:l+1
    A[l+2,i] = T_b
  end
  A[l+2,l+2] = M*v0^2/k/T0*(mb[1]*nm_b+mb[2]*na_b)*v_b
  A[l+2,l+3] = nm_b+na_b

  for i = 1:l
    A[l+3,i] = 2.5*T_b+ei_b[i]+e0_b
  end
  A[l+3,l+1] = 1.5*T_b+ef_b
  A[l+3,l+2] = 1/v_b*(3.5*nm_b*T_b+2.5*na_b*T_b+sum((ei_b.+e0_b).*ni_b)+ef_b*na_b)
  A[l+3,l+3] = 2.5*nm_b+1.5*na_b

  AA = inv(A); #println("AA = ", AA, "\n", size(AA), "\n")
  #AA = A; println("AA = ", AA, "\n", size(AA), "\n")
  #display(UnicodePlots.spy(AA))
  #spy(sparse(A), ms=5)
  #display(PyPlot.spy(A))
  #display(Plots.spy(A))

  # Equilibrium constant for DR processes
  Kdr = (m[1]*h^2/(m[2]*m[2]*2*pi*k*temp))^(3/2)*Z_rot*exp.(-e_i/(k*temp))*exp(D/temp); #println("Kdr = ", Kdr, "\n")

  # Equilibrium constant for VT processes
  Kvt = exp.((e_i[1:end-1]-e_i[2:end])/(k*temp)); #println("Kvt = ", Kvt, "\n")

  # Dissociation processes
  kd = zeros(2,l)
  kd = kdis(temp) * Delta*n0/v0;
  #println("kd = ", kd, "\n", size(kd), "\n")

  # Recombination processes
  kr = zeros(2,l)
  for iM = 1:2
    kr[iM,:] = kd[iM,:] .* Kdr * n0
  end
  #println("kr = ", kr, "\n", size(kr), "\n")

  # VT processes: i+1 -> i
  kvt_down = zeros(2,Lmax)
  kvt_up   = zeros(2,Lmax)
  kvt_down = kvt_ssh!(temp) * Delta*n0/v0; #println("kvt_down = ", kvt_down, "\n")
  for ip = 1:2
    kvt_up[ip,:] = kvt_down[ip,:] .* Kvt
  end
  #println("kvt_up = ", kvt_up, "\n", size(kvt_up), "\n")

  # VV processes
  kvv_down = zeros(Lmax,Lmax)
  kvv_up   = zeros(Lmax,Lmax)
  kvv_down = kvv_ssh!(temp) * Delta*n0/v0
  deps     = e_i[1:end-1]-e_i[2:end]
  for ip in 1:Lmax
  @. kvv_up[ip,:] = kvv_down[ip,:] * exp((deps[ip]-deps) / (k*temp))
  #@. kvv_up[ip,:] = kvv_down[ip,:] .* exp.((deps[ip].-deps) / (k*temp))
  end
  #println("kvv_up = ", kvv_up, "\n", size(kvv_up), "\n")

  RD  = zeros(l)
  RVT = zeros(l)
  RVV = zeros(l)

  @fastmath @inbounds for i1 in eachindex(ni_b)

    RD[i1] = nm_b*(na_b*na_b*kr[1,i1]-ni_b[i1]*kd[1,i1]) + na_b*(na_b*na_b*kr[2,i1]-ni_b[i1]*kd[2,i1])

      if i1 == 1 # 0<->1

        RVT[i1] = nm_b*(ni_b[i1+1]*kvt_down[1,i1]-ni_b[i1]*kvt_up[1,i1])+
                  na_b*(ni_b[i1+1]*kvt_down[2,i1]-ni_b[i1]*kvt_up[2,i1])

        RVV[i1] = ni_b[i1+1]*sum(ni_b[1:end-1] .* kvv_down[i1,:]) -
                  ni_b[i1]  *sum(ni_b[2:end]   .* kvv_up[i1,:])

      elseif i1 == l # Lmax <-> Lmax-1

        RVT[i1] = nm_b*(ni_b[i1-1]*kvt_up[1,i1-1]-ni_b[i1]*kvt_down[1,i1-1]) +
                  na_b*(ni_b[i1-1]*kvt_up[2,i1-1]-ni_b[i1]*kvt_down[2,i1-1])

        RVV[i1] = ni_b[i1-1]*sum(ni_b[2:end]   .* kvv_up[i1-1,:]) -
                  ni_b[i1]  *sum(ni_b[1:end-1] .* kvv_down[i1-1,:])

      else

        RVT[i1] = nm_b*(ni_b[i1+1] * kvt_down[1,i1] + ni_b[i1-1] * kvt_up[1,i1-1] -
                        ni_b[i1]   * (kvt_up[1,i1]  + kvt_down[1,i1-1])) +
                  na_b*(ni_b[i1+1] * kvt_down[2,i1] + ni_b[i1-1] * kvt_up[2,i1-1] -
                        ni_b[i1]   * (kvt_up[2,i1]  + kvt_down[2,i1-1]))

        RVV[i1] = ni_b[i1+1] * sum(ni_b[1:end-1]  .* kvv_down[i1,:]) +
                  ni_b[i1-1] * sum(ni_b[2:end]    .* kvv_up[i1-1,:]) -
                  ni_b[i1]   * (sum(ni_b[2:end]   .* kvv_up[i1,:]) +
                                sum(ni_b[1:end-1] .* kvv_down[i1-1,:]))
      end
  end

  #println("RD = ",  RD,  "\n", size(RD))
  #println("RVT = ", RVT, "\n", size(RVT))
  #println("RVV = ", RVV, "\n", size(RVV))

  B      = zeros(l+3)
  @fastmath @inbounds for i in eachindex(RD)
    B[i] = RD[i] + RVT[i] + RVV[i]
  end
  B[l+1] = - 2*sum(RD)
  #du    .= AA*B
  mul!(du,AA,B)

end
