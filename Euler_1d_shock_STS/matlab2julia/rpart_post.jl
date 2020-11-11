function [RDm, RDa, RVTm, RVTa, RVV] = rpart_post(y)

Lmax = l-1

ni_b = y(1:l)
na_b = y(l+1)
nm_b = sum(ni_b)
T_b  = y(l+3)

temp = T_b*T

sigma = 2
Theta_r = Be*h*c/k
Z_rot = temp./(sigma.*Theta_r)

Kdr = (m(1)*h^2/(m(2)*m(2)*2*pi*k*temp))^(3/2)*Z_rot* exp(-e_i'/(k*temp))*exp(D/temp)

Kvt = exp((e_i(1:end-1)-e_i(2:end))/(k*temp))

kd = kdis(temp) * Delta*n0/v0

kr = zeros(2,l)
for iM = 1:2
  kr(iM,:) = kd(iM,:) .* Kdr * n0
end

kvt_down = kvt_ssh(temp) * Delta*n0/v0
kvt_up = zeros(2,Lmax)
for ip = 1:2
  kvt_up(ip,:) = kvt_down(ip,:) .* Kvt
end

kvv_down = kvv_ssh(temp) * Delta*n0/v0
kvv_up = zeros(Lmax,Lmax)
deps = e_i(1:end-1)-e_i(2:end)
for ip = 1:Lmax
  kvv_up(ip,:) = kvv_down(ip,:) .* exp((deps(ip)-deps') / (k*temp))
end

RDm  = zeros(l)
RDa  = zeros(l)
RVTm = zeros(l)
RVTa = zeros(l)
RVV  = zeros(l)

for i1 = 1:l
    RDm(i1) = nm_b*(na_b*na_b*kr(1,i1)-ni_b(i1)*kd(1,i1))
    RDa(i1) = na_b*(na_b*na_b*kr(2,i1)-ni_b(i1)*kd(2,i1))
    if i1 == 1
        RVTm(i1) = nm_b*(ni_b(i1+1)*kvt_down(1,i1) - ni_b(i1)*kvt_up(1,i1))
        RVTa(i1) = na_b*(ni_b(i1+1)*kvt_down(2,i1) - ni_b(i1)*kvt_up(2,i1))
        RVV(i1) = ni_b(i1+1)*sum(ni_b(1:end-1) .* kvv_down(i1,:)') - ni_b(i1)*sum(ni_b(2:end) .* kvv_up(i1,:))

    elseif i1 == l
        RVTm(i1) = nm_b*(ni_b(i1-1)*kvt_up(1,i1-1) - ni_b(i1)*kvt_down(1,i1-1));
        RVTa(i1) = na_b*(ni_b(i1-1)*kvt_up(2,i1-1) - ni_b(i1)*kvt_down(2,i1-1));
        RVV(i1) = ni_b(i1-1)*sum(ni_b(2:end) .* kvv_up(i1-1,:)') - ...
                     ni_b(i1)*sum(ni_b(1:end-1) .* kvv_down(i1-1,:)');

    else
        RVTm(i1) = nm_b*(ni_b(i1+1)*kvt_down(1,i1)+ni_b(i1-1)*kvt_up(1,i1-1)-...
                               ni_b(i1)*(kvt_up(1,i1)+kvt_down(1,i1-1)));
        RVTa(i1) = na_b*(ni_b(i1+1)*kvt_down(2,i1)+ni_b(i1-1)*kvt_up(2,i1-1)-...
                               ni_b(i1)*(kvt_up(2,i1)+kvt_down(2,i1-1)));
        RVV(i1) = ni_b(i1+1)*sum(ni_b(1:end-1) .* kvv_down(i1,:)') + ...
                   ni_b(i1-1)*sum(ni_b(2:end) .* kvv_up(i1-1,:)') - ...
                     ni_b(i1)*(sum(ni_b(2:end) .* kvv_up(i1,:)') + ...
                                sum(ni_b(1:end-1) .* kvv_down(i1-1,:)'));
    end
end

RDm  = RDm  * n0*v0/Delta
RDa  = RDa  * n0*v0/Delta
RVTm = RVTm * n0*v0/Delta
RVTa = RVTa * n0*v0/Delta
RVV  = RVV  * n0*v0/Delta
