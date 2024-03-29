function dy = rpart_ML(t,y)

% add the current folder to the Python search path
% https://www.mathworks.com/help/matlab/matlab_external/call-user-defined-custom-module.html
if count(py.sys.path,'') == 0
    insert(py.sys.path,int32(0),'');
end

format long e
global c h k m l e_i e_0 Be D n0 v0 T0 Delta

Lmax = l-1;

ni_b = y(1:l);
na_b = y(l+1);
nm_b = sum(ni_b);
v_b = y(l+2);
T_b = y(l+3);

temp = T_b*T0;
temperature = temp;

velocity = v_b*v0;

% ef_b = 0.5*D/T0;
% ei_b = e_i/(k*T0);
% e0_b = e_0/(k*T0);
% 
% sigma = 2;
% Theta_r = Be*h*c/k;
% Z_rot = temp./(sigma.*Theta_r);
% 
% M = sum(m);
% mb = m/M;

% A = zeros(l+3,l+3);
% 
% for i=1:l
%     A(i,i) = v_b;
%     A(i,l+2) = ni_b(i);
% end
% 
% A(l+1,l+1) = v_b;
% A(l+1,l+2) = na_b;
% 
% for i=1:l+1
%     A(l+2,i) = T_b;
% end
% 
% A(l+2,l+2) = M*v0^2/k/T0*(mb(1)*nm_b+mb(2)*na_b)*v_b;
% A(l+2,l+3) = nm_b+na_b;
% 
% for i=1:l
%     A(l+3,i) = 2.5*T_b+ei_b(i)+e0_b;
% end
% 
% A(l+3,l+1) = 1.5*T_b+ef_b;
% A(l+3,l+2) = 1/v_b*(3.5*nm_b*T_b+2.5*na_b*T_b+sum((ei_b+e0_b).*ni_b)+ef_b*na_b);
% A(l+3,l+3) = 2.5*nm_b+1.5*na_b;
% 
% AA = sparse(A);
% 
% B = zeros(l+3,1);

%Kdr = (m(1)*h^2/(m(2)*m(2)*2*pi*k*temp))^(3/2)*Z_rot* exp(-e_i'/(k*temp))*exp(D/temp);

%Kvt = exp((e_i(1:end-1)-e_i(2:end))/(k*temp))';

%kd = kdis(temp) * Delta*n0/v0;

%kr = zeros(2,l);
%for iM = 1:2
%   kr(iM,:) = kd(iM,:) .* Kdr * n0;
%end

%% VT i+1 -> i
%kvt_down = kvt_ssh(temp) * Delta*n0/v0;
%kvt_up = zeros(2,Lmax);
%for ip = 1:2
%    kvt_up(ip,:) = kvt_down(ip,:) .* Kvt;
%end

%% VV
%kvv_down = kvv_ssh(temp) * Delta*n0/v0;
%kvv_up = zeros(Lmax,Lmax);
%deps = e_i(1:end-1)-e_i(2:end);
%for ip = 1:Lmax
%    kvv_up(ip,:) = kvv_down(ip,:) .* exp((deps(ip)-deps') / (k*temp));
%end

%RD = zeros(l,1);
%RVT = zeros(l,1);
%RVV = zeros(l,1);

% Instead of the for loop, we call the ML model to predict 
% the full vector of RD, for example, given as input the following args:
% [x_s, time_s, Temp, rho, p, v, E, ni_n, na_n] or
% more properly a subset of features, namely:
% [Temp, v, ni_n, na_n]

%RD_ML = zeros(l,1);
%RD_MLd = zeros(l,1);
%input = [temperature; velocity; ni_b/(nm_b+na_b); na_b/(nm_b+na_b)];
%disp(input')
%pause(10)
%input = [temperature; velocity; ni_b*n0; na_b*n0];
%RD_ML = py.run_regression.regressor(input);
%RD_MLd = double(RD_ML);

%disp(RD_MLd)
%pause(10)

%for i1 = 1:l
%  RD(i1) = nm_b*(na_b*na_b*kr(1,i1)-ni_b(i1)*kd(1,i1)) + na_b*(na_b*na_b*kr(2,i1)-ni_b(i1)*kd(2,i1));
%end

%disp("RD")
%fprintf('%d\n',RD);

%disp("RD_MLd")
%fprintf('%d\n',RD_MLd);

    % fprintf('%i, %d, %d, %d, %d %d, %d\n', i1, RD(i1), ni_b(i1), nm_b, na_b, kr(1,i1), kd(1,i1))

%    if i1 == 1 % 0<->1
%
%        RVT(i1) = nm_b*(ni_b(i1+1)*kvt_down(1,i1) - ni_b(i1)*kvt_up(1,i1))+...
%                  na_b*(ni_b(i1+1)*kvt_down(2,i1) - ni_b(i1)*kvt_up(2,i1));
%
%        RVV(i1) = ni_b(i1+1)*sum(ni_b(1:end-1) .* kvv_down(i1,:)') - ...
%                  ni_b(i1)  *sum(ni_b(2:end)   .* kvv_up(i1,:)');
%
%    elseif i1 == l % Lmax <-> Lmax-1
%
%        RVT(i1) = nm_b*(ni_b(i1-1)*kvt_up(1,i1-1) - ni_b(i1)*kvt_down(1,i1-1))+...
%                  na_b*(ni_b(i1-1)*kvt_up(2,i1-1) - ni_b(i1)*kvt_down(2,i1-1));
%
%        RVV(i1) = ni_b(i1-1)*sum(ni_b(2:end)   .* kvv_up(i1-1,:)') - ...
%                  ni_b(i1)  *sum(ni_b(1:end-1) .* kvv_down(i1-1,:)');
%
%    else
%
%        RVT(i1) = nm_b*(ni_b(i1+1)*kvt_down(1,i1)+ni_b(i1-1)*kvt_up(1,i1-1)-...
%                  ni_b(i1)*(kvt_up(1,i1)+kvt_down(1,i1-1)))+...
%                  na_b*(ni_b(i1+1)*kvt_down(2,i1)+ni_b(i1-1)*kvt_up(2,i1-1)-...
%                  ni_b(i1)*(kvt_up(2,i1)+kvt_down(2,i1-1)));
%
%        RVV(i1) = ni_b(i1+1)*sum(ni_b(1:end-1) .* kvv_down(i1,:)') + ...
%                  ni_b(i1-1)*sum(ni_b(2:end) .* kvv_up(i1-1,:)') - ...
%                  ni_b(i1)*(sum(ni_b(2:end) .* kvv_up(i1,:)') + ...
%                            sum(ni_b(1:end-1) .* kvv_down(i1-1,:)'));
%    end
%end

%input = [ni_b; na_b; velocity; temperature;];
%RD_ML = py.run_regression_RD.regressor(input);
%RD_MLd = double(RD_ML);
%disp(size(RD_MLd))

%B(1:l) = RD_MLd(1:l); % + RVT + RVV;
%B(l+1) = - 2*sum(RD_MLd); %- 2*sum(RD);

input = [ni_b; na_b; velocity; temperature;];
%RD_ML = py.run_regression_dy.regressor(input);
RD_ML = py.run_regression.regressor(input);
RD_MLd = double(RD_ML);
%disp(size(RD_MLd))
dy = RD_MLd';

%dy = AA^(-1)*B;