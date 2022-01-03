function dy = rpart_fho(t,y)
%	������� ������� ������ ������ ��� ��
%	��������� �����
%   ��������������� ������ 1-��������� ������������

format long e
global c h k m l Be D n0 v0 T0 Delta sw_z;
global en2_i en2_0 eo2_i eo2_0 eno_i eno_0;

% ����� ������������� ��������� ������� N2-O2-NO
l1 = l(1); l2 = l(2); l3 = l(3);
lall = l1+l2+l3;

% ����� ������������� �������������� ������
Lmax1 = l1-1; Lmax2 = l2-1; Lmax3 = l3-1;

% ���������� � ������������ ����
nn2i_b = y(1:l1);
no2i_b = y(l1+1:l1+l2);
nnoi_b = y(l1+l2+1:lall);
nn_b = y(lall + 1);
no_b = y(lall + 2);
v_b = y(lall + 3);
T_b = y(lall + 4);

% �������� ��������� ������� N2-O2-NO
nn2_b = sum(nn2i_b);
no2_b = sum(no2i_b);
nno_b = sum(nnoi_b);

% ����� �������� ��������� ������� � ������
na_b = nn_b+no_b;
nm_b = sum(y(1:lall));

% ��������� �� �������, �
% xx = t*Delta;
% disp(xx)

% ���������� � ��������� ����
% ����������� �����, �
temp = T_b*T0;

% ������� ����������� ������ � ������� NO � ����. ����
efn_b = 0.5*D(1)/T0;
efo_b = 0.5*D(2)/T0;
efno_b = efn_b+efo_b-D(3)/T0;

% ������������� ������� ������� � ����. ����
en2i_b = en2_i/(k*T0);
en20_b = en2_0/(k*T0);
eo2i_b = eo2_i/(k*T0);
eo20_b = eo2_0/(k*T0);
enoi_b = eno_i/(k*T0);
eno0_b = eno_0/(k*T0);

% ������ ��������� ������� N2-O2-NO
sigma = [2 2 1];

% ������������������ ������������ ����������� ��������, �
Theta_r = Be*h*c/k;

% ������������� ��������� ������� N2-O2-NO
Z_rot = temp./(sigma.*Theta_r);

% ��������� ����� ������� � ������, ��
M = sum(m);

% ������������ ����� ������
mb = m/M;

%% ���������� ������� ������������� ��� ������� A*X=B
A = zeros(sum(l)+4,sum(l)+4);

% ������������ ��������� ��� �������
for i=1:l1 % N2
    A(i,i) = v_b;
    A(i,lall + 3) = nn2i_b(i);
end
for i=1:l2 % O2
    A(l1+i,l1+i) = v_b;
    A(l1+i,lall + 3) = no2i_b(i);
end
for i=1:l3 % NO
    A(l1+l2+i,l1+l2+i) = v_b;
    A(l1+l2+i,lall + 3) = nnoi_b(i);
end

% ������������ ��������� ��� ������
% N
A(lall+1,lall+1) = v_b;
A(lall+1,lall + 3) = nn_b;
% O
A(lall+2,lall+2) = v_b;
A(lall+2,lall + 3) = no_b;

% ��������� ���������� ��������
for i=1:lall+2
    A(lall+3,i) = T_b;
end
A(lall+3,lall+3) = M*v0^2/k/T0*(mb(1)*nn2_b+mb(2)*no2_b+mb(3)*nno_b+...
    mb(4)*nn_b+mb(5)*no_b)*v_b;
A(lall+3,lall+4) = nm_b+na_b;

% ��������� ���������� �������
for i=1:l1 % N2
    A(lall+4,i) = 2.5*T_b+en2i_b(i)+en20_b;
end
for i=1:l2 % O2
    A(lall+4,l1+i) = 2.5*T_b+eo2i_b(i)+eo20_b;
end
for i=1:l3 % NO
    A(lall+4,l1+l2+i) = 2.5*T_b+enoi_b(i)+eno0_b+efno_b;
end
A(lall+4,lall+1) = 1.5*T_b+efn_b; % N
A(lall+4,lall+2) = 1.5*T_b+efo_b; % O
A(lall+4,lall+3) = 1/v_b*(3.5*nm_b*T_b+2.5*na_b*T_b+...
    sum((en2i_b+en20_b).*nn2i_b)+sum((eo2i_b+eo20_b).*no2i_b)+...
    sum((enoi_b+eno0_b).*nnoi_b)+efno_b*nno_b+efn_b*nn_b+efo_b*no_b);
A(lall+4,lall+4) = 2.5*nm_b+1.5*na_b;

AA = sparse(A);% figure(1); spy(A);

%% ������������ �������� ���������

% ��������� ������������� ������ � �������� ��������� ���������:
% k_rec / k_dis, �^3
Kdr_n2 = (m(1)*h^2/(m(4)*m(4)*2*pi*k*temp))^(3/2)*Z_rot(1)* ...
    exp(-en2_i'/(k*temp))*exp(D(1)/temp);
Kdr_o2 = (m(2)*h^2/(m(5)*m(5)*2*pi*k*temp))^(3/2)*Z_rot(2)* ...
    exp(-eo2_i'/(k*temp))*exp(D(2)/temp);
Kdr_no = (m(3)*h^2/(m(4)*m(5)*2*pi*k*temp))^(3/2)*Z_rot(3)* ...
    exp(-eno_i'/(k*temp))*exp(D(3)/temp);

% kb_exchange / kf_exchange
Kz_n2 = (m(1)*m(5)/(m(3)*m(4)))^1.5*Z_rot(1)/Z_rot(3)*...
    exp((repmat(eno_i',l(1),1)-repmat(en2_i,1,l(3)))/(k*temp))*...
    exp((D(1)-D(3))/temp);
Kz_o2 = (m(2)*m(4)/(m(3)*m(5)))^1.5*Z_rot(2)/Z_rot(3)*...
    exp((repmat(eno_i',l(2),1)-repmat(eo2_i,1,l(3)))/(k*temp))*...
    exp((D(2)-D(3))/temp);

% kb_VT(i-1->i) / kf_VT(i->i-1)
Kvt_n2 = exp((en2_i(1:end-1)-en2_i(2:end))/(k*temp))';
Kvt_o2 = exp((eo2_i(1:end-1)-eo2_i(2:end))/(k*temp))';
Kvt_no = exp((eno_i(1:end-1)-eno_i(2:end))/(k*temp))';

% ������������ �������� �����������
kd_n2 = kdis(temp,1) * Delta*n0/v0;
kd_o2 = kdis(temp,2) * Delta*n0/v0;
kd_no = kdis(temp,3) * Delta*n0/v0;

% ������������ �������� ������������
kr_n2 = zeros(5,l1); kr_o2 = zeros(5,l2); kr_no = zeros(5,l3);
for iM = 1:5
    kr_n2(iM,:) = kd_n2(iM,:) .* Kdr_n2 * n0;
    kr_o2(iM,:) = kd_o2(iM,:) .* Kdr_o2 * n0;
    kr_no(iM,:) = kd_no(iM,:) .* Kdr_no * n0;
end

% disp(temp)
% 
% dataset = [temp, kd_n2(1,:), kd_n2(2,:), kd_n2(3,:), kd_n2(4,:), kd_n2(5,:)];
% save dataset_STS_kd_N2.txt dataset -ascii -append
% dataset = [temp, kr_n2(1,:), kr_n2(2,:), kr_n2(3,:), kr_n2(4,:), kr_n2(5,:)];
% save dataset_STS_kr_N2.txt dataset -ascii -append
% dataset = [temp, kd_n2(1,:), kd_n2(2,:), kd_n2(3,:), kd_n2(4,:), kd_n2(5,:), kr_n2(1,:), kr_n2(2,:), kr_n2(3,:), kr_n2(4,:), kr_n2(5,:)];
% save dataset_STS_kd_kr_N2.txt dataset -ascii -append
% 
% dataset = [temp, kd_o2(1,:), kd_o2(2,:), kd_o2(3,:), kd_o2(4,:), kd_o2(5,:)];
% save dataset_STS_kd_O2.txt dataset -ascii -append
% dataset = [temp, kr_o2(1,:), kr_o2(2,:), kr_o2(3,:), kr_o2(4,:), kr_o2(5,:)];
% save dataset_STS_kr_O2.txt dataset -ascii -append
% dataset = [temp, kd_o2(1,:), kd_o2(2,:), kd_o2(3,:), kd_o2(4,:), kd_o2(5,:), kr_o2(1,:), kr_o2(2,:), kr_o2(3,:), kr_o2(4,:), kr_o2(5,:)];
% save dataset_STS_kd_kr_O2.txt dataset -ascii -append
% 
% dataset = [temp, kd_no(1,:), kd_no(2,:), kd_no(3,:), kd_no(4,:), kd_no(5,:)];
% save dataset_STS_kd_NO.txt dataset -ascii -append
% dataset = [temp, kr_no(1,:), kr_no(2,:), kr_no(3,:), kr_no(4,:), kr_no(5,:)];
% save dataset_STS_kr_NO.txt dataset -ascii -append
% dataset = [temp, kd_no(1,:), kd_no(2,:), kd_no(3,:), kd_no(4,:), kd_no(5,:), kr_no(1,:), kr_no(2,:), kr_no(3,:), kr_no(4,:), kr_no(5,:)];
% save dataset_STS_kd_kr_NO.txt dataset -ascii -append

% ������������ �������� ������� ����������
if  strcmp(sw_z,'Savelev')
    % ������������� ��������� STELLAR database
    [kf_n2, kf_o2] = k_ex_savelev_st(temp);
    kf_n2 = kf_n2 * Delta*n0/v0;
    kf_o2 = kf_o2 * Delta*n0/v0;
    kb_n2 = kf_n2 .* Kz_n2;
    kb_o2 = kf_o2 .* Kz_o2;
elseif  strcmp(sw_z,'Starik')
    [kf_n2, kb_o2] = k_ex_starik(temp);
    kf_n2 = kf_n2 * Delta*n0/v0;
    kb_o2 = kb_o2 * Delta*n0/v0;
    kb_n2 = kf_n2 .* Kz_n2;
    kf_o2 = kb_o2 ./ Kz_o2;
elseif strcmp(sw_z,'Stellar')
    % ������ ������������ STELLAR database
    [kf_n2, kf_o2] = k_ex_st_interp(temp);
    kf_n2 = kf_n2 * Delta*n0/v0;
    kf_o2 = kf_o2 * Delta*n0/v0;
    kb_n2 = kf_n2 .* Kz_n2;
    kb_o2 = kf_o2 .* Kz_o2;
else
    disp('Error! Check switch on Zeldovich reaction model.');
    return;
end


% ������������ �������� VT ��������� i+1 -> i
kvt_down_n2 = zeros(5,Lmax1);
kvt_down_o2 = zeros(5,Lmax2);
kvt_down_no = zeros(5,Lmax3);
for iM = 1:5
    for i1 = 1:Lmax1
        kvt_down_n2(iM,i1) = kvt_fho(1, iM, temp, i1, i1-1) * Delta*n0/v0;
    end
    for i2 = 1:Lmax2
        kvt_down_o2(iM,i2) = kvt_fho(2, iM, temp, i2, i2-1) * Delta*n0/v0;
    end
    for i3 = 1:Lmax3
        kvt_down_no(iM,i3) = kvt_fho(3, iM, temp, i3, i3-1) * Delta*n0/v0;
    end
end

% ������������ �������� VT ��������� i -> i+1
kvt_up_n2 = zeros(5,Lmax1); kvt_up_o2 = zeros(5,Lmax2); kvt_up_no = zeros(5,Lmax3);
for ip = 1:5
    kvt_up_n2(ip,:) = kvt_down_n2(ip,:) .* Kvt_n2;
    kvt_up_o2(ip,:) = kvt_down_o2(ip,:) .* Kvt_o2;
    kvt_up_no(ip,:) = kvt_down_no(ip,:) .* Kvt_no;
end

% ������������ �������� VV ������� (i,j)->(i-1,j+1)
kvv_down_n2 = zeros(Lmax1,Lmax1);
kvv_down_o2 = zeros(Lmax2,Lmax2);
kvv_down_no = zeros(Lmax3,Lmax3);
for i = 1:Lmax1
    for j = 1:Lmax1
        kvv_down_n2(i,j) = kvv_fho(1, 1, temp, i, j-1, i-1, j) * ...
             Delta*n0/v0;
    end
end
for i = 1:Lmax2
    for j = 1:Lmax2
        kvv_down_o2(i,j) = kvv_fho(2, 2, temp, i, j-1, i-1, j) * ...
             Delta*n0/v0;
    end
end
for i = 1:Lmax3
    for j = 1:Lmax3
        kvv_down_no(i,j) = kvv_fho(3, 3, temp, i, j-1, i-1, j) * ...
             Delta*n0/v0;
    end
end

% ������������ �������� VV ������� (i-1,j+1)->(i,j)
kvv_up_n2 = zeros(Lmax1,Lmax1);
kvv_up_o2 = zeros(Lmax2,Lmax2);
kvv_up_no = zeros(Lmax3,Lmax3);
deps_n2 = en2_i(1:end-1)-en2_i(2:end);
deps_o2 = eo2_i(1:end-1)-eo2_i(2:end);
deps_no = eno_i(1:end-1)-eno_i(2:end);
for ip = 1:Lmax1
    kvv_up_n2(ip,:) = kvv_down_n2(ip,:) .* ...
        exp((deps_n2(ip)-deps_n2') / (k*temp));
end
for ip = 1:Lmax2
    kvv_up_o2(ip,:) = kvv_down_o2(ip,:) .* ...
        exp((deps_o2(ip)-deps_o2') / (k*temp));
end
for ip = 1:Lmax3
    kvv_up_no(ip,:) = kvv_down_no(ip,:) .* ...
        exp((deps_no(ip)-deps_no') / (k*temp));
end

% ������������ �������� VV' ������� (i,j)->(i-1,j+1)
kvvs_d_n2_o2 = zeros(Lmax1,Lmax2);
kvvs_d_n2_no = zeros(Lmax1,Lmax3);
kvvs_d_o2_n2 = zeros(Lmax2,Lmax1);
kvvs_d_o2_no = zeros(Lmax2,Lmax3);
kvvs_d_no_n2 = zeros(Lmax3,Lmax1);
kvvs_d_no_o2 = zeros(Lmax3,Lmax2);
for i = 1:Lmax1
    for j = 1:Lmax2
        kvvs_d_n2_o2(i,j) = kvv_fho(1, 2, temp, i, j-1, i-1, j) * ...
             Delta*n0/v0;
    end
    for j = 1:Lmax3
        kvvs_d_n2_no(i,j) = kvv_fho(1, 3, temp, i, j-1, i-1, j) * ...
             Delta*n0/v0;
    end
end
for i = 1:Lmax2
    for j = 1:Lmax1
        kvvs_d_o2_n2(i,j) = kvv_fho(2, 1, temp, i, j-1, i-1, j) * ...
             Delta*n0/v0;
    end
    for j = 1:Lmax3
        kvvs_d_o2_no(i,j) = kvv_fho(2, 3, temp, i, j-1, i-1, j) * ...
             Delta*n0/v0;
    end
end
for i = 1:Lmax3
    for j = 1:Lmax1
        kvvs_d_no_n2(i,j) = kvv_fho(3, 1, temp, i, j-1, i-1, j) * ...
             Delta*n0/v0;
    end
    for j = 1:Lmax2
        kvvs_d_no_o2(i,j) = kvv_fho(3, 2, temp, i, j-1, i-1, j) * ...
             Delta*n0/v0;
    end
end

% ������������ �������� VV' ������� (i-1,j+1)->(i,j)
kvvs_u_n2_o2 = zeros(Lmax1,Lmax2);
kvvs_u_n2_no = zeros(Lmax1,Lmax3);
kvvs_u_o2_n2 = zeros(Lmax2,Lmax1);
kvvs_u_o2_no = zeros(Lmax2,Lmax3);
kvvs_u_no_n2 = zeros(Lmax3,Lmax1);
kvvs_u_no_o2 = zeros(Lmax3,Lmax2);
for ip = 1:Lmax1
    kvvs_u_n2_o2(ip,:) = kvvs_d_n2_o2(ip,:) .* ...
        exp((deps_n2(ip)-deps_o2') / (k*temp));
    kvvs_u_n2_no(ip,:) = kvvs_d_n2_no(ip,:) .* ...
        exp((deps_n2(ip)-deps_no') / (k*temp));
end
for ip = 1:Lmax2
    kvvs_u_o2_n2(ip,:) = kvvs_d_o2_n2(ip,:) .* ...
        exp((deps_o2(ip)-deps_n2') / (k*temp));
    kvvs_u_o2_no(ip,:) = kvvs_d_o2_no(ip,:) .* ...
        exp((deps_o2(ip)-deps_no') / (k*temp));
end
for ip = 1:Lmax3
    kvvs_u_no_n2(ip,:) = kvvs_d_no_n2(ip,:) .* ...
        exp((deps_no(ip)-deps_n2') / (k*temp));
    kvvs_u_no_o2(ip,:) = kvvs_d_no_o2(ip,:) .* ...
        exp((deps_no(ip)-deps_o2') / (k*temp));
end

%% ���������� ������-������� ������ ������ B
RDn2 = zeros(l1,1); RDo2 = zeros(l2,1); RDno = zeros(l3,1);
RZn2 = zeros(l1,1); RZo2 = zeros(l2,1); RZno = zeros(l3,1);
RVTn2 = zeros(l1,1); RVTo2 = zeros(l2,1); RVTno = zeros(l3,1);
RVVn2 = zeros(l1,1); RVVo2 = zeros(l2,1); RVVno = zeros(l3,1);
RVVsn2 = zeros(l1,1); RVVso2 = zeros(l2,1); RVVsno = zeros(l3,1);
for i1 = 1:l1
    RDn2(i1) = nn2_b*(nn_b*nn_b*kr_n2(1,i1)-nn2i_b(i1)*kd_n2(1,i1)) + ...
               no2_b*(nn_b*nn_b*kr_n2(2,i1)-nn2i_b(i1)*kd_n2(2,i1)) + ...
               nno_b*(nn_b*nn_b*kr_n2(3,i1)-nn2i_b(i1)*kd_n2(3,i1)) + ...
                nn_b*(nn_b*nn_b*kr_n2(4,i1)-nn2i_b(i1)*kd_n2(4,i1)) + ...
                no_b*(nn_b*nn_b*kr_n2(5,i1)-nn2i_b(i1)*kd_n2(5,i1));
    RZn2(i1) = sum(nnoi_b*nn_b.*kb_n2(i1,:)' - nn2i_b(i1)*no_b*kf_n2(i1,:)');
    if i1 == 1 % ����������� ������� 0<->1
        RVTn2(i1) = nn2_b*(nn2i_b(i1+1)*kvt_down_n2(1,i1) - nn2i_b(i1)*kvt_up_n2(1,i1))+...
                    no2_b*(nn2i_b(i1+1)*kvt_down_n2(2,i1) - nn2i_b(i1)*kvt_up_n2(2,i1))+...
                    nno_b*(nn2i_b(i1+1)*kvt_down_n2(3,i1) - nn2i_b(i1)*kvt_up_n2(3,i1))+...
                     nn_b*(nn2i_b(i1+1)*kvt_down_n2(4,i1) - nn2i_b(i1)*kvt_up_n2(4,i1))+...
                     no_b*(nn2i_b(i1+1)*kvt_down_n2(5,i1) - nn2i_b(i1)*kvt_up_n2(5,i1));
        RVVn2(i1) = nn2i_b(i1+1)*sum(nn2i_b(1:end-1) .* kvv_down_n2(i1,:)') - ...
                     nn2i_b(i1)*sum(nn2i_b(2:end) .* kvv_up_n2(i1,:)');
        RVVsn2(i1) = nn2i_b(i1+1)*(sum(no2i_b(1:end-1) .* kvvs_d_n2_o2(i1,:)')) - ...
                     nn2i_b(i1)*(sum(no2i_b(2:end) .* kvvs_u_n2_o2(i1,:)')) + ...
                     nn2i_b(i1+1)*(sum(nnoi_b(1:end-1) .* kvvs_d_n2_no(i1,:)')) - ...
                     nn2i_b(i1)*(sum(nnoi_b(2:end) .* kvvs_u_n2_no(i1,:)'));

    elseif i1 == l1 % ����������� ������� Lmax <-> Lmax-1
        RVTn2(i1) = nn2_b*(nn2i_b(i1-1)*kvt_up_n2(1,i1-1) - nn2i_b(i1)*kvt_down_n2(1,i1-1))+...
                    no2_b*(nn2i_b(i1-1)*kvt_up_n2(2,i1-1) - nn2i_b(i1)*kvt_down_n2(2,i1-1))+...
                    nno_b*(nn2i_b(i1-1)*kvt_up_n2(3,i1-1) - nn2i_b(i1)*kvt_down_n2(3,i1-1))+...
                     nn_b*(nn2i_b(i1-1)*kvt_up_n2(4,i1-1) - nn2i_b(i1)*kvt_down_n2(4,i1-1))+...
                     no_b*(nn2i_b(i1-1)*kvt_up_n2(5,i1-1) - nn2i_b(i1)*kvt_down_n2(5,i1-1));
        RVVn2(i1) = nn2i_b(i1-1)*sum(nn2i_b(2:end) .* kvv_up_n2(i1-1,:)') - ...
                     nn2i_b(i1)*sum(nn2i_b(1:end-1) .* kvv_down_n2(i1-1,:)');
        RVVsn2(i1) = nn2i_b(i1-1)*(sum(no2i_b(2:end) .* kvvs_u_n2_o2(i1-1,:)')) - ...
                     nn2i_b(i1)*(sum(no2i_b(1:end-1) .* kvvs_d_n2_o2(i1-1,:)')) + ...
                     nn2i_b(i1-1)*(sum(nnoi_b(2:end) .* kvvs_u_n2_no(i1-1,:)')) - ...
                     nn2i_b(i1)*(sum(nnoi_b(1:end-1) .* kvvs_d_n2_no(i1-1,:)'));

    else
        RVTn2(i1) = nn2_b*(nn2i_b(i1+1)*kvt_down_n2(1,i1)+nn2i_b(i1-1)*kvt_up_n2(1,i1-1)-...
                               nn2i_b(i1)*(kvt_up_n2(1,i1)+kvt_down_n2(1,i1-1)))+...
                    no2_b*(nn2i_b(i1+1)*kvt_down_n2(2,i1)+nn2i_b(i1-1)*kvt_up_n2(2,i1-1)-...
                               nn2i_b(i1)*(kvt_up_n2(2,i1)+kvt_down_n2(2,i1-1)))+...
                    nno_b*(nn2i_b(i1+1)*kvt_down_n2(3,i1)+nn2i_b(i1-1)*kvt_up_n2(3,i1-1)-...
                               nn2i_b(i1)*(kvt_up_n2(3,i1)+kvt_down_n2(3,i1-1)))+...
                     nn_b*(nn2i_b(i1+1)*kvt_down_n2(4,i1)+nn2i_b(i1-1)*kvt_up_n2(4,i1-1)-...
                               nn2i_b(i1)*(kvt_up_n2(4,i1)+kvt_down_n2(4,i1-1)))+...
                     no_b*(nn2i_b(i1+1)*kvt_down_n2(5,i1)+nn2i_b(i1-1)*kvt_up_n2(5,i1-1)-...
                               nn2i_b(i1)*(kvt_up_n2(5,i1)+kvt_down_n2(5,i1-1)));
        RVVn2(i1) = nn2i_b(i1+1)*sum(nn2i_b(1:end-1) .* kvv_down_n2(i1,:)') + ...
                   nn2i_b(i1-1)*sum(nn2i_b(2:end) .* kvv_up_n2(i1-1,:)') - ...
                     nn2i_b(i1)*(sum(nn2i_b(2:end) .* kvv_up_n2(i1,:)') + ...
                                sum(nn2i_b(1:end-1) .* kvv_down_n2(i1-1,:)'));
        RVVsn2(i1) = nn2i_b(i1+1)*(sum(no2i_b(1:end-1) .* kvvs_d_n2_o2(i1,:)')) + ...
                     nn2i_b(i1-1)*(sum(no2i_b(2:end) .* kvvs_u_n2_o2(i1-1,:)')) - ...
                       nn2i_b(i1)*(sum(no2i_b(2:end) .* kvvs_u_n2_o2(i1,:)') + ...
                                   sum(no2i_b(1:end-1) .* kvvs_d_n2_o2(i1-1,:)')) + ...
                     nn2i_b(i1+1)*(sum(nnoi_b(1:end-1) .* kvvs_d_n2_no(i1,:)')) + ...
                     nn2i_b(i1-1)*(sum(nnoi_b(2:end) .* kvvs_u_n2_no(i1-1,:)')) - ...
                       nn2i_b(i1)*(sum(nnoi_b(2:end) .* kvvs_u_n2_no(i1,:)') + ...
                                   sum(nnoi_b(1:end-1) .* kvvs_d_n2_no(i1-1,:)'));
    end
end
for i2 = 1:l2
    RDo2(i2) = nn2_b*(no_b*no_b*kr_o2(1,i2)-no2i_b(i2)*kd_o2(1,i2)) + ...
               no2_b*(no_b*no_b*kr_o2(2,i2)-no2i_b(i2)*kd_o2(2,i2)) + ...
               nno_b*(no_b*no_b*kr_o2(3,i2)-no2i_b(i2)*kd_o2(3,i2)) + ...
                nn_b*(no_b*no_b*kr_o2(4,i2)-no2i_b(i2)*kd_o2(4,i2)) + ...
                no_b*(no_b*no_b*kr_o2(5,i2)-no2i_b(i2)*kd_o2(5,i2));
    RZo2(i2) = sum(nnoi_b*no_b.*kb_o2(i2,:)'-no2i_b(i2)*nn_b*kf_o2(i2,:)');
    if i2 == 1 % ����������� ������� 0<->1
        RVTo2(i2) = nn2_b*(no2i_b(i2+1)*kvt_down_o2(1,i2) - no2i_b(i2)*kvt_up_o2(1,i2))+...
                    no2_b*(no2i_b(i2+1)*kvt_down_o2(2,i2) - no2i_b(i2)*kvt_up_o2(2,i2))+...
                    nno_b*(no2i_b(i2+1)*kvt_down_o2(3,i2) - no2i_b(i2)*kvt_up_o2(3,i2))+...
                     nn_b*(no2i_b(i2+1)*kvt_down_o2(4,i2) - no2i_b(i2)*kvt_up_o2(4,i2))+...
                     no_b*(no2i_b(i2+1)*kvt_down_o2(5,i2) - no2i_b(i2)*kvt_up_o2(5,i2));
        RVVo2(i2) = no2i_b(i2+1)*sum(no2i_b(1:end-1) .* kvv_down_o2(i2,:)') - ...
                     no2i_b(i2)*sum(no2i_b(2:end) .* kvv_up_o2(i2,:)');
        RVVso2(i2) = no2i_b(i2+1)*(sum(nn2i_b(1:end-1) .* kvvs_d_o2_n2(i2,:)')) - ...
                     no2i_b(i2)*(sum(nn2i_b(2:end) .* kvvs_u_o2_n2(i2,:)')) + ...
                     no2i_b(i2+1)*(sum(nnoi_b(1:end-1) .* kvvs_d_o2_no(i2,:)')) - ...
                     no2i_b(i2)*(sum(nnoi_b(2:end) .* kvvs_u_o2_no(i2,:)'));

    elseif i2 == l2 % ����������� ������� Lmax <-> Lmax-1
        RVTo2(i2) = nn2_b*(no2i_b(i2-1)*kvt_up_o2(1,i2-1) - no2i_b(i2)*kvt_down_o2(1,i2-1))+...
                    no2_b*(no2i_b(i2-1)*kvt_up_o2(2,i2-1) - no2i_b(i2)*kvt_down_o2(2,i2-1))+...
                    nno_b*(no2i_b(i2-1)*kvt_up_o2(3,i2-1) - no2i_b(i2)*kvt_down_o2(3,i2-1))+...
                     nn_b*(no2i_b(i2-1)*kvt_up_o2(4,i2-1) - no2i_b(i2)*kvt_down_o2(4,i2-1))+...
                     no_b*(no2i_b(i2-1)*kvt_up_o2(5,i2-1) - no2i_b(i2)*kvt_down_o2(5,i2-1));
        RVVo2(i2) = no2i_b(i2-1)*sum(no2i_b(2:end) .* kvv_up_o2(i2-1,:)') - ...
                     no2i_b(i2)*sum(no2i_b(1:end-1) .* kvv_down_o2(i2-1,:)');
        RVVso2(i2) = no2i_b(i2-1)*(sum(nn2i_b(2:end) .* kvvs_u_o2_n2(i2-1,:)')) - ...
                     no2i_b(i2)*(sum(nn2i_b(1:end-1) .* kvvs_d_o2_n2(i2-1,:)')) + ...
                     no2i_b(i2-1)*(sum(nnoi_b(2:end) .* kvvs_u_o2_no(i2-1,:)')) - ...
                     no2i_b(i2)*(sum(nnoi_b(1:end-1) .* kvvs_d_o2_no(i2-1,:)'));

    else
        RVTo2(i2) = nn2_b*(no2i_b(i2+1)*kvt_down_o2(1,i2)+no2i_b(i2-1)*kvt_up_o2(1,i2-1)-...
                               no2i_b(i2)*(kvt_up_o2(1,i2)+kvt_down_o2(1,i2-1)))+...
                    no2_b*(no2i_b(i2+1)*kvt_down_o2(2,i2)+no2i_b(i2-1)*kvt_up_o2(2,i2-1)-...
                               no2i_b(i2)*(kvt_up_o2(2,i2)+kvt_down_o2(2,i2-1)))+...
                    nno_b*(no2i_b(i2+1)*kvt_down_o2(3,i2)+no2i_b(i2-1)*kvt_up_o2(3,i2-1)-...
                               no2i_b(i2)*(kvt_up_o2(3,i2)+kvt_down_o2(3,i2-1)))+...
                     nn_b*(no2i_b(i2+1)*kvt_down_o2(4,i2)+no2i_b(i2-1)*kvt_up_o2(4,i2-1)-...
                               no2i_b(i2)*(kvt_up_o2(4,i2)+kvt_down_o2(4,i2-1)))+...
                     no_b*(no2i_b(i2+1)*kvt_down_o2(5,i2)+no2i_b(i2-1)*kvt_up_o2(5,i2-1)-...
                               no2i_b(i2)*(kvt_up_o2(5,i2)+kvt_down_o2(5,i2-1)));
        RVVo2(i2) = no2i_b(i2+1)*sum(no2i_b(1:end-1) .* kvv_down_o2(i2,:)') + ...
                   no2i_b(i2-1)*sum(no2i_b(2:end) .* kvv_up_o2(i2-1,:)') - ...
                     no2i_b(i2)*(sum(no2i_b(2:end) .* kvv_up_o2(i2,:)') + ...
                                sum(no2i_b(1:end-1) .* kvv_down_o2(i2-1,:)'));
        RVVso2(i2) = no2i_b(i2+1)*(sum(nn2i_b(1:end-1) .* kvvs_d_o2_n2(i2,:)')) + ...
                     no2i_b(i2-1)*(sum(nn2i_b(2:end) .* kvvs_u_o2_n2(i2-1,:)')) - ...
                       no2i_b(i2)*(sum(nn2i_b(2:end) .* kvvs_u_o2_n2(i2,:)') + ...
                                   sum(nn2i_b(1:end-1) .* kvvs_d_o2_n2(i2-1,:)')) + ...
                     no2i_b(i2+1)*(sum(nnoi_b(1:end-1) .* kvvs_d_o2_no(i2,:)')) + ...
                     no2i_b(i2-1)*(sum(nnoi_b(2:end) .* kvvs_u_o2_no(i2-1,:)')) - ...
                       no2i_b(i2)*(sum(nnoi_b(2:end) .* kvvs_u_o2_no(i2,:)') + ...
                                   sum(nnoi_b(1:end-1) .* kvvs_d_o2_no(i2-1,:)'));
    end
end
for i3 = 1:l3
    RDno(i3) = nn2_b*(nn_b*no_b*kr_no(1,i3)-nnoi_b(i3)*kd_no(1,i3)) + ...
               no2_b*(nn_b*no_b*kr_no(2,i3)-nnoi_b(i3)*kd_no(2,i3)) + ...
               nno_b*(nn_b*no_b*kr_no(3,i3)-nnoi_b(i3)*kd_no(3,i3)) + ...
                nn_b*(nn_b*no_b*kr_no(4,i3)-nnoi_b(i3)*kd_no(4,i3)) + ...
                no_b*(nn_b*no_b*kr_no(5,i3)-nnoi_b(i3)*kd_no(5,i3));
    RZno(i3) = sum(nn2i_b*no_b.*kf_n2(:,i3)-nnoi_b(i3)*nn_b*kb_n2(:,i3))+...
        sum(no2i_b*nn_b.*kf_o2(:,i3)-nnoi_b(i3)*no_b*kb_o2(:,i3));
    if i3 == 1 % ����������� ������� 0<->1
        RVTno(i3) = nn2_b*(nnoi_b(i3+1)*kvt_down_no(1,i3) - nnoi_b(i3)*kvt_up_no(1,i3))+...
                    no2_b*(nnoi_b(i3+1)*kvt_down_no(2,i3) - nnoi_b(i3)*kvt_up_no(2,i3))+...
                    nno_b*(nnoi_b(i3+1)*kvt_down_no(3,i3) - nnoi_b(i3)*kvt_up_no(3,i3))+...
                     nn_b*(nnoi_b(i3+1)*kvt_down_no(4,i3) - nnoi_b(i3)*kvt_up_no(4,i3))+...
                     no_b*(nnoi_b(i3+1)*kvt_down_no(5,i3) - nnoi_b(i3)*kvt_up_no(5,i3));
        RVVno(i3) = nnoi_b(i3+1)*sum(nnoi_b(1:end-1) .* kvv_down_no(i3,:)') - ...
                     nnoi_b(i3)*sum(nnoi_b(2:end) .* kvv_up_no(i3,:)');
        RVVsno(i3) = nnoi_b(i3+1)*(sum(nn2i_b(1:end-1) .* kvvs_d_no_n2(i3,:)')) - ...
                     nnoi_b(i3)*(sum(nn2i_b(2:end) .* kvvs_u_no_n2(i3,:)')) + ...
                     nnoi_b(i3+1)*(sum(no2i_b(1:end-1) .* kvvs_d_no_o2(i3,:)')) - ...
                     nnoi_b(i3)*(sum(no2i_b(2:end) .* kvvs_u_no_o2(i3,:)'));

    elseif i3 == l3 % ����������� ������� Lmax <-> Lmax-1
        RVTno(i3) = nn2_b*(nnoi_b(i3-1)*kvt_up_no(1,i3-1) - nnoi_b(i3)*kvt_down_no(1,i3-1))+...
                    no2_b*(nnoi_b(i3-1)*kvt_up_no(2,i3-1) - nnoi_b(i3)*kvt_down_no(2,i3-1))+...
                    nno_b*(nnoi_b(i3-1)*kvt_up_no(3,i3-1) - nnoi_b(i3)*kvt_down_no(3,i3-1))+...
                     nn_b*(nnoi_b(i3-1)*kvt_up_no(4,i3-1) - nnoi_b(i3)*kvt_down_no(4,i3-1))+...
                     no_b*(nnoi_b(i3-1)*kvt_up_no(5,i3-1) - nnoi_b(i3)*kvt_down_no(5,i3-1));
        RVVno(i3) = nnoi_b(i3-1)*sum(nnoi_b(2:end) .* kvv_up_no(i3-1,:)') - ...
                     nnoi_b(i3)*sum(nnoi_b(1:end-1) .* kvv_down_no(i3-1,:)');
        RVVsno(i3) = nnoi_b(i3-1)*(sum(nn2i_b(2:end) .* kvvs_u_no_n2(i3-1,:)')) - ...
                     nnoi_b(i3)*(sum(nn2i_b(1:end-1) .* kvvs_d_no_n2(i3-1,:)')) + ...
                     nnoi_b(i3-1)*(sum(no2i_b(2:end) .* kvvs_u_no_o2(i3-1,:)')) - ...
                     nnoi_b(i3)*(sum(no2i_b(1:end-1) .* kvvs_d_no_o2(i3-1,:)'));

    else
        RVTno(i3) = nn2_b*(nnoi_b(i3+1)*kvt_down_no(1,i3)+nnoi_b(i3-1)*kvt_up_no(1,i3-1)-...
                               nnoi_b(i3)*(kvt_up_no(1,i3)+kvt_down_no(1,i3-1)))+...
                    no2_b*(nnoi_b(i3+1)*kvt_down_no(2,i3)+nnoi_b(i3-1)*kvt_up_no(2,i3-1)-...
                               nnoi_b(i3)*(kvt_up_no(2,i3)+kvt_down_no(2,i3-1)))+...
                    nno_b*(nnoi_b(i3+1)*kvt_down_no(3,i3)+nnoi_b(i3-1)*kvt_up_no(3,i3-1)-...
                               nnoi_b(i3)*(kvt_up_no(3,i3)+kvt_down_no(3,i3-1)))+...
                     nn_b*(nnoi_b(i3+1)*kvt_down_no(4,i3)+nnoi_b(i3-1)*kvt_up_no(4,i3-1)-...
                               nnoi_b(i3)*(kvt_up_no(4,i3)+kvt_down_no(4,i3-1)))+...
                     no_b*(nnoi_b(i3+1)*kvt_down_no(5,i3)+nnoi_b(i3-1)*kvt_up_no(5,i3-1)-...
                               nnoi_b(i3)*(kvt_up_no(5,i3)+kvt_down_no(5,i3-1)));
         RVVno(i3) = nnoi_b(i3+1)*sum(nnoi_b(1:end-1) .* kvv_down_no(i3,:)') + ...
                   nnoi_b(i3-1)*sum(nnoi_b(2:end) .* kvv_up_no(i3-1,:)') - ...
                     nnoi_b(i3)*(sum(nnoi_b(2:end) .* kvv_up_no(i3,:)') + ...
                                sum(nnoi_b(1:end-1) .* kvv_down_no(i3-1,:)'));
        RVVsno(i3) = nnoi_b(i3+1)*(sum(nn2i_b(1:end-1) .* kvvs_d_no_n2(i3,:)')) + ...
                     nnoi_b(i3-1)*(sum(nn2i_b(2:end) .* kvvs_u_no_n2(i3-1,:)')) - ...
                       nnoi_b(i3)*(sum(nn2i_b(2:end) .* kvvs_u_no_n2(i3,:)') + ...
                                   sum(nn2i_b(1:end-1) .* kvvs_d_no_n2(i3-1,:)')) + ...
                     nnoi_b(i3+1)*(sum(no2i_b(1:end-1) .* kvvs_d_no_o2(i3,:)')) + ...
                     nnoi_b(i3-1)*(sum(no2i_b(2:end) .* kvvs_u_no_o2(i3-1,:)')) - ...
                       nnoi_b(i3)*(sum(no2i_b(2:end) .* kvvs_u_no_o2(i3,:)') + ...
                                   sum(no2i_b(1:end-1) .* kvvs_d_no_o2(i3-1,:)'));
    end
end

B = zeros(lall+4,1);
B(1:l1) = RDn2 + RZn2 + RVTn2 + RVVn2 + RVVsn2;
B(l1+1:l1+l2) = RDo2 + RZo2 + RVTo2 + RVVo2 + RVVso2;
B(l1+l2+1:lall) = RDno + RZno + RVTno + RVVno + RVVsno;
B(lall+1) = - sum(RDno) - 2*sum(RDn2) - sum(RZn2) + sum(RZo2);
B(lall+2) = - sum(RDno) - 2*sum(RDo2) + sum(RZn2)-sum(RZo2);

dy = AA^(-1)*B;
