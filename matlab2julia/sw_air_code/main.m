% ��������� ������� ������������� � ���������� ����������
% �� ������� ������ � �������
% ����������� ������

clear

tic

format long e
global c h h_bar N_a k sw_u sw_z
global m osc_mass ram_masses sw_o l om_e om_x_e Be D r0 em re
global CA nA n0 v0 T0 Delta FHO_data vss_data CAZ nAZ EaZ
global en2_i en2_0 eo2_i eo2_0 eno_i eno_0 xc ef

%% swich
% ������ �������� ������������� �������
sw_o = 1; % ��������������� ��.
% sw_o = 2; % ������������� ��.

% ��������� � ������ ���������
sw_arr = 'Park';
% sw_arr = 'Scanlon';

% ���� �� �������� ������ ��
sw_u = 'D/6k';
% sw_u = '3T';
% sw_u = 'Inf';

% ���� �� ������ ������� ����������
sw_z = 'Savelev';
% sw_z = 'Starik';
% sw_z = 'Stellar';

%% ���������
c = 2.99e8;         % �������� ����� (�/���)
h = 6.6261e-34;     % ���������� ������ (��*���)
k = 1.3807e-23;     % ���������� ��������� (��/�)
N_a = 6.0221e23;    % ���������� �������� (1/����)
h_bar = h/(2*pi);
R = 8.3145;         % ������������� ������� ���������� (��/����/�)

% ������������������ ���������� ������� N2-O2-NO, �^-1
om_e = [235857 158019 190420];
om_x_e = [1432 1198 1407.5];
Be = [1.998 1.4377 1.6720]*100;

% ��������� ����������������� ������ �� ��������
% load('wurster_data_times.mat');
% data_40_60 = sortrows(data_40_60);
% data_22_77 = sortrows(data_22_77);
% data_5_95 = sortrows(data_5_95);
% data_exp = [data_5_95; data_22_77; data_40_60];

if  strcmp(sw_arr,'Park')
    ARR_D_data = load('arr_d_park.dat');
    ARR_Z_data = load('arr_z_park.dat');

    % ������� �����������, K
    D = ARR_D_data([3,6,9],1)';

    % ��������� � ������ ���������
    % �����������
    CA = ARR_D_data([1,4,7],:);     % m^3/sec
    nA = ARR_D_data([2,5,8],:);

    % ������� ����������
    CAZ = ARR_Z_data(:,1);          % m^3/sec
    nAZ = ARR_Z_data(:,2);
    EaZ = ARR_Z_data(:,3);          % ��

elseif strcmp(sw_arr,'Scanlon')
    ARR_D_data = load('arr_d_scanlon.dat');
    ARR_Z_data = load('arr_z_scanlon.dat');

    % ������� �����������, K
    D = ARR_D_data([3,6,9],1)'/k; % J->K

    % ��������� � ������ ���������
    % �����������
    CA = ARR_D_data([1,4,7],:);     % m^3/sec
    nA = ARR_D_data([2,5,8],:);

    % ������� ����������
    CAZ = ARR_Z_data(:,1);          % m^3/sec
    nAZ = ARR_Z_data(:,2);
    EaZ = ARR_Z_data(:,3);          % ��

else
    disp('Error! Check Arrhenius input.')
    return;
end

% ����� ������������� �������
QN = [47 36 39; 33 26 28];
l = QN(sw_o,:);

% ������������� ������� � ������� �� �������� ������, ��
en2_i = en_vibr(1);
en2_0 = en_vibr_0(1);
eo2_i = en_vibr(2);
eo2_0 = en_vibr_0(2);
eno_i = en_vibr(3);
eno_0 = en_vibr_0(3);

% ������� �����������, ��
ef = [0, 0, (0.5*(D(1)+D(2))-D(3)), 0.5*D(1), 0.5*D(2)]*k;

% ����� ������� � ������, ��
mu = [28 32 30 14 16]*1e-3; % kg/mol
m = mu/N_a; % N2-O2-NO-N-O

% ����� ����������� N2-O2-NO, ��
osc_mass = [0.5*m(4), ...
            0.5*m(5), ...
            m(4)*m(5)/(m(4)+m(5))];

% ��������� mass(atom1)/mass(molecule), mass(atom2)/mass(molecule)
ram_masses = [0.5, 0.5; ...      'N2'
              0.5, 0.5; ...      'O2'
              0.4668, 0.5332]; % 'NO'

% ��������� ������: beta [A^-1], E [��], Svt
fho_data_n2 = [3.9e10,  1 * k,  0.9; ...           'N2+N2'
               3.9e10,  6 * k,  0.95; ...          'N2+O2'
               4e10,  2 * k,  0.75; ...            'N2+NO'
               4.6e10,  1 * k,  0.99; ...          'N2+N'
               7.3e10,  500 * k,  0.175]; %        'N2+O'
fho_data_o2 = [4.1e10,  150 * k,  0.333333; ...    'O2+N2'
            4.3e10,  40 * k,  0.99; ...            'O2+O2'
            4.1e10,  150 * k,  0.333333; ...       'O2+NO'
            7.3e10,  10 * k,  0.25; ...            'O2+N'
            2.6e10,  17000 * k,  0.2]; %           'O2+O'
fho_data_no = [4.4e10,  20 * k,  0.9; ...          'NO+N2'
               6.75e10,  1500 * k,  0.2; ...       'NO+O2'
               6.25e10,  4500 * k,  0.03; ...      'NO+NO'
               5e10,  200 * k,  0.3183; ...        'NO+N'
               7.9e10,  16000 * k,  0.06]; %       'NO+O'
FHO_data = cat(3,fho_data_n2,fho_data_o2,fho_data_no);

% ��������� ������ VSS: dref [�], omega
vss_data_n2 = [4.04e-10, 0.686; ...                    'N2+N2'
               3.604e-10, 0.703; ...                   'N2+O2'
               4.391e-10, 0.756; ...                   'N2+NO'
               4.088e-10, 0.762; ...                   'N2+N'
               3.2220000000000004e-10, 0.702]; %       'N2+O'

vss_data_o2 = [3.604e-10, 0.703; ...                   'O2+N2'
               3.8960000000000003e-10, 0.7; ...        'O2+O2'
               4.054e-10, 0.718; ...                   'O2+NO'
               3.7210000000000004e-10, 0.757; ...      'O2+N'
               3.734e-10, 0.76]; %                     'O2+O'

vss_data_no = [4.391e-10, 0.756; ...                   'NO+N2'
               4.054e-10, 0.718; ...                   'NO+O2'
               4.2180000000000003e-10, 0.737; ...      'NO+NO'
               4.028e-10, 0.788; ...                   'NO+N'
               3.693e-10, 0.752]; %                    'NO+O'

vss_data = cat(3,vss_data_n2,vss_data_o2,vss_data_no);

% ����-���. ������ ������������ N2-N2, O2-O2, NO-NO, N-N, O-O
R0 = [3.621e-10 3.458e-10 3.47e-10 3.298e-10 2.75e-10]; % m

% ������� �����. ��� ������������ N2-N2, O2-O2, NO-NO, N-N, O-O, �������-�����
EM = [97.5 107.4 119 71.4 80]; % K

% ����-���. ������
r0 = 0.5*[R0(1)+R0; R0(2)+R0; R0(3)+R0];

% ������� �����. ���, �������-�����
em = [sqrt(EM(1)*R0(1)^6 * EM.*R0.^6)./r0(1,:).^6;...
      sqrt(EM(2)*R0(2)^6 * EM.*R0.^6)./r0(2,:).^6;...
      sqrt(EM(3)*R0(3)^6 * EM.*R0.^6)./r0(3,:).^6];

% ���������� ����������, �
re = [1.097 1.207 1.151]*1e-10; % N2-O2-NO

%% ��������� ������� � ���������� ������ �� Wurster, 1991
% �������� ��, ��/���
us_95_5_exp = [3.87; 3.49; 3.15; 2.97];
us_78_22_exp = [3.85; 3.52; 3.26; 2.99];
us_60_40_exp = [3.85; 3.47; 3.24; 3.06];

% ������ �������: xN2-xO2-us
incon = [0.95, 0.05, us_95_5_exp(1)];
% incon = [0.777, 0.223, us_78_22_exp(1)];
% incon = [0.6, 0.4, us_60_40_exp(1)];

% ��������, ����������� � ���. ����������� � ���������� ������
Torr = 133.322;
p0 = 2.25*Torr; % Pa
T0 = 300;       % K
Tv0n2 = T0;     % K
Tv0o2 = T0;     % K
Tv0no = T0;     % K

% ��������� ������������� ���������
Zv0_n2 = sum(exp(-en2_i/Tv0n2/k));
Zv0_o2 = sum(exp(-eo2_i/Tv0o2/k));
Zv0_no = sum(exp(-eno_i/Tv0no/k));

% ��������� �������� ��������� �����, �^-3
n0 = p0/(k*T0);

% ������� ���������� ������� ����� �������������� ��� ������������
% N2-N2 (���������������� ���)
sigma0 = pi*R0(1)^2;                % �^2
lambda0 = 1/(sqrt(2)*n0*sigma0);    % �
Delta = lambda0;                    % �

%% ������� �������
% ������ ����� � ����� xc = nc/n
xc = zeros(1,5);
xc(1) = incon(1);
xc(2) = incon(2);

% �������� ������� �����, �/���
v0 = incon(3)*1e3;

% �������� ����� ���� ����� ��
gamma0 = 1.4;                   % ���������� ��������
rho0_c = m.*xc*n0;              % �������� ��������� ���������, ��/�^-3
rho0 = sum(rho0_c);             % ��������� ��������� �����, ��/�^-3
mu0_mix = sum(rho0_c./mu)/rho0; % �������� ����� �����, (��/����)^-1
R_bar = R*mu0_mix;              % �������� ������� ����������, ��/��/�
a0 = sqrt(gamma0*R_bar*T0);     % ������� �������� �����, �/���
M0 = v0/a0;                     % ��������� ����� ����

% ������������ ��������� �� ������� ��
NN = in_con;
n1 = NN(1)
v1 = NN(2)
T1 = NN(3)

% ��������� ����� �� �� � �������. ����
Y0_bar = zeros(sum(l)+4,1);

% ������������ ���. ������� �������
Y0_bar(1:l(1)) = xc(1)*n1/Zv0_n2*exp(-en2_i/Tv0n2/k);             % N2
Y0_bar(l(1)+1:l(1)+l(2)) = xc(2)*n1/Zv0_o2*exp(-eo2_i/Tv0o2/k);   % O2
Y0_bar(l(1)+l(2)+1:sum(l)) = xc(3)*n1/Zv0_no*exp(-eno_i/Tv0no/k); % NO
Y0_bar(sum(l)+1) = xc(4);   % N
Y0_bar(sum(l)+2) = xc(5);   % O
Y0_bar(sum(l)+3) = v1;      % �/� ��������
Y0_bar(sum(l)+4) = T1;      % �/� �����������

% ������� ��������������
x_w = 2;      % m

% �������� �������������� � �������. ����
% xspan = (0:x_w*1e-1:x_w)./Delta;
xspan = [0, x_w]./Delta;

% �������
options = odeset('RelTol', 1e-5, 'AbsTol', 1e-5);
[X,Y] = ode15s(@rpart_fho, xspan, Y0_bar,options);

dataset = [X, Y];
save solution_XY.dat dataset -ascii

toc
%% �������� ���������
% ���������� �� ��, ��
x_s = X*Delta*100;

% ����������� �����, �
Temp = Y(:,sum(l)+4)*T0;

% �������� �����, �/���
v = Y(:,sum(l)+3)*v0;

% ������������, �^-3
nn2_i = Y(:,1:l(1))*n0;
no2_i = Y(:,l(1)+1:l(1)+l(2))*n0;
nno_i = Y(:,l(1)+l(2)+1:sum(l))*n0;

% �������� ��������� �������, �^-3
nn2 = sum(nn2_i,2);
no2 = sum(no2_i,2);
nno = sum(nno_i,2);

% �������� ��������� ������, �^-3
nn = Y(:,sum(l)+1)*n0; % N
no = Y(:,sum(l)+2)*n0; % O

% ����� �������� ��������� ������� � ������, �^-3
n_a = nn+no;
n_m = nn2+no2+nno;
%
time_s = X*Delta/v0;    % sec
time_mcs = time_s*1e6;  % mcsec

% �������� � ������������� ��������
Npoint = length(X);
Nall = n_m+n_a;
nn2i_n = nn2_i./repmat(Nall,1,l(1));
no2i_n = no2_i./repmat(Nall,1,l(2));
nnoi_n = nno_i./repmat(Nall,1,l(3));
nn2_n = sum(nn2i_n,2);
no2_n = sum(no2i_n,2);
nno_n = sum(nnoi_n,2);
nn_n = nn./Nall;
no_n = no./Nall;

%% �������� �������������� ������
% ������ P,RHO,H � �������������� ����
rho = m(1)*sum(nn2_i,2)+m(2)*sum(no2_i,2)+m(3)*sum(nno_i,2)+...
    m(4)*nn+m(5)*no;
p = (n_m+n_a)*k.*Temp;
Tvn2 = en2_i(2)./(k*log(nn2i_n(:,1)./nn2i_n(:,2)));
Tvo2 = eo2_i(2)./(k*log(no2i_n(:,1)./no2i_n(:,2)));
Tvno = eno_i(2)./(k*log(nnoi_n(:,1)./nnoi_n(:,2)));
ev_n2 = repmat(en2_i'+en2_0,Npoint,1).*nn2_i;
ev_o2 = repmat(eo2_i'+eo2_0,Npoint,1).*no2_i;
ev_no = repmat(eno_i'+eno_0,Npoint,1).*nno_i;
e_v = sum(ev_n2,2)+sum(ev_o2,2)+sum(ev_no,2);
e_f = 0.5*k*(D(1)*nn+D(2)*no)+k*(0.5*(D(1)+D(2))-D(3))*sum(nno_i,2);
e_tr = 1.5*(n_m+n_a)*k.*Temp;
e_rot = n_m*k.*Temp;
H = (3.5*n_m*k.*Temp+2.5*n_a*k.*Temp+e_v+e_f)./rho;
%
e_v0 = n0*(xc(1)/Zv0_n2*sum(exp(-en2_i/Tv0n2/k).*(en2_i+en2_0))+...
    xc(2)/Zv0_o2*sum(exp(-eo2_i/Tv0o2/k).*(eo2_i+eo2_0))+...
    xc(3)/Zv0_no*sum(exp(-eno_i/Tv0no/k).*(eno_i+eno_0)));
e_f0 = 0.5*k*(D(1)*xc(4)*n0+D(2)*xc(5)*n0)+k*(0.5*(D(1)+D(2))-D(3))*xc(3)*n0;
u10 = rho0*v0;
u20 = rho0*v0^2+p0;
u30 = (3.5*(sum(xc(1:3)))*n0*k*T0+2.5*(sum(xc(4:5)))*n0*k*T0+e_v0+e_f0)/rho0+v0^2/2;

% ������� � ���
u1 = u10-rho.*v;
u2 = u20-rho.*v.^2-p;
u3 = u30-H-v.^2/2;

% ������������� �������
d1 = max(abs(u1)/u10);
d2 = max(abs(u2)/u20);
d3 = max(abs(u3)/u30);

tol = 1e-4;
if (d1>tol)||(d2>tol)||(d3>tol)
    disp('Big error!');
    return;
end

beep
msgbox('Woah! Finally done!','Success');
