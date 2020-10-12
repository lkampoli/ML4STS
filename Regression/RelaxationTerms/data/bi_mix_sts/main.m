% ��������� ������� ������������� � ���������� ����������
% �� ������� ������ � �������� ����� A2/A

clear all
format long e
global c h h_bar N_a k m sw_o sw_sp e_i e_0 l om_e om_x_e Be D xc r0 em re ...
    CA nA n0 v0 T0 Delta
tic

%% ������� ���������
% ���� ������� � ������: 1 - N2/N; 2 - O2/O
sw_sp = 1;

% ������ �������� ������������� �������: 1 - �����. ����������, 2 - ����.
% ����������
sw_o = 1;

% ��������� ������ ����� [xA2; xA] (xc = nc/n)
xc = [1; 0];

% ���� ��������������, �
x_w = 1000;

%% ���������
c = 2.99e8;         % �������� ����� (�/���)
h = 6.6261e-34;     % ���������� ������ (��*���)
k = 1.3807e-23;     % ���������� ��������� (��/�)
N_a = 6.0221e23;	% ���������� �������� (1/����)
h_bar = h/(2*pi);
R = 8.3145;         % ������������� ������� ���������� (��/����/�)

% �������� �������������
load('data_species.mat');

% ������������������ ����������
om_e = OMEGA(sw_sp,1); % m^-1
om_x_e = OMEGA(sw_sp,2); % m^-1
Be = BE(sw_sp); % m^-1

% ������� �����������, �
D = ED(sw_sp);

% ��������� � ������ ���������
CA = CArr(sw_sp,:); % �^3/�
nA = NArr(sw_sp,:);

% ����� ������������� �������
l = QN(sw_sp,sw_o);

% ������������� ������� � ������� �� �������� ������ � ������� ��������
% ������, ��
e_i = en_vibr;
e_0 = en_vibr_0;

% ����� ������� � ������, ��
mu = [MU(sw_sp); 0.5*MU(sw_sp)]*1e-3;
m = mu / N_a;

% ������� ������������ ��������-��������
sigma0 = pi*R0(sw_sp,1)^2;

% ����-���. ������, ����. ���-���, ���-��
r0 = [R0(sw_sp,1); 0.5*(R0(sw_sp,1)+R0(sw_sp,2))];

% ������� �����. ���, �������-�����
em = [EM(sw_sp,1); sqrt(EM(sw_sp,1)*EM(sw_sp,2)*...
    R0(sw_sp,1)^6*R0(sw_sp,2)^6)/r0(2)^6];

% ��������������� ����������, �
re = RE(sw_sp);

%% ������� � ���������� ������
% ��������� ��������, ��
p0 = 0.8*133.322; % Pa

% ��������� ����������� ���� � ������������� �����������, �
T0 = 300;
Tv0 = T0;

% ����� ����
M0 = 13.4;

% ��������� �������� ��������� �����, �^-3
n0 = p0/(k*T0);

%% ������ �������� � ���������� ������, �/�
if xc(1) ~= 0
    gamma0 = 1.4; % ���������� ��������
else
    gamma0 = 5/3;
end

% ��������� �������� ��������� ����������� �����, ��/�^3
rho0_c = m.*xc*n0;

% ��������� ��������� �����, ��/�^3
rho0 = sum(rho0_c);

% (�������� ����� �����)^-1, ����/��
mu_mix = sum(rho0_c./mu)/rho0;

% ������� ���������� �����, ��/��/�
R_bar = R*mu_mix;

% ������� �������� �����, �/�
a0 = sqrt(gamma0*R_bar*T0);

% �������� ����������� ������, �/�
v0 = M0*a0;

%% ������������ ��������� �� ������� ��
NN = in_con;
n1 = NN(1);
T1 = NN(2);
v1 = NN(3);

% ����������� ������������� �������������� �����
Zvibr_0 = sum(exp(-e_i/Tv0/k));

% ������ ������� ������
Y0_bar = zeros(l+3,1);
Y0_bar(1:l) = xc(1)*n1/Zvibr_0*exp(-e_i/Tv0/k); % ������������ ���. �������
Y0_bar(l+1) = xc(2)*n1; % ���. ���-�� ������
Y0_bar(l+2) = v1; % ��������
Y0_bar(l+3) = T1; % �����������

%% ������� �������
Delta = 1/(sqrt(2)*n0*sigma0); % ������� ���������� ������� ����� �������������� �2-�2
xspan = [0, x_w]./Delta; % �������� �������������� � �������. ����

options = odeset('RelTol', 1e-12, 'AbsTol', 1e-12);
[X,Y] = ode15s(@rpart, xspan, Y0_bar,options);

%% �������� ������
% ���������� �� �������� ������, ��
x_s = X*Delta*100;

% ����������� ����, �
Temp = Y(:,l+3)*T0;

% �������� ����, �/�
v = Y(:,l+2)*v0;

% ������������, �^-3
n_i = Y(:,1:l)*n0;

% �������� ��������� ������, �^-3
n_a = Y(:,l+1)*n0;

% �������� ��������� �������, �^-3
n_m = sum(n_i,2);

time_s = X*Delta/v0; % sec

% �������� � ������������� ��������
Npoint = length(X);
Nall = sum(n_i,2)+n_a;

% ������������� ������������
ni_n = n_i./repmat(Nall,1,l);

% ������������� �������� ��������� �������
nm_n = sum(ni_n,2);

% ������������� �������� ��������� ������
na_n = n_a./Nall;

%% �������� �������������� ������
% ��������� �����, ��/�^3
rho = m(1)*n_m + m(2)*n_a;

% ��������, ��
p = Nall*k.*Temp;

% ������������� ������� �����, ��/�^3
e_v = repmat(e_i'+e_0,Npoint,1).*n_i;
e_v = sum(e_v,2);

e_v0 = n0*xc(1)/Zvibr_0*sum(exp(-e_i/Tv0/k).*(e_i+e_0));

% ������� ����������� �����, ��/�^3
e_f = 0.5*D*n_a*k;
e_f0 = 0.5*D*xc(2)*n0*k;

% �������������� ������� �����, ��/�^3
e_tr = 1.5*Nall*k.*Temp;
e_tr0 = 1.5*n0*k.*T0;

% ������������ ������� �����, ��/�^3
e_rot = n_m*k.*Temp;
e_rot0 = n0*xc(1)*k.*T0;

% ������ ������� ������� ������, ��/�^3
E = e_tr+e_rot+e_v+e_f;
E0 = e_tr0+e_rot0+e_v0+e_f0;

% ��������� �����, ��/��
H = (E+p)./rho;
H0 = (E0+p0)./rho0;

u10 = rho0*v0;
u20 = rho0*v0^2+p0;
u30 = H0+v0^2/2;

% ������� � ���
u1 = u10-rho.*v;
u2 = u20-rho.*v.^2-p;
u3 = u30-H-v.^2/2;

% ������������� �������
d1 = max(abs(u1)/u10);
d2 = max(abs(u2)/u20);
d3 = max(abs(u3)/u30);

% clc
disp('Relative error of conservation law of:');
disp(['mass = ',num2str(d1)]);
disp(['momentum = ',num2str(d2)]);
disp(['energy = ',num2str(d3)]);

%% �������� ������ ������
RDm = zeros(Npoint,l);
RDa = zeros(Npoint,l);
RVTm = zeros(Npoint,l);
RVTa = zeros(Npoint,l);
RVV = zeros(Npoint,l);

for i = 1:Npoint
    input = Y(i,:)';
    [rdm, rda, rvtm, rvta, rvv] = rpart_post(input); % �^-3*�^-1
    RDm(i,:) = rdm;
    RDa(i,:) = rda;
    RVTm(i,:) = rvtm;
    RVTa(i,:) = rvta;
    RVV(i,:) = rvv;
end

RD_mol = RDm+RDa;
RVT = RVTm+RVTa;
RD_at = -2*sum(RD_mol,2);

%dataset = [x_s, time_s, n_m, n_a, nm_n, na_n, Temp, rho, v, p, E];
%save dataset.dat dataset -ascii

dataset = [x_s, time_s, Temp, rho, p, v, E, ni_n, na_n, RD_mol, RD_at];
save solution_DR.dat dataset -ascii

%dataset = [x_s, time_s, Temp, RD_mol, RD_at];
%save datasetDR.dat dataset -ascii

%dataset = [x_s, time_s, Temp, RVT];
%save datasetVT.dat dataset -ascii

%dataset = [x_s, time_s, Temp, RVV];
%save datasetVV.dat dataset -ascii

toc
