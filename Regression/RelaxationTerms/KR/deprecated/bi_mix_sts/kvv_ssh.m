function kdown = kvv_ssh(t)
% rate coef-s of VV exchanges, A2 - A2
% kdown - array of k_(i->i-1)^(j->j+1)
% t - temperature, K

global c h k m om_e om_x_e l sw_o r0

mu = 0.5 * m(1); % reduced mass of A2-A2
m_osc = 0.25 * m(1); % reduced mass of osc-or

if sw_o == 1 % number of vibr. levels
    om10 = om_e-2*om_x_e; % lenear oscillation frequency, anh.os.
elseif sw_o == 2
    om10 = om_e; % har.os.
end

Lmax = l-1; % max level
om0 = 2*pi*c*om10;% circular oscillation frequency, sec^-1
%
a = 17.5 / r0(1); % inverse radius in 1st approximation m^-1
Z = r0(1)^2 / sqrt(mu) * sqrt(8*pi*k*t); % collision frequency, m^3/sec
%
Q10 = 0.5^4 / m_osc * a^2 / om0^2 * 4 * k * t;
k10 = Q10 * Z;

kdown = zeros(Lmax,Lmax);
j_up = 0:Lmax-1;
if sw_o == 2
    for i_down = 1:Lmax
	kdown(i_down,:) = i_down * (j_up+1) * k10;
    end
elseif sw_o == 1
    % anharmonicity factor for VV transitions
    aA = a * 1e-10; % A^-1
    mu_amu = mu / 1.6605e-27; % amu
    dE = om_x_e*1.4388e-2; % K
    delta = 0.427 / aA * sqrt(mu_amu / t) * dE; % Gorgietz book
    for i_down = 1:Lmax
        kdown(i_down,:) = i_down * (j_up+1) * k10 .* exp(-delta .* abs(i_down-1-j_up)) .* ...
            (1.5-0.5 * exp(-delta .* abs(i_down-1-j_up))) .* ...
            exp((j_up-i_down+1) * h * c * om_x_e / (k * t));
    end
end