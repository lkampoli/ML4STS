function e = en_vibr_0(sp)
% ������� ������� ������������� ������� �������� �� 0� ������
format long e
global c h om_e om_x_e sw_o

if sw_o == 1
    e = h*c*(0.5*om_e(sp)-0.25*om_x_e(sp)); % ���.�.
else
    e = h*c*0.5*om_e(sp); % �.�.
end
