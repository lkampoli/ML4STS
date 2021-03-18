MODULE DR_Module

contains

  subroutine compute_DR(T)

  use constants
  use DissRec

  implicit none

  real(WP), intent(in) :: T
  real(WP) :: Kdr_n2_constFac, Kdr_o2_constFac, Kdr_no_constFac, kT, f_1oktemp, Z_rot(3)
  integer :: i, j

  kT = k*T
  f_1oktemp = One/(kT)
  Z_rot = T/(sigma*Theta_r)

  Kdr_n2_constFac = (m(1)*h*h/(m(4)*m(4)*TwoPi*kT))**(f_3o2)*Z_rot(1)*exp(D(1)/T)
  Kdr_o2_constFac = (m(2)*h*h/(m(5)*m(5)*TwoPi*kT))**(f_3o2)*Z_rot(2)*exp(D(2)/T)
  Kdr_no_constFac = (m(3)*h*h/(m(4)*m(5)*TwoPi*kT))**(f_3o2)*Z_rot(3)*exp(D(3)/T)

  Kdr_n2(:) = exp(-en2_i(:)*f_1oktemp) * Kdr_n2_constFac
  Kdr_o2(:) = exp(-eo2_i(:)*f_1oktemp) * Kdr_o2_constFac
  Kdr_no(:) = exp(-eno_i(:)*f_1oktemp) * Kdr_no_constFac

! kb_VT(i-1->i) / kf_VT(i->i-1)
  do i = 1,l1-1
    Kvt_n2(i) = exp((en2_i(i)-en2_i(i+1))*f_1oktemp)
  end do
  do i = 1,l2-1
    Kvt_o2(i) = exp((eo2_i(i)-eo2_i(i+1))*f_1oktemp)
  end do
  do i = 1,l3-1
    Kvt_no(i) = exp((eno_i(i)-eno_i(i+1))*f_1oktemp)
  end do

  kd_n2 = kdis(T,1) * Deltan0v0
  kd_o2 = kdis(T,2) * Deltan0v0
  kd_no = kdis(T,3) * Deltan0v0

  do j=1,l1
    kr_n2(:,j) = kd_n2(:,j) * Kdr_n2(j) * n0
  enddo
  do j=1,l2
    kr_o2(:,j) = kd_o2(:,j) * Kdr_o2(j) * n0
  enddo
  do j=1,l3
    kr_no(:,j) = kd_no(:,j) * Kdr_no(j) * n0
  enddo
  end subroutine

END MODULE
