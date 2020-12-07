module EX_Module

  use constants

  implicit none

  private
  public :: compute_EX

  real(WP), dimension(l1,l3), public :: kf_n2, kb_n2
  real(WP), dimension(l2,l3), public :: kf_o2, kb_o2

contains

  subroutine compute_EX(T)

  use constants
  use Zeldovich

  implicit none

  real(WP), intent(in) :: T
  real(WP) :: tmp1, tmp2, kT, f_1oktemp, Z_rot(3)
  integer :: i, j

  kT = k*T
  f_1oktemp = One/(kT)
  Z_rot = T/(sigma*Theta_r)

  tmp1 = (m(1)*m(5)/(m(3)*m(4)))**f_3o2*Z_rot(1)/Z_rot(3) * exp((D(1)-D(3))/T)
  tmp2 = (m(2)*m(4)/(m(3)*m(5)))**f_3o2*Z_rot(2)/Z_rot(3) * exp((D(2)-D(3))/T)

  do j = 1,l3
    Kz_n2(:,j) = exp((eno_i(j)-en2_i(:))*f_1oktemp) * tmp1
  end do
  do j = 1,l3
    Kz_o2(:,j) = exp((eno_i(j)-eo2_i(:))*f_1oktemp) * tmp2
  end do

  call k_ex_savelev_st(T, kf_n2, kf_o2)

  do j = 1,l3
    kf_n2(:,j) = kf_n2(:,j) * Deltan0v0
    kb_n2(:,j) = kf_n2(:,j) * Kz_n2(:,j)
  end do
  do j = 1,l3
    kf_o2(:,j) = kf_o2(:,j) * Deltan0v0
    kb_o2(:,j) = kf_o2(:,j) * Kz_o2(:,j)
  end do
  end subroutine

end module
