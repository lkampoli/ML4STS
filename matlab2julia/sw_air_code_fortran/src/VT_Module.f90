module VT_Module

  use constants

  implicit none

  private
  public :: compute_VT

contains

  subroutine compute_VT(T)

  use constants
  use ExchangeVT

  implicit none

  real(WP), intent(in) :: T
  real(WP) :: tmp1, tmp2, kT, f_1oktemp, Z_rot(3)
  integer :: i, j, iM, i1, i2, i3

  kT = k*T
  f_1oktemp = One/(kT)
  Z_rot = T/(sigma*Theta_r)

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

  do iM=1,5
    do i1=1,Lmax1
       kvt_down_n2(iM,i1) = kvt_fho(1, iM, T, i1, i1-1) * Deltan0v0
    end do
  end do
  do iM=1,5
    do i2=1,Lmax2
       kvt_down_o2(iM,i2) = kvt_fho(2, iM, T, i2, i2-1) * Deltan0v0
    end do
  end do
  do iM=1,5
    do i3=1,Lmax3
       kvt_down_no(iM,i3) = kvt_fho(3, iM, T, i3, i3-1) * Deltan0v0
    end do
  end do

! VT: i -> i+1
  do j=1,Lmax1
    kvt_up_n2(:,j) = kvt_down_n2(:,j) * Kvt_n2(j)
  end do
  do j=1,Lmax2
    kvt_up_o2(:,j) = kvt_down_o2(:,j) * Kvt_o2(j)
  end do
  do j=1,Lmax3
    kvt_up_no(:,j) = kvt_down_no(:,j) * Kvt_no(j)
  end do
  end subroutine

end module
