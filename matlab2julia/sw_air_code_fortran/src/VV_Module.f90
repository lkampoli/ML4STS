module VV_Module

  use constants

  implicit none

  private
  public :: compute_VV, compute_VVs

contains

  subroutine compute_VV(T)

  use constants
  use ExchangeVV

  implicit none

  real(WP), intent(in) :: T
  real(WP) :: tmp1, tmp2, kT, f_1oktemp
  integer :: i, j, iM, i1, i2, i3

  kT = k*T
  f_1oktemp = One/(kT)

! VV (i,j) -> (i-1,j+1)
  do i=1,Lmax1
    do j=1,Lmax1
      kvv_down_n2(i,j) = kvv_fho(1, 1, T, i, j-1, i-1, j) * Deltan0v0
    end do
  end do
  do i=1,Lmax2
    do j=1,Lmax2
       kvv_down_o2(i,j) = kvv_fho(2, 2, T, i, j-1, i-1, j) * Deltan0v0
    end do
  end do
  do i=1,Lmax3
    do j=1,Lmax3
       kvv_down_no(i,j) = kvv_fho(3, 3, T, i, j-1, i-1, j) * Deltan0v0
    end do
  end do

! VV (i-1,j+1) -> (i,j)
  do i=1,Lmax1
    deps_n2(i) = en2_i(i)-en2_i(i+1)
  end do
  do i=1,Lmax2
    deps_o2(i) = eo2_i(i)-eo2_i(i+1)
  end do
  do i=1,Lmax3
    deps_no(i) = eno_i(i)-eno_i(i+1)
  end do
  do j=1,Lmax1
     kvv_up_n2(j,:) = kvv_down_n2(j,:) * exp((deps_n2(j)-deps_n2(:)) * f_1oktemp)
  end do
  do j=1,Lmax2
     kvv_up_o2(j,:) = kvv_down_o2(j,:) * exp((deps_o2(j)-deps_o2(:)) * f_1oktemp)
  end do
  do j=1,Lmax3
     kvv_up_no(j,:) = kvv_down_no(j,:) * exp((deps_no(j)-deps_no(:)) * f_1oktemp)
  end do
  end subroutine

  subroutine compute_VVs(T)

  use constants
  use ExchangeVV

  implicit none

  real(WP), intent(in) :: T
  real(WP) :: tmp1, tmp2, kT, f_1oktemp
  integer :: i, j, iM, i1, i2, i3

  kT = k*T
  f_1oktemp = One/(kT)

! VV' (i,j) -> (i-1,j+1)
  do i = 1,Lmax1
    do j = 1,Lmax2
      kvvs_d_n2_o2(i,j) = kvv_fho(1, 2, T, i, j-1, i-1, j) * Deltan0v0
    enddo
  enddo
  do i = 1,Lmax1
    do j = 1,Lmax3
      kvvs_d_n2_no(i,j) = kvv_fho(1, 3, T, i, j-1, i-1, j) * Deltan0v0
    enddo
  enddo
  do i = 1,Lmax2
    do j = 1,Lmax1
      kvvs_d_o2_n2(i,j) = kvv_fho(2, 1, T, i, j-1, i-1, j) * Deltan0v0
    enddo
  enddo
  do i = 1,Lmax2
    do j = 1,Lmax3
      kvvs_d_o2_no(i,j) = kvv_fho(2, 3, T, i, j-1, i-1, j) * Deltan0v0
    enddo
  enddo
  do i = 1,Lmax3
    do j = 1,Lmax1
      kvvs_d_no_n2(i,j) = kvv_fho(3, 1, T, i, j-1, i-1, j) * Deltan0v0
    enddo
  enddo
  do i = 1,Lmax3
    do j = 1,Lmax2
      kvvs_d_no_o2(i,j) = kvv_fho(3, 2, T, i, j-1, i-1, j) * Deltan0v0
    enddo
  enddo

! VV' (i-1,j+1) -> (i,j)
  do i = 1,Lmax1
    kvvs_u_n2_o2(i,:) = kvvs_d_n2_o2(i,:) * exp((deps_n2(i)-deps_o2(:)) * f_1oktemp)
    kvvs_u_n2_no(i,:) = kvvs_d_n2_no(i,:) * exp((deps_n2(i)-deps_no(:)) * f_1oktemp)
  enddo
  do i = 1,Lmax2
    kvvs_u_o2_n2(i,:) = kvvs_d_o2_n2(i,:) * exp((deps_o2(i)-deps_n2(:)) * f_1oktemp)
    kvvs_u_o2_no(i,:) = kvvs_d_o2_no(i,:) * exp((deps_o2(i)-deps_no(:)) * f_1oktemp)
  enddo
  do i = 1,Lmax3
    kvvs_u_no_n2(i,:) = kvvs_d_no_n2(i,:) * exp((deps_no(i)-deps_n2(:)) * f_1oktemp)
    kvvs_u_no_o2(i,:) = kvvs_d_no_o2(i,:) * exp((deps_no(i)-deps_o2(:)) * f_1oktemp)
  enddo
  end subroutine

end module
