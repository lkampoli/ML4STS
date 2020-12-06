
 Module init_energy
 use constants
 implicit none

 real(WP) :: Zv0_n2, Zv0_o2, Zv0_no
!real(WP) :: en2_0, eo2_0, eno_0
 real(WP) :: om_0, om_x_e_0
!real(WP), dimension(l(1)) :: en2_i
!real(WP), dimension(l(2)) :: eo2_i
!real(WP), dimension(l(3)) :: eno_i
 real(WP), dimension(l(1),l(3)) :: eno_i_en2_i
 real(WP), dimension(l(1),l(3)) :: en2_i_eno_i
 real(WP), dimension(l(2),l(3)) :: eno_i_eo2_i
 real(WP), dimension(l(2),l(3)) :: eo2_i_eno_i

 contains

 subroutine PopulateEnergy()
 use constants
 implicit none
 integer :: i
! if (sw_o == 1) then
!   om_0 = om_e(1)-om_x_e(1)
!   om_x_e_0 = om_x_e(1)
!   en2_i(1) = 0.
!   do i = 1, size(en2_i)-1
!     en2_i(i+1) = h*c*(om_0*i-om_x_e_0*i*i)
!   end do
!   om_0 = om_e(2)-om_x_e(2)
!   om_x_e_0 = om_x_e(2)
!   eo2_i(1) = 0.
!   do i = 1, size(eo2_i)-1
!     eo2_i(i+1) = h*c*(om_0*i-om_x_e_0*i*i)
!   end do
!   om_0 = om_e(3)-om_x_e(3)
!   om_x_e_0 = om_x_e(3)
!   eno_i(1) = 0.
!   do i = 1, size(eno_i)-1
!     eno_i(i+1) = h*c*(om_0*i-om_x_e_0*i*i)
!   end do
! else
!   do i = 1, size(en2_i)-1
!     en2_i(i+1) = h*c*om_e(1)*i
!   end do
!   do i = 1, size(eo2_i)-1
!     eo2_i(i+1) = h*c*om_e(2)*i
!   end do
!   do i = 1, size(eno_i)-1
!     eno_i(i+1) = h*c*om_e(3)*i
!   end do
! end if
! if (sw_o == 1) then
!   en2_0 = h*c*(f_1o2*om_e(1)-f_1o4*om_x_e(1))
!   eo2_0 = h*c*(f_1o2*om_e(2)-f_1o4*om_x_e(2))
!   eno_0 = h*c*(f_1o2*om_e(3)-f_1o4*om_x_e(3))
! else
!   en2_0 = h*c*f_1o2*om_e(1)
!   eo2_0 = h*c*f_1o2*om_e(2)
!   eno_0 = h*c*f_1o2*om_e(3)
! end if
 end subroutine

 subroutine ComputePartitionFunctions()
 use constants
 implicit none
 integer :: i
 Zv0_n2 = Zero
 Zv0_o2 = Zero
 Zv0_no = Zero
 do i = 1, size(en2_i)
    Zv0_n2 = Zv0_n2 + (exp(-en2_i(i)*f_1oTv0n2k))
 end do
 do i = 1, size(eo2_i)
    Zv0_o2 = Zv0_o2 + (exp(-eo2_i(i)*f_1oTv0o2k))
 end do
 do i = 1, size(eno_i)
    Zv0_no = Zv0_no + (exp(-eno_i(i)*f_1oTv0nok))
 end do
 end subroutine

 End module init_energy
