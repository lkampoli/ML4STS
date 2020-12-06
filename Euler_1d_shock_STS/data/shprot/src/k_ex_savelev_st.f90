
! N2(i) + O = NO(k) + N
! O2(i) + N = NO(k) + O

 Module Zeldovich

 contains

 subroutine k_ex_savelev_st(T, kf_n2, kf_o2)
 use constants
!use init_energy
 implicit none

 real(WP), dimension(l1,l3), intent(out) :: kf_n2
 real(WP), dimension(l2,l3), intent(out) :: kf_o2
 real(WP),                   intent(in)  :: T

 real(WP) :: en2(l1), eo2(l2)

 real(WP), dimension(l3) :: eno, eno_eV,                        &
                            Ear_n2, Ear_n2_J, Ear_o2, Ear_o2_J, &
                            k_eq_n2,k_eq_o2,                    &
                            sum1_n2, sum1_o2, sum2_n2, sum2_o2, &
                            B1_n2, B2_n2, B1_o2, B2_o2,         &
                            C_n2, C_o2

 real(WP) :: Zv_n2, Zv_o2, kT, kU, f_1oTp1oU, f_1okT, f_1okU

 integer :: i, j, i_sw_n2, i_sw_o2, ln2, lno, lo2

 kT = k*T; f_1okT = One/kT
 kU = k*U_EX; f_1okU = One/kU
 f_1oTp1oU = One/T + One/U_EX

! call PopulateEnergy()

! vibr. energy, J
 en2 = en2_i + en2_0
 eo2 = eo2_i + eo2_0
 eno = eno_i + eno_0
 eno_eV = eno*6.242e18 ! eV

 Ear_n2 = 2.8793 + 1.02227*eno_eV ! eV
 Ear_n2_J = Ear_n2/6.242e18 ! J

 do i = 1,l3
   if(eno_eV(i) < 1.3706) then
     Ear_o2(i) = 0.098
   else if(eno_eV(i) > 1.3706 .and. eno_eV(i) < 2.4121) then
     Ear_o2(i) = -0.6521+0.54736*eno_eV(i)
   else if(eno_eV(i) > 2.4121) then
     Ear_o2(i) = -1.8451+1.04189*eno_eV(i)
   else
     write(*,*) " Something wrong with energy threshold level!"
   end if
 end do
 Ear_o2_J = Ear_o2/6.242e18 ! J

! equilibrium coefficient
 k_eq_n2 = AA(1) * (1+eno_eV/Three) * T**bb(1) * exp(-Ear_n2_J * f_1okT) ! m3/s
 k_eq_o2 = AA(2) * (Ear_o2+0.8)     * T**bb(2) * exp(-Ear_o2_J * f_1okT) ! m3/s

! vibr. partial function
 Zv_n2 = sum(exp(-en2*f_1okT))
 Zv_o2 = sum(exp(-eo2*f_1okT))

! energy threshold, for each k -> e_i
 sum1_n2 = 0.0_WP
 sum2_n2 = 0.0_WP
 sum1_o2 = 0.0_WP
 sum2_o2 = 0.0_WP

 do i=1,l3
   do j=1,l1
     if(en2(j) < Ear_n2_J(i)) then
       i_sw_n2 = j
     end if
   end do
   sum1_n2(i) = sum(exp(-(Ear_n2_J(i)-en2(1:i_sw_n2)) *f_1okU))
   sum2_n2(i) = sum(exp( (Ear_n2_J(i)-en2(i_sw_n2+1:))*f_1okT))
 end do
 do i=1,l3
   do j=1,l2
     if(eo2(j) < Ear_o2_J(i)) then
       i_sw_o2 = j
     end if
   end do
   sum1_o2(i) = sum(exp(-(Ear_o2_J(i)-eo2(1:i_sw_o2)) *f_1okU))
   sum2_o2(i) = sum(exp( (Ear_o2_J(i)-eo2(i_sw_o2+1:))*f_1okT))
 end do

! normalizing coefficient
 C_n2  = Zv_n2 * (sum1_n2+sum2_n2)**(-1)
!C_n2  = Zv_n2 * (One/(sum1_n2+sum2_n2))
 C_o2  = Zv_o2 * (sum1_o2+sum2_o2)**(-1)
!C_o2  = Zv_o2 * (One/(sum1_o2+sum2_o2))
 B1_n2 = C_n2  * k_eq_n2*exp(-Ear_n2_J*f_1okU)
 B2_n2 = C_n2  * k_eq_n2*exp(Ear_n2_J*f_1okT)
 B1_o2 = C_o2  * k_eq_o2*exp(-Ear_o2_J*f_1okU)
 B2_o2 = C_o2  * k_eq_o2*exp(Ear_o2_J*f_1okT)

 kf_n2 = 0.0_WP
 kf_o2 = 0.0_WP

 do i=1,l1
   do j=1,l3
     if(en2(i) < Ear_n2_J(j)) then
       !kf_n2(i,j) = (B1_n2(j)*exp(en2(i)/k*(1/T+1/U)))
       kf_n2(i,j) = (B1_n2(j)*exp(en2(i)/k*(f_1oTp1oU)))
     else if(en2(i) > Ear_n2_J(j)) then
       kf_n2(i,j) = B2_n2(j)
     else
       write(*,*) "As usual, shit happens!"
     end if
   end do
 end do
 do i=1,l2
   do j=1,l3
     if(eo2(i) < Ear_o2_J(j)) then
       !kf_o2(i,j) = (B1_o2(j)*exp(eo2(i)/k*(1/T+1/U)))
       kf_o2(i,j) = (B1_o2(j)*exp(eo2(i)/k*(f_1oTp1oU)))
     else if(eo2(i) > Ear_o2_J(j)) then
       kf_o2(i,j) = B2_o2(j)
     else
       write(*,*) "As usual, shit happens!"
     end if
   end do
 end do
 end subroutine k_ex_savelev_st

 End module Zeldovich
