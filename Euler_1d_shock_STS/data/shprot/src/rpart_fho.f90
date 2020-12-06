
 Module ComputeRHS
 use constants
 implicit none

 contains

 subroutine rpart_fho(neq, t, y, dy)

 use constants
 use init_energy
 use DissRec
 use DR_Module
 use Zeldovich
 use EX_Module
 use ExchangeVT
 use VT_Module
 use ExchangeVV
 use VV_Module
 use omp_lib

 implicit none

 integer,  intent(in)  :: neq
 real(WP), intent(in)  :: t
 real(WP), intent(in)  :: y(neq)
 real(WP), intent(out) :: dy(neq)

 real(WP) :: en20_b, eo20_b, eno0_b
 real(WP) :: nn_b, nno_b, no2_b, no_b, na_b, nm_b, nn2_b
 real(WP) :: v_b, T_b, temp, ktemp, f_1oktemp, tmp1, tmp2
 real(WP) :: nn_b_nn_b, no_b_no_b, nn_b_no_b

 integer :: i, j, iM, i1, i2, i3

 real(WP) :: start, stop

 real(WP) :: mb(5), Z_rot(3)
 real(WP), dimension(l1+l2+l3+4,l1+l2+l3+4) :: A = 0.0_WP

 real*8  :: Ainv(size(A,1),size(A,2)) = 0.
 real*8  :: work(size(A,1))            ! work array for LAPACK
 integer :: n,info,ipiv(size(A,1))     ! pivot indices

 real(WP) :: kvt_down(3, 5, l1+l2+l3) = Zero
 real(WP) :: kvv_down(3, l1+l2+l3-1, l1+l2+l3-1) = Zero

!***********************************************************************
 nn2i_b = y(1:l1)
 no2i_b = y(l1+1:l1+l2)
 nnoi_b = y(l1+l2+1:lall)
 nn_b   = y(lall+1)
 no_b   = y(lall+2)
 v_b    = y(lall+3)
 T_b    = y(lall+4)

 nn2_b  = sum(nn2i_b)
 no2_b  = sum(no2i_b)
 nno_b  = sum(nnoi_b)

 na_b   = nn_b+no_b
 nm_b   = sum(y(1:lall))

 temp   = T_b*T0

 en2i_b = en2_i*f_1okT0
 en20_b = en2_0*f_1okT0
 eo2i_b = eo2_i*f_1okT0
 eo20_b = eo2_0*f_1okT0
 enoi_b = eno_i*f_1okT0
 eno0_b = eno_0*f_1okT0

 Z_rot  = temp/(sigma*Theta_r)
 tMass  = sum(m)
 mb     = m/tMass

 ktemp  = k*temp
 f_1oktemp = One/ktemp

! **********************************************************************
! Building the A*X=B system
 do i=1,l1 ! N2
   A(i,i) = v_b
   A(i,lall + 3) = nn2i_b(i)
 end do
 do i=1,l2 ! O2
   A(l1+i,l1+i) = v_b
   A(l1+i,lall + 3) = no2i_b(i)
 end do
 do i=1,l3 ! NO
   A(l1+l2+i,l1+l2+i) = v_b
   A(l1+l2+i,lall + 3) = nnoi_b(i)
 end do
 do i=1,lall+2
   A(lall+3,i) = T_b
 end do
 do i=1,l1 ! N2
   A(lall+4,i) = f_5o2*T_b+en2i_b(i)+en20_b
 end do
 do i=1,l2 ! O2
   A(lall+4,l1+i) = f_5o2*T_b+eo2i_b(i)+eo20_b
 end do
 do i=1,l3 ! NO
   A(lall+4,l1+l2+i) = f_5o2*T_b+enoi_b(i)+eno0_b+efno_b
 end do
!N
 A(lall+1,lall+1) = v_b
 A(lall+1,lall+3) = nn_b
!O
 A(lall+2,lall+2) = v_b
 A(lall+2,lall+3) = no_b

 A(lall+3,lall+3) = tMass*v0*v0/(k*T0)*(mb(1)*nn2_b+mb(2)*no2_b+mb(3)*nno_b+mb(4)*nn_b+mb(5)*no_b)*v_b
 A(lall+3,lall+4) = nm_b+na_b

 A(lall+4,lall+1) = f_3o2*T_b+efn_b ! N
 A(lall+4,lall+2) = f_3o2*T_b+efo_b ! O

 A(lall+4,lall+3) = One/v_b*(3.5*nm_b*T_b+f_5o2*na_b*T_b + &
                    sum((en2i_b+en20_b)*nn2i_b) +          &
                    sum((eo2i_b+eo20_b)*no2i_b) +          &
                    sum((enoi_b+eno0_b)*nnoi_b) +          &
                    efno_b*nno_b+efn_b*nn_b+efo_b*no_b)

 A(lall+4,lall+4) = f_5o2*nm_b+f_3o2*na_b

! dissociation/recombination (DR) processes
  call compute_DR(temp)

! Zeldovich exchange reactions
  call compute_EX(temp)

! VT processes
  call compute_VT(temp)

! VV processes
  call compute_VV(temp)
  call compute_VVs(temp)

! update RHS
! call update_RHS()

 nn_b_nn_b = nn_b*nn_b
 no_b_no_b = no_b*no_b
 nn_b_no_b = nn_b*no_b

 do i1=1,l1

   RDn2(i1) = nn2_b*(nn_b_nn_b*kr_n2(1,i1)-nn2i_b(i1)*kd_n2(1,i1)) + &
              no2_b*(nn_b_nn_b*kr_n2(2,i1)-nn2i_b(i1)*kd_n2(2,i1)) + &
              nno_b*(nn_b_nn_b*kr_n2(3,i1)-nn2i_b(i1)*kd_n2(3,i1)) + &
              nn_b *(nn_b_nn_b*kr_n2(4,i1)-nn2i_b(i1)*kd_n2(4,i1)) + &
              no_b *(nn_b_nn_b*kr_n2(5,i1)-nn2i_b(i1)*kd_n2(5,i1))

   RZn2(i1) = sum(nnoi_b*nn_b*kb_n2(i1,:) - nn2i_b(i1)*no_b*kf_n2(i1,:))

   if (i1 == 1) then ! 0<->1

     RVTn2(i1) = nn2_b*(nn2i_b(i1+1)*kvt_down_n2(1,i1) - nn2i_b(i1)*kvt_up_n2(1,i1)) + &
                 no2_b*(nn2i_b(i1+1)*kvt_down_n2(2,i1) - nn2i_b(i1)*kvt_up_n2(2,i1)) + &
                 nno_b*(nn2i_b(i1+1)*kvt_down_n2(3,i1) - nn2i_b(i1)*kvt_up_n2(3,i1)) + &
                 nn_b *(nn2i_b(i1+1)*kvt_down_n2(4,i1) - nn2i_b(i1)*kvt_up_n2(4,i1)) + &
                 no_b *(nn2i_b(i1+1)*kvt_down_n2(5,i1) - nn2i_b(i1)*kvt_up_n2(5,i1))

     RVVn2(i1) = nn2i_b(i1+1)*sum(nn2i_b(1:size(nn2i_b)-1) * kvv_down_n2(i1,:)) - &
                 nn2i_b(i1)  *sum(nn2i_b(2:size(nn2i_b))   * kvv_up_n2(i1,:))

     RVVsn2(i1) = nn2i_b(i1+1)*(sum(no2i_b(1:size(no2i_b)-1) * kvvs_d_n2_o2(i1,:))) - &
                  nn2i_b(i1)  *(sum(no2i_b(2:size(no2i_b))   * kvvs_u_n2_o2(i1,:))) + &
                  nn2i_b(i1+1)*(sum(nnoi_b(1:size(nnoi_b)-1) * kvvs_d_n2_no(i1,:))) - &
                  nn2i_b(i1)  *(sum(nnoi_b(2:size(nnoi_b))   * kvvs_u_n2_no(i1,:)))

   else if (i1 == l1) then ! Lmax <-> Lmax-1

     RVTn2(i1) = nn2_b*(nn2i_b(i1-1)*kvt_up_n2(1,i1-1) - nn2i_b(i1)*kvt_down_n2(1,i1-1)) + &
                 no2_b*(nn2i_b(i1-1)*kvt_up_n2(2,i1-1) - nn2i_b(i1)*kvt_down_n2(2,i1-1)) + &
                 nno_b*(nn2i_b(i1-1)*kvt_up_n2(3,i1-1) - nn2i_b(i1)*kvt_down_n2(3,i1-1)) + &
                 nn_b *(nn2i_b(i1-1)*kvt_up_n2(4,i1-1) - nn2i_b(i1)*kvt_down_n2(4,i1-1)) + &
                 no_b *(nn2i_b(i1-1)*kvt_up_n2(5,i1-1) - nn2i_b(i1)*kvt_down_n2(5,i1-1))

     RVVn2(i1) = nn2i_b(i1-1)*sum(nn2i_b(2:size(nn2i_b))   * kvv_up_n2(i1-1,:)) - &
                 nn2i_b(i1)  *sum(nn2i_b(1:size(nn2i_b)-1) * kvv_down_n2(i1-1,:))

     RVVsn2(i1) = nn2i_b(i1-1)*(sum(no2i_b(2:size(no2i_b))   * kvvs_u_n2_o2(i1-1,:))) - &
                  nn2i_b(i1)  *(sum(no2i_b(1:size(no2i_b)-1) * kvvs_d_n2_o2(i1-1,:))) + &
                  nn2i_b(i1-1)*(sum(nnoi_b(2:size(nnoi_b))   * kvvs_u_n2_no(i1-1,:))) - &
                  nn2i_b(i1)  *(sum(nnoi_b(1:size(nnoi_b)-1) * kvvs_d_n2_no(i1-1,:)));
   else

     RVTn2(i1) =                                                                                                                &
     nn2_b*(nn2i_b(i1+1)*kvt_down_n2(1,i1)+nn2i_b(i1-1)*kvt_up_n2(1,i1-1) - nn2i_b(i1)*(kvt_up_n2(1,i1)+kvt_down_n2(1,i1-1))) + &
     no2_b*(nn2i_b(i1+1)*kvt_down_n2(2,i1)+nn2i_b(i1-1)*kvt_up_n2(2,i1-1) - nn2i_b(i1)*(kvt_up_n2(2,i1)+kvt_down_n2(2,i1-1))) + &
     nno_b*(nn2i_b(i1+1)*kvt_down_n2(3,i1)+nn2i_b(i1-1)*kvt_up_n2(3,i1-1) - nn2i_b(i1)*(kvt_up_n2(3,i1)+kvt_down_n2(3,i1-1))) + &
     nn_b *(nn2i_b(i1+1)*kvt_down_n2(4,i1)+nn2i_b(i1-1)*kvt_up_n2(4,i1-1) - nn2i_b(i1)*(kvt_up_n2(4,i1)+kvt_down_n2(4,i1-1))) + &
     no_b *(nn2i_b(i1+1)*kvt_down_n2(5,i1)+nn2i_b(i1-1)*kvt_up_n2(5,i1-1) - nn2i_b(i1)*(kvt_up_n2(5,i1)+kvt_down_n2(5,i1-1)))

     RVVn2(i1) = nn2i_b(i1+1)* sum(nn2i_b(1:size(nn2i_b)-1)  * kvv_down_n2(i1,:)) + &
                 nn2i_b(i1-1)* sum(nn2i_b(2:size(nn2i_b))    * kvv_up_n2(i1-1,:)) - &
                 nn2i_b(i1)  *(sum(nn2i_b(2:size(nn2i_b))    * kvv_up_n2(i1,:))   + &
                               sum(nn2i_b(1:size(nn2i_b)-1)  * kvv_down_n2(i1-1,:)))

     RVVsn2(i1) = nn2i_b(i1+1)*(sum(no2i_b(1:size(no2i_b)-1) * kvvs_d_n2_o2(i1,:)))   + &
                  nn2i_b(i1-1)*(sum(no2i_b(2:size(no2i_b))   * kvvs_u_n2_o2(i1-1,:))) - &
                  nn2i_b(i1)  *(sum(no2i_b(2:size(no2i_b))   * kvvs_u_n2_o2(i1,:))    + &
                                sum(no2i_b(1:size(no2i_b)-1) * kvvs_d_n2_o2(i1-1,:))) + &
                  nn2i_b(i1+1)*(sum(nnoi_b(1:size(nnoi_b)-1) * kvvs_d_n2_no(i1,:)))   + &
                  nn2i_b(i1-1)*(sum(nnoi_b(2:size(nnoi_b))   * kvvs_u_n2_no(i1-1,:))) - &
                  nn2i_b(i1)  *(sum(nnoi_b(2:size(nnoi_b))   * kvvs_u_n2_no(i1,:))    + &
                                sum(nnoi_b(1:size(nnoi_b)-1) * kvvs_d_n2_no(i1-1,:)))
   end if
 end do

 do i2=1,l2

   RDo2(i2) = nn2_b*(no_b_no_b*kr_o2(1,i2)-no2i_b(i2)*kd_o2(1,i2)) + &
              no2_b*(no_b_no_b*kr_o2(2,i2)-no2i_b(i2)*kd_o2(2,i2)) + &
              nno_b*(no_b_no_b*kr_o2(3,i2)-no2i_b(i2)*kd_o2(3,i2)) + &
              nn_b *(no_b_no_b*kr_o2(4,i2)-no2i_b(i2)*kd_o2(4,i2)) + &
              no_b *(no_b_no_b*kr_o2(5,i2)-no2i_b(i2)*kd_o2(5,i2))

   RZo2(i2) = sum(nnoi_b*no_b*kb_o2(i2,:)-no2i_b(i2)*nn_b*kf_o2(i2,:))

   if (i2 == 1) then ! 0<->1

     RVTo2(i2) = nn2_b*(no2i_b(i2+1)*kvt_down_o2(1,i2) - no2i_b(i2)*kvt_up_o2(1,i2)) + &
                 no2_b*(no2i_b(i2+1)*kvt_down_o2(2,i2) - no2i_b(i2)*kvt_up_o2(2,i2)) + &
                 nno_b*(no2i_b(i2+1)*kvt_down_o2(3,i2) - no2i_b(i2)*kvt_up_o2(3,i2)) + &
                 nn_b *(no2i_b(i2+1)*kvt_down_o2(4,i2) - no2i_b(i2)*kvt_up_o2(4,i2)) + &
                 no_b *(no2i_b(i2+1)*kvt_down_o2(5,i2) - no2i_b(i2)*kvt_up_o2(5,i2))

     RVVo2(i2) = no2i_b(i2+1)*sum(no2i_b(1:size(no2i_b)-1) * kvv_down_o2(i2,:)) - &
                 no2i_b(i2)  *sum(no2i_b(2:size(no2i_b))   * kvv_up_o2(i2,:))

     RVVso2(i2) = no2i_b(i2+1)*(sum(nn2i_b(1:size(nn2i_b)-1) * kvvs_d_o2_n2(i2,:))) - &
                  no2i_b(i2)  *(sum(nn2i_b(2:size(nn2i_b))   * kvvs_u_o2_n2(i2,:))) + &
                  no2i_b(i2+1)*(sum(nnoi_b(1:size(nnoi_b)-1) * kvvs_d_o2_no(i2,:))) - &
                  no2i_b(i2)  *(sum(nnoi_b(2:size(nnoi_b))   * kvvs_u_o2_no(i2,:)))

   else if (i2 == l2) then ! Lmax <-> Lmax-1

     RVTo2(i2) = nn2_b*(no2i_b(i2-1)*kvt_up_o2(1,i2-1) - no2i_b(i2)*kvt_down_o2(1,i2-1)) + &
                 no2_b*(no2i_b(i2-1)*kvt_up_o2(2,i2-1) - no2i_b(i2)*kvt_down_o2(2,i2-1)) + &
                 nno_b*(no2i_b(i2-1)*kvt_up_o2(3,i2-1) - no2i_b(i2)*kvt_down_o2(3,i2-1)) + &
                 nn_b *(no2i_b(i2-1)*kvt_up_o2(4,i2-1) - no2i_b(i2)*kvt_down_o2(4,i2-1)) + &
                 no_b *(no2i_b(i2-1)*kvt_up_o2(5,i2-1) - no2i_b(i2)*kvt_down_o2(5,i2-1))

     RVVo2(i2) = no2i_b(i2-1)*sum(no2i_b(2:size(no2i_b))   * kvv_up_o2(i2-1,:)) - &
                 no2i_b(i2)  *sum(no2i_b(1:size(no2i_b)-1) * kvv_down_o2(i2-1,:))

     RVVso2(i2) = no2i_b(i2-1)*(sum(nn2i_b(2:size(nn2i_b))   * kvvs_u_o2_n2(i2-1,:))) - &
                  no2i_b(i2)  *(sum(nn2i_b(1:size(nn2i_b)-1) * kvvs_d_o2_n2(i2-1,:))) + &
                  no2i_b(i2-1)*(sum(nnoi_b(2:size(nnoi_b))   * kvvs_u_o2_no(i2-1,:))) - &
                  no2i_b(i2)  *(sum(nnoi_b(1:size(nnoi_b)-1) * kvvs_d_o2_no(i2-1,:)))

   else

     RVTo2(i2) =                                                                                                                &
     nn2_b*(no2i_b(i2+1)*kvt_down_o2(1,i2)+no2i_b(i2-1)*kvt_up_o2(1,i2-1) - no2i_b(i2)*(kvt_up_o2(1,i2)+kvt_down_o2(1,i2-1))) + &
     no2_b*(no2i_b(i2+1)*kvt_down_o2(2,i2)+no2i_b(i2-1)*kvt_up_o2(2,i2-1) - no2i_b(i2)*(kvt_up_o2(2,i2)+kvt_down_o2(2,i2-1))) + &
     nno_b*(no2i_b(i2+1)*kvt_down_o2(3,i2)+no2i_b(i2-1)*kvt_up_o2(3,i2-1) - no2i_b(i2)*(kvt_up_o2(3,i2)+kvt_down_o2(3,i2-1))) + &
     nn_b *(no2i_b(i2+1)*kvt_down_o2(4,i2)+no2i_b(i2-1)*kvt_up_o2(4,i2-1) - no2i_b(i2)*(kvt_up_o2(4,i2)+kvt_down_o2(4,i2-1))) + &
     no_b *(no2i_b(i2+1)*kvt_down_o2(5,i2)+no2i_b(i2-1)*kvt_up_o2(5,i2-1) - no2i_b(i2)*(kvt_up_o2(5,i2)+kvt_down_o2(5,i2-1)))

     RVVo2(i2) = no2i_b(i2+1)* sum(no2i_b(1:size(no2i_b)-1)  * kvv_down_o2(i2,:)) + &
                 no2i_b(i2-1)* sum(no2i_b(2:size(no2i_b))    * kvv_up_o2(i2-1,:)) - &
                 no2i_b(i2)  *(sum(no2i_b(2:size(no2i_b))    * kvv_up_o2(i2,:))   + &
                               sum(no2i_b(1:size(no2i_b)-1)  * kvv_down_o2(i2-1,:)))

     RVVso2(i2) = no2i_b(i2+1)*(sum(nn2i_b(1:size(nn2i_b)-1) * kvvs_d_o2_n2(i2,:)))   + &
                  no2i_b(i2-1)*(sum(nn2i_b(2:size(nn2i_b))   * kvvs_u_o2_n2(i2-1,:))) - &
                  no2i_b(i2)  *(sum(nn2i_b(2:size(nn2i_b))   * kvvs_u_o2_n2(i2,:))    + &
                                sum(nn2i_b(1:size(nn2i_b)-1) * kvvs_d_o2_n2(i2-1,:))) + &
                  no2i_b(i2+1)*(sum(nnoi_b(1:size(nnoi_b)-1) * kvvs_d_o2_no(i2,:)))   + &
                  no2i_b(i2-1)*(sum(nnoi_b(2:size(nnoi_b))   * kvvs_u_o2_no(i2-1,:))) - &
                  no2i_b(i2)  *(sum(nnoi_b(2:size(nnoi_b))   * kvvs_u_o2_no(i2,:))    + &
                                sum(nnoi_b(1:size(nnoi_b)-1) * kvvs_d_o2_no(i2-1,:)))
   end if
 end do

 do i3=1,l3

   RDno(i3) = nn2_b*(nn_b_no_b*kr_no(1,i3)-nnoi_b(i3)*kd_no(1,i3)) + &
              no2_b*(nn_b_no_b*kr_no(2,i3)-nnoi_b(i3)*kd_no(2,i3)) + &
              nno_b*(nn_b_no_b*kr_no(3,i3)-nnoi_b(i3)*kd_no(3,i3)) + &
              nn_b *(nn_b_no_b*kr_no(4,i3)-nnoi_b(i3)*kd_no(4,i3)) + &
              no_b *(nn_b_no_b*kr_no(5,i3)-nnoi_b(i3)*kd_no(5,i3))

   RZno(i3) = sum(nn2i_b*no_b*kf_n2(:,i3)-nnoi_b(i3)*nn_b*kb_n2(:,i3)) + &
              sum(no2i_b*nn_b*kf_o2(:,i3)-nnoi_b(i3)*no_b*kb_o2(:,i3))

   if (i3 == 1) then ! 0<->1

     RVTno(i3) = nn2_b*(nnoi_b(i3+1)*kvt_down_no(1,i3) - nnoi_b(i3)*kvt_up_no(1,i3)) + &
                 no2_b*(nnoi_b(i3+1)*kvt_down_no(2,i3) - nnoi_b(i3)*kvt_up_no(2,i3)) + &
                 nno_b*(nnoi_b(i3+1)*kvt_down_no(3,i3) - nnoi_b(i3)*kvt_up_no(3,i3)) + &
                 nn_b *(nnoi_b(i3+1)*kvt_down_no(4,i3) - nnoi_b(i3)*kvt_up_no(4,i3)) + &
                 no_b *(nnoi_b(i3+1)*kvt_down_no(5,i3) - nnoi_b(i3)*kvt_up_no(5,i3))

     RVVno(i3) = nnoi_b(i3+1)*sum(nnoi_b(1:size(nnoi_b)-1) * kvv_down_no(i3,:)) - &
                 nnoi_b(i3)  *sum(nnoi_b(2:size(nnoi_b))   * kvv_up_no(i3,:))

     RVVsno(i3) = nnoi_b(i3+1)*(sum(nn2i_b(1:size(nn2i_b)-1) * kvvs_d_no_n2(i3,:))) - &
                  nnoi_b(i3)  *(sum(nn2i_b(2:size(nn2i_b))   * kvvs_u_no_n2(i3,:))) + &
                  nnoi_b(i3+1)*(sum(no2i_b(1:size(no2i_b)-1) * kvvs_d_no_o2(i3,:))) - &
                  nnoi_b(i3)  *(sum(no2i_b(2:size(no2i_b))   * kvvs_u_no_o2(i3,:)))

   else if (i3 == l3) then ! Lmax <-> Lmax-1

     RVTno(i3) = nn2_b*(nnoi_b(i3-1)*kvt_up_no(1,i3-1) - nnoi_b(i3)*kvt_down_no(1,i3-1)) + &
                 no2_b*(nnoi_b(i3-1)*kvt_up_no(2,i3-1) - nnoi_b(i3)*kvt_down_no(2,i3-1)) + &
                 nno_b*(nnoi_b(i3-1)*kvt_up_no(3,i3-1) - nnoi_b(i3)*kvt_down_no(3,i3-1)) + &
                 nn_b *(nnoi_b(i3-1)*kvt_up_no(4,i3-1) - nnoi_b(i3)*kvt_down_no(4,i3-1)) + &
                 no_b *(nnoi_b(i3-1)*kvt_up_no(5,i3-1) - nnoi_b(i3)*kvt_down_no(5,i3-1))

     RVVno(i3) = nnoi_b(i3-1)*sum(nnoi_b(2:size(nnoi_b))   * kvv_up_no(i3-1,:)) - &
                 nnoi_b(i3)  *sum(nnoi_b(1:size(nnoi_b)-1) * kvv_down_no(i3-1,:))

     RVVsno(i3) = nnoi_b(i3-1)*(sum(nn2i_b(2:size(nn2i_b))   * kvvs_u_no_n2(i3-1,:))) - &
                  nnoi_b(i3)  *(sum(nn2i_b(1:size(nn2i_b)-1) * kvvs_d_no_n2(i3-1,:))) + &
                  nnoi_b(i3-1)*(sum(no2i_b(2:size(no2i_b))   * kvvs_u_no_o2(i3-1,:))) - &
                  nnoi_b(i3)  *(sum(no2i_b(1:size(no2i_b)-1) * kvvs_d_no_o2(i3-1,:)))

   else

     RVTno(i3) =                                                                                                                &
     nn2_b*(nnoi_b(i3+1)*kvt_down_no(1,i3)+nnoi_b(i3-1)*kvt_up_no(1,i3-1) - nnoi_b(i3)*(kvt_up_no(1,i3)+kvt_down_no(1,i3-1))) + &
     no2_b*(nnoi_b(i3+1)*kvt_down_no(2,i3)+nnoi_b(i3-1)*kvt_up_no(2,i3-1) - nnoi_b(i3)*(kvt_up_no(2,i3)+kvt_down_no(2,i3-1))) + &
     nno_b*(nnoi_b(i3+1)*kvt_down_no(3,i3)+nnoi_b(i3-1)*kvt_up_no(3,i3-1) - nnoi_b(i3)*(kvt_up_no(3,i3)+kvt_down_no(3,i3-1))) + &
     nn_b *(nnoi_b(i3+1)*kvt_down_no(4,i3)+nnoi_b(i3-1)*kvt_up_no(4,i3-1) - nnoi_b(i3)*(kvt_up_no(4,i3)+kvt_down_no(4,i3-1))) + &
     no_b *(nnoi_b(i3+1)*kvt_down_no(5,i3)+nnoi_b(i3-1)*kvt_up_no(5,i3-1) - nnoi_b(i3)*(kvt_up_no(5,i3)+kvt_down_no(5,i3-1)))

     RVVno(i3) = nnoi_b(i3+1)* sum(nnoi_b(1:size(nnoi_b)-1)  * kvv_down_no(i3,:)) + &
                 nnoi_b(i3-1)* sum(nnoi_b(2:size(nnoi_b))    * kvv_up_no(i3-1,:)) - &
                 nnoi_b(i3)  *(sum(nnoi_b(2:size(nnoi_b))    * kvv_up_no(i3,:))   + &
                               sum(nnoi_b(1:size(nnoi_b)-1)  * kvv_down_no(i3-1,:)))

    RVVsno(i3) = nnoi_b(i3+1)*(sum(nn2i_b(1:size(nn2i_b)-1) * kvvs_d_no_n2(i3,:)))   + &
                 nnoi_b(i3-1)*(sum(nn2i_b(2:size(nn2i_b))   * kvvs_u_no_n2(i3-1,:))) - &
                 nnoi_b(i3)  *(sum(nn2i_b(2:size(nn2i_b))   * kvvs_u_no_n2(i3,:))    + &
                               sum(nn2i_b(1:size(nn2i_b)-1) * kvvs_d_no_n2(i3-1,:))) + &
                 nnoi_b(i3+1)*(sum(no2i_b(1:size(no2i_b)-1) * kvvs_d_no_o2(i3,:)))   + &
                 nnoi_b(i3-1)*(sum(no2i_b(2:size(no2i_b))   * kvvs_u_no_o2(i3-1,:))) - &
                 nnoi_b(i3)  *(sum(no2i_b(2:size(no2i_b))   * kvvs_u_no_o2(i3,:))    + &
                               sum(no2i_b(1:size(no2i_b)-1) * kvvs_d_no_o2(i3-1,:)))
   end if
 end do

 Brhs(1:l1)         = RDn2 + RZn2 + RVTn2 + RVVn2 + RVVsn2
 Brhs(l1+1:l1+l2)   = RDo2 + RZo2 + RVTo2 + RVVo2 + RVVso2
 Brhs(l1+l2+1:lall) = RDno + RZno + RVTno + RVVno + RVVsno
 Brhs(lall+1)       = - sum(RDno) - 2*sum(RDn2) - sum(RZn2) + sum(RZo2)
 Brhs(lall+2)       = - sum(RDno) - 2*sum(RDo2) + sum(RZn2) - sum(RZo2)

 dy = Zero ! just to be sure

 ! Store A in Ainv to prevent it from being overwritten by LAPACK
 Ainv = A
 n = size(A,1)
 call DGETRF(n,n,Ainv,n,ipiv,info)
 if (info.ne.0) stop 'Matrix is numerically singular!'
 call DGETRI(n,Ainv,n,ipiv,work,n,info)
 if (info.ne.0) stop 'Matrix inversion failed!'

!dy = matmul(Ainv, Brhs)
 CALL DGEMV('N', NEQ, NEQ, ONE, Ainv, NEQ, BRHS, 1, 0, DY, 1)

 end subroutine rpart_fho

 End module ComputeRHS
