
  Program shprot

  use kind_module
  use constants
  use dvode_f90_m
  use init_energy
  use ComputeRHS
! use ifport
! use iso_fortran_env, only: stdout => output_unit, &
!                             stderr => error_unit
  implicit none

  integer :: unit, i, j, ierror, Npoint

  real(WP) :: a0, M0, mu0_mix, R_bar, rho0, v_0, n1, v1, T1, C1, C2, &
              in_con(3), rho0_c(5),                                  &
              xc(5), Y0_bar(126)
!             xc(5) = 0.0_WP,                                        &
!             Y0_bar(l1+l2+l3+4) = 0.0_WP

 real(WP), dimension(:), allocatable :: time_s,   &
                                        time_mcs, &
                                        temp,     &
                                        v,        &
                                        nn2,      &
                                        no2,      &
                                        nno,      &
                                        nn_,      &
                                        no_,      &
                                        n_at,     &
                                        n_mol,    &
                                        nn2_n,    &
                                        no2_n,    &
                                        nno_n,    &
                                        nn_n,     &
                                        no_n,     &
                                        Nall

 real(WP), dimension(:,:), allocatable :: nn2_i,  &
                                          no2_i,  &
                                          nno_i,  &
                                          nn2i_n, &
                                          no2i_n, &
                                          nnoi_n, &
                                          ev_n2,  &
                                          ev_o2,  &
                                          ev_no

 real(WP), dimension(:), allocatable :: rho,                    &
                                        p,                      &
                                        Tvn2, Tvo2, Tvno,       &
                                        u1, u2, u3,             &
                                        e_tr, e_rot, e_v, e_f,  &
                                        enthalpy

 real(WP) :: e_v0, e_f0, u10, u20, u30, d1, d2, d3

! ... settings for odepack ... (read documentation)
 type(vode_opts) :: options

 external jac2

 integer :: itask, istate

!integer, parameter :: neq    = sum(l)+4
 integer, parameter :: neq    = l1+l2+l3+4
!integer            :: neq    != sum(l)+4
 integer, parameter :: nnz    = neq
! dlsodes***************************************************
!integer, parameter :: liw    = 31 + neq + 200 * neq
!integer, parameter :: liw    = 31 + neq + nnz
!integer, parameter :: liw    = 30 ! use this for dsolves
! dlsode****************************************************
 integer, parameter :: liw    = 20 + neq
! **********************************************************
 integer, parameter :: miter  = 2
 integer, parameter :: nyh    = neq !initial value of neq
 integer, parameter :: maxord = 5 !12 (if meth = 1) or 5 (if meth = 2)
 real*8 , parameter :: lenrat = 2
 integer, parameter :: lwm    = 2*nnz + 2*neq + (nnz+10*neq)/lenrat
! dlsodes***************************************************
!integer, parameter :: lrw    = 20 +  9 * neq + 2000 * neq ! use this for dlsodes
!integer, parameter :: lrw    = 20 +  9 * neq + lwm
!integer, parameter :: lrw    = 20 + nyh*(maxord + 1) + 3*neq + lwm
! dlsode****************************************************
 integer, parameter :: mlo    = 1
 integer, parameter :: mup    = 1
 integer, parameter :: lrw    = 22 +  9*neq + neq**2            ! MF=22
!integer, parameter :: lrw    = 22 + 10*neq + (2*mlo + mup)*neq ! MF=25
! **********************************************************
 integer, parameter :: mf     = 222
 integer, parameter :: itol   = 1 ! 1 or 2 according as atol is a scalar or array
 integer, parameter :: iopt   = 0 ! 0 to indicate no optional inputs used

 double precision :: t, tout, rtol
 double precision :: rwork(lrw), atolV(neq), atolS, rstats(22)!, ruser(22)
 integer :: iwork(liw), istats(31)!, iuser(31)
!************************************************************************

 double precision :: Xtin, Xtout
 real(WP) :: ysol(neq) !,t
 double precision dy(neq)

 double precision, allocatable :: abstol(:)

! number of integration points at which the ODE system will be solved
 integer, parameter :: npoints = size(xstep)

! Solution vector of all varibles at all steps
 real(WP), dimension(npoints)      :: xout
 real(WP), dimension(npoints, neq) :: y

! ... timing ...
 real(WP) :: start, stop

! ... outputting ...
 integer :: nout !channel number

! ****************************************
 character(20)  :: XXX
 character(75)  :: origin
 character(75)  :: path
 character(9)   :: base
 character(20)  :: compute_XY
 character(132) :: command_compute_XY
! ***************************************

 xc(5) = Zero
 Y0_bar(l1+l2+l3+4) = Zero

!call PopulateEnergy()
 call ComputePartitionFunctions()

 xc(1) = incon(1)
 xc(2) = incon(2)
 v_0   = incon(3)*Thousand

 rho0_c  = m*xc*n0
 rho0    = sum(rho0_c)
 mu0_mix = sum(rho0_c/mu)/rho0
 R_bar   = R*mu0_mix
 a0      = sqrt(gamma0*R_bar*T0)
 M0      = v_0/a0

 C1 = v_0*v_0*sum(xc*m)/(k*T0)
 C2 = f_1o2*C1

!TODO: also this should be a call
!but for the moment just copy from
!matlab implementation
!NN = in_con
 n1 = 5.761793697284453e+00 !NN(1)
 v1 = 1.735570644383367e-01 !NN(2)
 T1 = 2.445965060176968e+01 !NN(3)
! ******************************************************************

! Initialization
 Y0_bar(1:l(1))             = xc(1)*n1/Zv0_n2*exp(-en2_i/(Tv0n2*k)) ! N2
 Y0_bar(l(1)+1:l(1)+l(2))   = xc(2)*n1/Zv0_o2*exp(-eo2_i/(Tv0o2*k)) ! O2
 Y0_bar(l(1)+l(2)+1:sum(l)) = xc(3)*n1/Zv0_no*exp(-eno_i/(Tv0no*k)) ! NO
 Y0_bar(sum(l)+1)           = xc(4)                                 ! N
 Y0_bar(sum(l)+2)           = xc(5)                                 ! O
 Y0_bar(sum(l)+3)           = v1
 Y0_bar(sum(l)+4)           = T1

 y(1,:) = Y0_bar

! Solve the ODE system
 write(*,*) "Solving the ODE system ..."

 ysol  = Zero
 ysol  = Y0_bar
 Xtin  = Zero
 Xtout = x_w/Delta
 Xtout = Xtout/npoints

 if (.not. allocated(abstol)) then
   allocate(abstol(neq))
 end if
 abstol = 1.0d-8

 RTOL  = 1.0D-8
 ATOLS = 1.0D-8 ! scalar
 ATOLV = 1.0D-8 ! vector

 base = " python3 "
 path = "/home/lk/Public/ML4STS/Euler_1d_shock_STS/data/sw_air_code_fortran/DT_XY/"

 call cpu_time(start)
! read file is not threadsafe!
!!$omp parallel private(i,unit)
!!$omp do
 do i = 1, npoints-1

   itask  = 1 ! try to integrate to target time/space
   istate = 1 ! pretend every step is the first

   ! exactly the same steps of the Matlab
   ! implementation are used in order to
   ! fairly benchmark
   xtout = xstep(i+1)

   write(*,*) "npoint #", i
!!!
  options = set_opts(method_flag=22, abserr=1.0d-8, relerr=1.0d-8)
  call dvode_f90(rpart_fho, neq, ysol, xtin, xtout, itask, istate, options)
!  !call get_stats(rstats,istats)
!
!60  FORMAT(/'  No. steps =',I4,'   No. f-s =',I4,        &
!            '  No. J-s =',I4,'   No. LU-s =',I4/         &
!            '  No. nonlinear iterations =',I4/           &
!            '  No. nonlinear convergence failures =',I4/ &
!            '  No. error test failures =',I4/            &
!            '  No. g-s =',I4/)
!
!90 format(///' error halt... istate =',i3)
!
!70  FORMAT(//' Required RWORK size =',I8,'   IWORK size =',I4 ' No. steps =',I4,'   No. f-s =',I4,'   No. J-s =',I4, &
!'   No. LU-s =',I4/' No. of nonzeros in J =',I5, '   No. of nonzeros in LU =',I5)
!!!
   !***********************************************
   ! Fire the NN instead of calling the ODE solver
   !***********************************************
!!!
!   write( XXX, '(f20.10)' ) xtout
!   call getcwd(origin)
!   call chdir(path)
!   compute_XY = "run_regression_XY.py"
!   command_compute_XY = base//compute_XY//" "//XXX
!   call execute_command_line (command_compute_XY)
!   open(newunit=unit, file='result_XY.out', action='read')
!   !open(10, file='result_XY.bin', access='stream')
!   !open(10, file="result_XY.unf", form="unformatted")
!   read (unit, *) (ysol(j), j=1, size(ysol) - 1)
!   !read (10, *) ysol
!   !read (10) ysol
!   !close(10)
!   close(unit)
!   call chdir(origin)
!!!
  xout(i)  = xtout
  y(i+1,:) = ysol

 end do
!!$omp end parallel
 call cpu_time(stop)
 write(*,*) "Elapsed time: ", stop-start, "seconds"
 print '("Time = ",f6.3," seconds.")',stop-start

 xout = xout*Delta*100

 allocate(temp(size(y, dim=1)), stat=ierror)
 if (ierror /= 0) stop "problems allocating temp"
 allocate(v(size(y, dim=1)), stat=ierror)
 if (ierror /= 0) stop "problems allocating v"
 allocate(nn2_i(size(y, dim=1),l(1)), stat=ierror)
 if (ierror /= 0) stop "problems allocating nn2_i"
 allocate(no2_i(size(y, dim=1),2*l(1)+l(2)), stat=ierror)
 if (ierror /= 0) stop "problems allocating no2_i"
 allocate(nno_i(size(y, dim=1),l(1)+l(2)+sum(l)), stat=ierror)
 if (ierror /= 0) stop "problems allocating nno_i"
 allocate(nn_(size(y, dim=1)), stat=ierror)
 if (ierror /= 0) stop "problems allocating nn_"
 allocate(no_(size(y, dim=1)), stat=ierror)
 if (ierror /= 0) stop "problems allocating no_"
 allocate(time_s(size(y, dim=1)), stat=ierror)
 if (ierror /= 0) stop "problems allocating time_s"
 allocate(time_mcs(size(y, dim=1)), stat=ierror)
 if (ierror /= 0) stop "problems allocating time_mcs"
 allocate(nall(size(y, dim=1)), stat=ierror)
 if (ierror /= 0) stop "problems allocating nall"
 allocate(nn2i_n(size(y, dim=1),l(1)), stat=ierror)
 if (ierror /= 0) stop "problems allocating nn2i_n"
 allocate(no2i_n(size(y, dim=1),l(2)), stat=ierror)
 if (ierror /= 0) stop "problems allocating no2i_n"
 allocate(nnoi_n(size(y, dim=1),l(3)), stat=ierror)
 if (ierror /= 0) stop "problems allocating nnoi_n"
 allocate(nn2_n(size(y, dim=1)), stat=ierror)
 if (ierror /= 0) stop "problems allocating nn2_n"
 allocate(no2_n(size(y, dim=1)), stat=ierror)
 if (ierror /= 0) stop "problems allocating no2_n"
 allocate(nno_n(size(y, dim=1)), stat=ierror)
 if (ierror /= 0) stop "problems allocating nno_n"
 allocate(nn_n(size(y, dim=1)), stat=ierror)
 if (ierror /= 0) stop "problems allocating nn_n"
 allocate(no_n(size(y, dim=1)), stat=ierror)
 if (ierror /= 0) stop "problems allocating no_n"
 allocate(rho(size(y, dim=1)), stat=ierror)
 if (ierror /= 0) stop "problems allocating rho"
 allocate(p(size(y, dim=1)), stat=ierror)
 if (ierror /= 0) stop "problems allocating p"
 allocate(tvn2(size(y, dim=1)), stat=ierror)
 if (ierror /= 0) stop "problems allocating tvn2"
 allocate(tvo2(size(y, dim=1)), stat=ierror)
 if (ierror /= 0) stop "problems allocating tvo2"
 allocate(tvno(size(y, dim=1)), stat=ierror)
 if (ierror /= 0) stop "problems allocating tvno"
 allocate(e_f(size(y, dim=1)), stat=ierror)
 if (ierror /= 0) stop "problems allocating e_f"
 allocate(ev_n2(size(y, dim=1), l(1)), stat=ierror)
 if (ierror /= 0) stop "problems allocating ev_n2"
 allocate(ev_o2(size(y, dim=1), l(2)), stat=ierror)
 if (ierror /= 0) stop "problems allocating ev_o2"
 allocate(ev_no(size(y, dim=1), l(3)), stat=ierror)
 if (ierror /= 0) stop "problems allocating ev_no"
 allocate(e_v(size(y, dim=1)), stat=ierror)
 if (ierror /= 0) stop "problems allocating e_v"
 allocate(e_tr(size(y, dim=1)), stat=ierror)
 if (ierror /= 0) stop "problems allocating e_tr"
 allocate(e_rot(size(y, dim=1)), stat=ierror)
 if (ierror /= 0) stop "problems allocating e_rot"
 allocate(enthalpy(size(y, dim=1)), stat=ierror)
 if (ierror /= 0) stop "problems allocating h"
 allocate(u1(size(y, dim=1)), stat=ierror)
 if (ierror /= 0) stop "problems allocating u1"
 allocate(u2(size(y, dim=1)), stat=ierror)
 if (ierror /= 0) stop "problems allocating u2"
 allocate(u3(size(y, dim=1)), stat=ierror)
 if (ierror /= 0) stop "problems allocating u3"

! time steps
 time_s   = xout*Delta/v_0 ! sec
 time_mcs = time_s*1e6 ! mcsec

! temperature
 temp = y(:,sum(l)+4)*t0

! velocity
 v = y(:,sum(l)+3)*v_0

! species molar fractions
 nn2_i = y(:,1:l(1))*n0
 no2_i = y(:,l(1)+1:l(1)+l(2))*n0
 nno_i = y(:,l(1)+l(2)+1:sum(l))*n0

 nn2 = sum(nn2_i, dim=2)
 no2 = sum(no2_i, dim=2)
 nno = sum(nno_i, dim=2)

 nn_   = y(:,sum(l)+1)*n0 ! N
 no_   = y(:,sum(l)+2)*n0 ! O
 n_at  = nn_+no_
 n_mol = nn2+no2+nno

 Npoint = size(xout)
 Nall   = n_mol+n_at
 nn2i_n = nn2_i/spread(Nall,2,l(1))
 no2i_n = no2_i/spread(Nall,2,l(2))
 nnoi_n = nno_i/spread(Nall,2,l(3))

 nn2_n = sum(nn2i_n,dim=2)
 no2_n = sum(no2i_n,dim=2)
 nno_n = sum(nnoi_n,dim=2)
 nn_n  = nn_/Nall
 no_n  = no_/Nall

! density and pressure
 rho = m(1)*sum(nn2_i,2)+m(2)*sum(no2_i,2)+m(3)*sum(nno_i,2)+m(4)*nn_+m(5)*no_
 p   = (n_mol+n_at)*k*Temp

! vibrational temperature
 Tvn2 = en2_i(2)/(k*log(nn2i_n(:,1)/nn2i_n(:,2)))
 Tvo2 = eo2_i(2)/(k*log(no2i_n(:,1)/no2i_n(:,2)))
 Tvno = eno_i(2)/(k*log(nnoi_n(:,1)/nnoi_n(:,2)))

! vibrational energies
 ev_n2 = spread(en2_i+en2_0,1,Npoint)*nn2_i
 ev_o2 = spread(eo2_i+eo2_0,1,Npoint)*no2_i
 ev_no = spread(eno_i+eno_0,1,Npoint)*nno_i

! energy components
 e_tr  = f_3o2*k*(n_mol+n_at)*Temp
 e_rot = n_mol*k*Temp
 e_v   = sum(ev_n2,2)+sum(ev_o2,2)+sum(ev_no,2)
 e_f   = f_1o2*k*(D(1)*nn_+D(2)*no_)+k*(f_1o2*(D(1)+D(2))-D(3))*sum(nno_i,2)

 enthalpy = (f_7o2*n_mol*k*Temp+f_5o2*n_at*k*Temp+e_v+e_f)/rho

 e_v0 = n0*(xc(1)/Zv0_n2*sum(exp(-en2_i/(Tv0n2*k))*(en2_i+en2_0)) + &
            xc(2)/Zv0_o2*sum(exp(-eo2_i/(Tv0o2*k))*(eo2_i+eo2_0)) + &
            xc(3)/Zv0_no*sum(exp(-eno_i/(Tv0no*k))*(eno_i+eno_0)))

 e_f0 = f_1o2*k*(D(1)*xc(4)*n0+D(2)*xc(5)*n0)+k*(f_1o2*(D(1)+D(2))-D(3))*xc(3)*n0

 u10 = rho0*v_0
 u20 = rho0*v_0*v_0+p0
 u30 = (f_7o2*(sum(xc(1:3)))*n0*k*T0+f_5o2*(sum(xc(4:5)))*n0*k*T0+e_v0+e_f0)/rho0+v_0*v_0*f_1o2
 u1  = u10-rho*v
 u2  = u20-rho*v*v-p
 u3  = u30-enthalpy-v*v*f_1o2

 d1 = maxval(abs(u1)/u10)
 d2 = maxval(abs(u2)/u20)
 d3 = maxval(abs(u3)/u30)

 if ((d1>tol).or.(d2>tol).or.(d3>tol)) then
   write(*,*) "Big error!"
 end if

! Write output files
 call execute_command_line('rm T.dat')
 call execute_command_line('rm V.dat')
 call execute_command_line('rm rho-p.dat')
 call execute_command_line('rm energy.dat')
 call execute_command_line('rm fractions.dat')

! FIXME: tvn2, tvo2, tvno are NaN
 open(1, file = 'T.dat', action='write', status='replace')
 do i = 1,npoints-1
   write(1,'(6ES15.7)') time_s(i), xout(i), temp(i), tvn2(i), tvo2(i), tvno(i)
 end do
 close(1)

 open(2, file = 'V.dat', action='write', status='replace')
 do i = 1,npoints-1
   write(2,'(6ES15.7)') time_s(i), xout(i), v(i), u1(i), u2(i), u3(i)
 end do
 close(2)

 open(3, file = 'fractions.dat', action='write', status='replace')
 do i = 1,npoints-1
   write(3,'(7ES15.7)') time_s(i), xout(i), nn2_n(i), no2_n(i), nno_n(i), nn_n(i), no_n(i)
 end do
 close(3)

 open(4, file = 'rho-p.dat', action='write', status='replace')
 do i = 1,npoints-1
   write(4,'(4ES15.7)') time_s(i), xout(i), rho(i), p(i)
 end do
 close(4)

 open(5, file = 'energy.dat', action='write', status='replace')
 do i = 1,npoints-1
   write(5,'(9ES15.7)') time_s(i), xout(i), e_tr(i), e_rot(i), e_v(i), e_v0, e_f(i), e_f0, enthalpy(i)
 end do
 close(5)

 open(6, file = 'energy_n2i.dat', action='write', status='replace')
 do i = 1,npoints
   write(6,'(47ES15.7)') (nn2i_n(i,j), j=1,l(1))
 end do
 close(6)

 open(7, file = 'energy_o2i.dat', action='write', status='replace')
 do i = 1,npoints
   write(7,'(36ES15.7)') (no2i_n(i,j), j=1,l(2))
 end do
 close(7)

 open(8, file = 'energy_noi.dat', action='write', status='replace')
 do i = 1,npoints
   write(8,'(39ES15.7)') (nnoi_n(i,j), j=1,l(3))
 end do
 close(8)

 deallocate(nn2, no2, nno)
 deallocate(n_at, n_mol)

 deallocate(temp, stat=ierror)
 if (ierror /= 0) stop "problems deallocating temp"
 deallocate(v, stat=ierror)
 if (ierror /= 0) stop "problems deallocating v"
 deallocate(nn2_i, stat=ierror)
 if (ierror /= 0) stop "problems deallocating nn2_i"
 deallocate(no2_i, stat=ierror)
 if (ierror /= 0) stop "problems deallocating no2_i"
 deallocate(nno_i, stat=ierror)
 if (ierror /= 0) stop "problems deallocating nno_i"
 deallocate(nn_, stat=ierror)
 if (ierror /= 0) stop "problems deallocating nn_"
 deallocate(no_, stat=ierror)
 if (ierror /= 0) stop "problems deallocating no_"
 deallocate(time_s, stat=ierror)
 if (ierror /= 0) stop "problems deallocating time_s"
 deallocate(time_mcs, stat=ierror)
 if (ierror /= 0) stop "problems deallocating time_mcs"
 deallocate(nall, stat=ierror)
 if (ierror /= 0) stop "problems deallocating nall"
 deallocate(nn2i_n, stat=ierror)
 if (ierror /= 0) stop "problems deallocating nn2i_n"
 deallocate(no2i_n, stat=ierror)
 if (ierror /= 0) stop "problems deallocating no2i_n"
 deallocate(nnoi_n, stat=ierror)
 if (ierror /= 0) stop "problems deallocating nnoi_n"
 deallocate(nn2_n, stat=ierror)
 if (ierror /= 0) stop "problems deallocating nn2_n"
 deallocate(no2_n, stat=ierror)
 if (ierror /= 0) stop "problems deallocating no2_n"
 deallocate(nno_n, stat=ierror)
 if (ierror /= 0) stop "problems deallocating nno_n"
 deallocate(nn_n, stat=ierror)
 if (ierror /= 0) stop "problems deallocating nn_n"
 deallocate(no_n, stat=ierror)
 if (ierror /= 0) stop "problems deallocating no_n"
 deallocate(rho, stat=ierror)
 if (ierror /= 0) stop "problems deallocating rho"
 deallocate(p, stat=ierror)
 if (ierror /= 0) stop "problems deallocating p"
 deallocate(tvn2, stat=ierror)
 if (ierror /= 0) stop "problems deallocating tvn2"
 deallocate(tvo2, stat=ierror)
 if (ierror /= 0) stop "problems deallocating tvo2"
 deallocate(tvno, stat=ierror)
 if (ierror /= 0) stop "problems deallocating tvno"
 deallocate(e_f, stat=ierror)
 if (ierror /= 0) stop "problems deallocating e_f"
 deallocate(ev_n2, stat=ierror)
 if (ierror /= 0) stop "problems deallocating ev_n2"
 deallocate(ev_o2, stat=ierror)
 if (ierror /= 0) stop "problems deallocating ev_o2"
 deallocate(ev_no, stat=ierror)
 if (ierror /= 0) stop "problems deallocating ev_no"
 deallocate(e_v, stat=ierror)
 if (ierror /= 0) stop "problems deallocating e_v"
 deallocate(e_tr, stat=ierror)
 if (ierror /= 0) stop "problems deallocating e_tr"
 deallocate(e_rot, stat=ierror)
 if (ierror /= 0) stop "problems deallocating e_rot"
 deallocate(enthalpy, stat=ierror)
 if (ierror /= 0) stop "problems deallocating h"
 deallocate(u1, stat=ierror)
 if (ierror /= 0) stop "problems deallocating u1"
 deallocate(u2, stat=ierror)
 if (ierror /= 0) stop "problems deallocating u2"
 deallocate(u3, stat=ierror)
 if (ierror /= 0) stop "problems deallocating u3"

 write(*,*) " ... Finish!"

 End program
