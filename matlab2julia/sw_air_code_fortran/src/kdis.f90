
 ! Calculation of dissociation rates, collision with mol/at
 ! Treanor-Marrone model
 ! t - temperature
 ! species - molecule species
 Module DissRec

 contains

 function kdis(t, species) result(kd)
 use constants
! use init_energy
 implicit none

 integer, intent(in) :: species
 real(WP), intent(in) :: t ! temperature
 real(WP), dimension(:), allocatable :: Z
 real(WP), dimension(:,:), allocatable :: kd
 real(WP), dimension(5) :: kd_eq
 real(WP) :: ZvT, ZvU, U, uk, tk, ko1oTp1oU, f_1otk, f_1ouk
 integer :: i, j, ierror

 allocate(Z(l(species)), stat=ierror)
 if (ierror /= 0) stop "Problems allocating Z"
 allocate(kd(5,l(species)), stat=ierror)
 if (ierror /= 0) stop "Problems allocating kd"

! equilibrium coefficients
 kd_eq = CA(species,:)*(t**nA(species,:))*exp(-D(species)/t) !m3/s

! parameter of TM model
 if (sw_u == "D/6k") then
   U = D(species)/Six
 else if (sw_u == "3T") then
   U = Three*t
 else if (sw_u == "Inf") then
   U = 999999999999999999. !Inf
 else
   write(*,*) 'Error! Check switch on parameter U.'
   return
 end if

! call PopulateEnergy()

!equil. vibr. partition function
 if (species == 1) then
   ZvT = sum(exp(-en2_i/(t*k)))
   ZvU = sum(exp(en2_i/(U*k)))
   Z = ZvT / ZvU * exp(en2_i/k*(1/t + 1/U))
 else if (species == 2) then
   ZvT = sum(exp(-eo2_i/(t*k)))
   ZvU = sum(exp(eo2_i/(U*k)))
   Z = ZvT / ZvU * exp(eo2_i/k*(1/t + 1/U))
 else if (species == 3) then
   ZvT = sum(exp(-eno_i/(t*k)))
   ZvU = sum(exp(eno_i/(U*k)))
   Z = ZvT / ZvU * exp(eno_i/k*(1/t + 1/U))
 end if

 do j = 1, l(species)
   kd(:,j) = kd_eq(:) * Z(j) !m^3/s
 end do
 end function kdis

 End module DissRec
