program shock1d
implicit none

real :: x, t, rho, p, u, e, gamma, dx, dt
integer :: i, j

!dx = 0.001
!dt = 0.001
dx = 0.01
dt = 0.01
!dx = 0.1
!dt = 0.1

gamma = 1.4

!open(unit=69, file='datashock1d.dat')
!open(unit=69, file='datashock1dlite.dat')
open(unit=69, file='datashock1d001.dat')

x = 0.0
t = 0.0

!do i = 0, 1000 ! x
do i = 0, 100 ! x
!do i = 0, 10 ! x
  x = i * dx
  !do j = 0, 2000 ! t
  do j = 0, 200 ! t
  !do j = 0, 20 ! t
    t = j * dt
    if(x < 0.5 + 0.1 * t) then
      rho = 1.4
    else
      rho = 1.0
    endif
    u = 0.1
    p = 1.0
    e = p/(rho*(gamma-1)) + 0.5*u*u
   write(69,"(6f20.12)") x,t,rho,u,p,e
  enddo
enddo

close(69)

end program
