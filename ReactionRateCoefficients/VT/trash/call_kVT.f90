program rates
implicit none

integer :: u, n, i
real, dimension(46) :: kVT

integer T
character(10) Tstring, arg
character(7) :: base
character(16) :: exe1, exe2
character(50) :: command1, command2

T = 3756 !K
write( Tstring, '(i10)' )  T
write(*,*) trim(Tstring)

n = 46

base = "python3"
exe1 = "load_kVT_down.py"
exe2 = "load_kVT_up.py"
arg  = Tstring

command1 = base//" "//exe1//" "//arg
command2 = base//" "//exe2//" "//arg

call execute_command_line (command1)
call execute_command_line (command2)

!call execute_command_line ("python3 load_kVT_down.py 2750")
!call execute_command_line ("python3 load_kVT_up.py 2750")

open(newunit=u, file='kVT_down.out', action='read')
do i=1,n
  read(u,*)  kVT(i)
enddo
close(u)

write(*,*)  (kVT(i), i=1, n)
open(newunit=u, file='kVT_up.out', action='read')
read(u,*)  (kVT(i), i=1, n)
close(u)
write(*,*)  (kVT(i), i=1, n)

end program
