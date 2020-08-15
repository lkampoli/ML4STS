program eta
implicit none

integer :: u!, i, ifail
!character execmd*255
real :: value

!call execute_command_line ("python3 load_eta.py 0.2 5000", exitstat=i)
!print *, "Exit status of external_prog.exe was ", i

!call execute_command_line ("python3 load_eta.py 0.2 5000", wait=.false.)
!print *, "Now reindexing files in the background"

!execmd = "python3 load_eta.py 0.2 5000"
!ifail = system(execmd)
!if (ifail /= 0) call exit(ifail)

! https://stackoverflow.com/questions/53447665/save-command-line-output-to-variable-in-fortran
!call execute_command_line ("python3 load_eta.py 0.2 5000 > result&")
call execute_command_line ("python3 load_eta.py 0.2 5000")
open(newunit=u, file='out', action='read')
read(u, *) value
write(*, *) 'Managed to read the value ', value

end program
