
program readme
real :: a(5), i
!double precision :: b(5,4)
open(1, file='test.unf', form='unformatted')
read(1) a
!read(1) b
close(1)
write(*,*) a
!do i = 1, 5
!    write(*,*) b(i,:)
!end do
end program
