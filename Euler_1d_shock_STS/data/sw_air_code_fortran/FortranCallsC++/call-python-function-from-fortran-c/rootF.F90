program main
   implicit none
   real*8 A, B, t, x
   A = 0.4d0
   B = 0.3d0
   t = 0.1d0

   call root(A,B,t,x)

   write(*,*) x

end program main
