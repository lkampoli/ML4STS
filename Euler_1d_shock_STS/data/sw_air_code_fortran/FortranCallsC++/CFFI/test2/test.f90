program call_python
  use, intrinsic :: iso_c_binding
  implicit none

  interface
    subroutine add_five(x_c, n) bind (c)
        use iso_c_binding
        integer(c_int) :: n
        real(c_double) :: x_c(n)
    end subroutine add_five
    subroutine minus_one(x_c, n) bind (c)
        use iso_c_binding
        integer(c_int) :: n
        real(c_double) :: x_c(n)
    end subroutine minus_one
  end interface

  real(c_double) :: x(10)
  x = 69.

  print *, x
  call add_five(x, size(x))
  print *, x
  call minus_one(x, size(x))
  print *, x

end program call_python
