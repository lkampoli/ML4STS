! test.f90
program call_python
  use, intrinsic :: iso_c_binding
  implicit none
  interface
     subroutine hello_world() bind (c)
     end subroutine hello_world
  end interface

  call hello_world()

end program call_python
