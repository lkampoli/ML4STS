program test
  use libfdeep
  implicit none

  type(fdeep) :: f

  ! Create an object of type fdeep
  f = fdeep(3, 4)

  ! Call bound procedures (member functions)
  !write(*,*) f%load_model("example_model.json")
  !write(*,*) f%predict(10d0)
  write(*,*) f%load(60), " should be ", 63
  write(*,*) f%predict(10d0), " should be ", 14.0d0

  call fdeep_speaker("From Fortran!")

  ! The destructor should be called automatically here, but this is not yet
  ! implemented in gfortran. So let's do it manually.
#ifdef __GNUC__
   call f%delete
#endif
End program
