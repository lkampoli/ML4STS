program test_TF_Version
  use fTF
  implicit none

  character(c_char) :: version
  version = TF_Version()

!  print*, version
  write(*,*) "OK!"

end program
