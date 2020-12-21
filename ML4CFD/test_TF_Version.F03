program test_TF_Version
  use ftf
  implicit none

  character(c_char) :: version
! version = TF_Version()

  type(TF_Buffer) :: tfBuffer
  type(TF_Graph) :: tfGraph

  tfGraph = TF_NewGraph()

  write(*,*) "OK!"

end program
