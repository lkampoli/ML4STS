
program test_simpleNN

  use mod_kinds, only: ik, rk
  use mod_network, only: network_type

  implicit none

  type(network_type) :: net

  real(rk), allocatable :: result(:), input(:)
  character(len=100), dimension(:), allocatable :: args

  allocate(args(1))
  call get_command_argument(1,args(1))

  call net % load(args(1))

  input = [1156]

  result = net % output(input)

  print *, 'Prediction:', result

end program
