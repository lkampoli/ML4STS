! test_save_and_load.f90

! TO RUN
! ./test_save_and_load

! build a model using config file
! save the model to file
! load the model back
! assert that predictions match

program test_save_and_load
  use mod_kinds, only: ik, rk
  use mod_network, only: network_type

  implicit none

  type(network_type) :: net
  real(rk), allocatable :: result(:), input(:)

  ! load network from config file
  call net%load('../../ExampleModels/NN.txt')

  ! save network to config file
  !call net%save('../../ExampleModels/NN_saved.txt')

  input = [5164]

  ! run test input through network
  result = net%output(input)

  print *, "input = ", input
  print *, "result = ", result

end program test_save_and_load
