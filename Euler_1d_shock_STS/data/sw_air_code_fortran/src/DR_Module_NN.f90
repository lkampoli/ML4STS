MODULE DR_Module_NN
  use fann ! it contains all the wrappers to C/C++ library

contains

  subroutine compute_DR_NN(T)

  use constants
  !use mod_kinds, only: ik, rk
  !use mod_network, only: network_type
  !use mod_ensemble, only: ensemble_type
  use, intrinsic :: ISO_C_BINDING
  use, intrinsic :: ISO_FORTRAN_ENV

  implicit none

  real(WP), intent(in) :: T
  character(10) :: temperature

  character(70)  :: origin
  character(70)  :: path
  character(9)   :: base
  character(20)  :: compute_DR_rates
  character(132) :: command_compute_DR_rates

  real :: start, finish

  integer :: u,i
  real(WP), dimension(l1*10) :: k_dr_N2
  real(WP), dimension(l2*10) :: k_dr_O2
  real(WP), dimension(l3*10) :: k_dr_NO

  type(C_PTR) :: ann
  type(C_PTR) :: train
  type(C_PTR) :: test

  integer, parameter :: sp = C_FLOAT
  integer, parameter :: ft = FANN_TYPE

  integer, parameter :: num_layer = 3
  integer, parameter :: nin = 3
  integer, parameter :: nout= 1
  integer, dimension(num_layer) :: layers

  integer, parameter :: ndata = 100
  real, dimension(nin,ndata) :: inTrainData
  real, dimension(nout,ndata) :: outTrainData

  integer :: max_epochs, epochs_between_reports
  real(sp) ::  desired_error

  real(ft), dimension(nin) :: x

  ! input
  layers(1) = nin
  ! hidden
  layers(2) = 10
  ! outout
  layers(3) = nout

  write(*,*) "in DR_NN ... "

  ! the net, with SIGMOID
  ann = fann_create_standard_array(num_layer,layers)
  call fann_set_activation_function_hidden(ann,enum_activation_function('FANN_SIGMOID'))
  call fann_set_activation_function_output(ann,enum_activation_function('FANN_SIGMOID'))
  call fann_print_connections(ann)

  ! my training data. Let's learn make a neural net which is a random generator :)
  call random_number(inTrainData)
  call random_number(outTrainData)
  !write(*,*) inTrainData

!  train = fann_create_train_from_callback(ndata,nin,nout,C_FUNLOC(mytrain_callback))
!
!  ! training
!  call fann_set_training_algorithm(ann,enum_training_algorithm('FANN_TRAIN_RPROP'))
!
  max_epochs = 10000
  epochs_between_reports = 1000
  desired_error = 0.001
!  call fann_train_on_data(ann,train,max_epochs,epochs_between_reports,desired_error)
!  call fann_print_connections(ann)
!
  ! testing
  x = T !(/0.1_ft,0.5_ft,1._ft/)

  ! running
!  print *, 'ann(x)= ',f_fann_run(ann,x)

  ! saving
!  print *,'saving...', fann_save(ann,f_c_string('arg.dat'))
!  call fann_destroy(ann)

  ! loading
!  print *,'loading...'
!  ann = fann_create_from_file(f_c_string('arg.dat'))
!  print *, 'loaded ann(x)= ',f_fann_run(ann,x)

!  write( temperature, '(i5)' ) int(T)
!
!  base = " python3 "
!  !path = "/home/lk/Public/ML4STS/Euler_1d_shock_STS/data/sw_air_code_fortran/DT/"
!  path = "/home/lk/Public/ML4STS/Euler_1d_shock_STS/data/sw_air_code_fortran/KR/"
!
!  call getcwd(origin)
!  call chdir(path)
!
!  compute_DR_rates = "run_regression_N2.py"
!  command_compute_DR_rates = base//compute_DR_rates//" "//temperature
!  !write(*,*) command_compute_DR_rates
!  call execute_command_line (command_compute_DR_rates)
!
!  compute_DR_rates = "run_regression_O2.py"
!  command_compute_DR_rates = base//compute_DR_rates//" "//temperature
!  !write(*,*) command_compute_DR_rates
!  call execute_command_line (command_compute_DR_rates)
!
!  compute_DR_rates = "run_regression_NO.py"
!  command_compute_DR_rates = base//compute_DR_rates//" "//temperature
!  !write(*,*) command_compute_DR_rates
!  call execute_command_line (command_compute_DR_rates)
!
!  open(newunit=u, file='result_N2.out', action='read')
!  read (u, *) (k_dr_N2(i), i=1, size(k_dr_N2) - 1)
!  close(u)
!  do i = 1, 5
!    kd_n2(i,:) = k_dr_N2((i-1)*l1+1:i*l1)
!    kr_n2(i,:) = k_dr_N2((5+(i-1))*l1+1:(5+i)*l1)
!  enddo
!  open(newunit=u, file='result_O2.out', action='read')
!  read (u, *) (k_dr_O2(i), i=1, size(k_dr_O2) - 1)
!  close(u)
!  do i = 1, 5
!    kd_o2(i,:) = k_dr_O2((i-1)*l2+1:i*l2)
!    kr_o2(i,:) = k_dr_O2((5+(i-1))*l2+1:(5+i)*l2)
!  enddo
!  open(newunit=u, file='result_NO.out', action='read')
!  read (u, *) (k_dr_NO(i), i=1, size(k_dr_NO) - 1)
!  close(u)
!  do i = 1, 5
!    kd_no(i,:) = k_dr_NO((i-1)*l3+1:i*l3)
!    kr_no(i,:) = k_dr_NO((5+(i-1))*l3+1:(5+i)*l3)
!  enddo
!
!  call chdir(origin)

  !type(network_type) :: NN
  !type(ensemble_type) :: ensemble
  !real(rk), allocatable :: result(:)
  !real(rk), allocatable :: input(:)
  !character(len=100), dimension(:), allocatable :: args

  ! load network from config file
  !call NN%load('../NN/NN.txt')

  !input = [T]

  ! run input through network
  !result = NN%output(input)

  !allocate(args(1))
  !call get_command_argument(1,args(1))

  ! build ensemble from members in specified directory
  !ensemble = ensemble_type(args(1), 0.0)

  ! run test input through network
  !result1 = ensemble%average(input)
  !print *, result

  end subroutine

!  subroutine mytrain_callback(num, num_input, num_output, input, output) bind(C)
!    implicit none
!
!    integer(C_INT), value :: num, num_input, num_output
!!#ifdef FIXEDFANN
!!    integer(FANN_TYPE), dimension(0:num_input-1) :: input
!!    integer(FANN_TYPE), dimension(0:num_output-1) :: output
!!    input(0:num_input-1) = int(inTrainData(1:num_input,num+1),FANN_TYPE)
!!    output(0:num_output-1) = int(outTrainData(1:num_output,num+1),FANN_TYPE)
!!#else
!    real(FANN_TYPE), dimension(0:num_input-1) :: input
!    real(FANN_TYPE), dimension(0:num_output-1) :: output
!    input(0:num_input-1) = real(inTrainData(1:num_input,num+1),FANN_TYPE)
!    output(0:num_output-1) = real(outTrainData(1:num_output,num+1),FANN_TYPE)
!!#endif

END MODULE
