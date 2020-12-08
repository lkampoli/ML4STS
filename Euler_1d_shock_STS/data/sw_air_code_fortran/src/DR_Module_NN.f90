MODULE DR_Module_NN

contains

  subroutine compute_DR_NN(T)

  use constants
  !use mod_kinds, only: ik, rk
  !use mod_network, only: network_type
  !use mod_ensemble, only: ensemble_type

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

  write( temperature, '(i5)' ) int(T)

  base = " python3 "
  !path = "/home/lk/Public/ML4STS/Euler_1d_shock_STS/data/sw_air_code_fortran/DT/"
  path = "/home/lk/Public/ML4STS/Euler_1d_shock_STS/data/sw_air_code_fortran/KR/"

  call getcwd(origin)
  call chdir(path)

  compute_DR_rates = "run_regression_N2.py"
  command_compute_DR_rates = base//compute_DR_rates//" "//temperature
  !write(*,*) command_compute_DR_rates
  call execute_command_line (command_compute_DR_rates)

  compute_DR_rates = "run_regression_O2.py"
  command_compute_DR_rates = base//compute_DR_rates//" "//temperature
  !write(*,*) command_compute_DR_rates
  call execute_command_line (command_compute_DR_rates)

  compute_DR_rates = "run_regression_NO.py"
  command_compute_DR_rates = base//compute_DR_rates//" "//temperature
  !write(*,*) command_compute_DR_rates
  call execute_command_line (command_compute_DR_rates)

  open(newunit=u, file='result_N2.out', action='read')
  read (u, *) (k_dr_N2(i), i=1, size(k_dr_N2) - 1)
  close(u)
  do i = 1, 5
    kd_n2(i,:) = k_dr_N2((i-1)*l1+1:i*l1)
    kr_n2(i,:) = k_dr_N2((5+(i-1))*l1+1:(5+i)*l1)
  enddo
  open(newunit=u, file='result_O2.out', action='read')
  read (u, *) (k_dr_O2(i), i=1, size(k_dr_O2) - 1)
  close(u)
  do i = 1, 5
    kd_o2(i,:) = k_dr_O2((i-1)*l2+1:i*l2)
    kr_o2(i,:) = k_dr_O2((5+(i-1))*l2+1:(5+i)*l2)
  enddo
  open(newunit=u, file='result_NO.out', action='read')
  read (u, *) (k_dr_NO(i), i=1, size(k_dr_NO) - 1)
  close(u)
  do i = 1, 5
    kd_no(i,:) = k_dr_NO((i-1)*l3+1:i*l3)
    kr_no(i,:) = k_dr_NO((5+(i-1))*l3+1:(5+i)*l3)
  enddo

  call chdir(origin)

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

END MODULE
