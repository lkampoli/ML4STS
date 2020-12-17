program example
  use fann
  implicit none

  type(C_PTR)                   :: ann
  type(C_PTR)                   :: train
  integer, parameter            :: sp = C_FLOAT
  integer, parameter            :: ft = FANN_TYPE
  integer, parameter            :: num_layer = 3
  integer, parameter            :: nin = 1
  integer, parameter            :: nout= 126
  integer, dimension(num_layer) :: layers
  integer, parameter            :: ndata = 1768
  real, dimension(nin,ndata)    :: inTrainData
  real, dimension(nout,ndata)   :: outTrainData
  real, dimension(ndata,nout+1) :: dataXY
  integer                       :: max_epochs, epochs_between_reports
  integer                       :: i,j
  real(sp)                      :: desired_error
  real(ft), dimension(nin)      :: x

! input
  layers(1) = nin
! hidden
  layers(2) = 100
! outout
  layers(3) = nout

! the net, with SIGMOID
  ann = fann_create_standard_array(num_layer,layers)
  call fann_set_activation_function_hidden(ann,enum_activation_function('FANN_SIGMOID'))
  call fann_set_activation_function_output(ann,enum_activation_function('FANN_SIGMOID'))
  call fann_print_connections(ann)

! opening the file for modifying
  open (1, file = '../../data/solution_XY_12.dat', status = 'old')
  do j=1,ndata
    read(1,*) dataXY(j,:)
  end do
  close(1)
  open(2, file = 'dataXYmod.dat', status='new')
  do j=1,ndata
    write(2,"(1E15.7)") dataXY(j,1)
    write(2,"(126E15.7)") (dataXY(j,i), i=2,nout+1)
  enddo
  close(2)

! training data
  call fann_read_train_from_file("dataXYmod.dat")
!  open (1, file = '../../data/solution_XY_12.dat', status = 'old')
!  do j=1,ndata
!    read(1,*) dataXY(j,:)
!  end do
!  close(1)
!  inTrainData(nin,:) = dataXY(:,1)
!  do i = 1,ndata !1768
!    do j = 1,nout !126
!      outTrainData(j,i) = dataXY(i,j+1)
!    enddo
!  enddo

!  train = fann_create_train_from_callback(ndata,nin,nout,C_FUNLOC(mytrain_callback))

!training
  call fann_set_training_algorithm(ann,enum_training_algorithm('FANN_TRAIN_RPROP'))

  max_epochs = 1000
  epochs_between_reports = 100
  desired_error = 0.001
  call fann_train_on_data(ann,train,max_epochs,epochs_between_reports,desired_error)
  call fann_print_connections(ann)

!testing

  x = (/15000._ft/)

!running
  print *, 'ann(x)= ',f_fann_run(ann,x)

!saving
  print *,'saving...', fann_save(ann,f_c_string('XY.net'))
  call fann_destroy(ann)

!loading
  print *,'loading...'
  ann = fann_create_from_file(f_c_string('XY.net'))
  print *, 'loaded ann(x)= ',f_fann_run(ann,x)

contains

 subroutine mytrain_callback(num, num_input, num_output, input, output) bind(C)
    implicit none

    integer(C_INT), value :: num, num_input, num_output
#ifdef FIXEDFANN
    integer(FANN_TYPE), dimension(0:num_input-1) :: input
    integer(FANN_TYPE), dimension(0:num_output-1) :: output

    input(0:num_input-1) = int(inTrainData(1:num_input,num+1),FANN_TYPE)
    output(0:num_output-1) = int(outTrainData(1:num_output,num+1),FANN_TYPE)

#else
    real(FANN_TYPE), dimension(0:num_input-1) :: input
    real(FANN_TYPE), dimension(0:num_output-1) :: output

    input(0:num_input-1) = real(inTrainData(1:num_input,num+1),FANN_TYPE)
    output(0:num_output-1) = real(outTrainData(1:num_output,num+1),FANN_TYPE)

#endif

  end subroutine  mytrain_callback

end program example
