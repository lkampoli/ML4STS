program skl2f
  use forpy_mod
!  use iso_fortran_env
!  use iso_c_binding
  implicit none

  integer                       :: ierror, args_len, int_value
  integer                       :: arr(1), i
  type(tuple)                   :: args
  type(list)                    :: my_list, paths
  type(dict)                    :: my_dict, kwargs
  type(ndarray)                 :: my_nd_arr, py_input, py_output
  type(object)                  :: model, retval, item
  type(module_py)               :: skl, mymodule
! real(kind=real32)             :: temperature, float_value
  real                          :: temperature, float_value
  character(len=:), allocatable :: str_value, return_string

  !arr(1) = 42

  ierror = forpy_initialize()

  if (ierror /= 0) then
    write (*,*) "Initialisation of forpy failed! Cannot test. Error code = ", ierror
    stop
  endif
 
  !ierror = list_create(my_list)
  !ierror = ndarray_create(my_nd_arr, arr)
 
  ! Instead of setting the environment variable PYTHONPATH,
  ! we can add the current directory "." to sys.path
  ierror = get_sys_path(paths)
!  if (ierror == 0) then
    ierror = paths%append(".")
!    call paths%destroy
!  endif
  
!  if (ierror /= 0) then
!    write(*,*) "Error setting PYTHONPATH. Cannot test...", ierror
!    call err_print
!    STOP
!  endif

  !ierror = import_py(skl_mod, "skl")
  !if (ierror /= 0) then
  !  write(*,*) "Could not import test module 'test_ndarray'. Cannot test..."
  !  STOP
  !endif

  ierror = import_py(skl, "skl")

  temperature = 1500.

  ierror = tuple_create(args, 4)
  ierror = args%setitem(0, 'model.sav')
  ierror = args%setitem(1, 'scalex.pkl')
  ierror = args%setitem(2, 'scaley.pkl')
! ierror = args%setitem(3, temperature)
  ierror = args%setitem(3, 1500.)

  ierror = args%len(args_len)
  write(*,*) args_len

  do i = 0, args_len-1  ! Python indices start at 0
    ierror = args%getitem(item, i)
    
    ! Use is_int, is_str, is_float, is_none ...
    ! to check if an object is of a certain Python type 
    if (is_int(item)) then
      ! Use cast to transform 'item' into Fortran type 
      ierror = cast(int_value, item)
      write(*,*) int_value
    else if (is_float(item)) then
      ! Use cast to transform 'item' into Fortran type 
      ierror = cast(float_value, item)
      write(*,*) float_value
    else if(is_str(item)) then
      ierror = cast(str_value, item)
      write(*,*) str_value 
    endif
    call item%destroy
  enddo

  ierror = dict_create(kwargs)
  ierror = kwargs%setitem("greeting", "hi")

  !ierror = call_py(retval, skl_mod, "run_regression", args)
  ierror = call_py(retval, skl, "print_args", args, kwargs)
  ierror = call_py(retval, skl, "run_regression", args)

  ierror = cast(return_string, retval)
  write(*,*) return_string

  !ierror = cast(my_nd_arr, retval)

  !ierror = my_nd_arr%get_data(arr)

  !ierror = my_list%append(19)
  !ierror = my_list%append("Hello world!")
  !ierror = my_list%append(3.14d0)
  !ierror = print_py(my_list)

  !call my_list%destroy
  !call my_nd_arr%destroy

  call args%destroy
  call kwargs%destroy
  call skl%destroy
  call retval%destroy
  call paths%destroy

  call forpy_finalize

end program
