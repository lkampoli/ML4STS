program intro_to_forpy
  use forpy_mod
  use iso_fortran_env
  use iso_c_binding
  implicit none

  integer :: ierror
  integer :: arr(1)

  type(list)      :: my_list, paths
  type(dict)      :: my_dict
  type(ndarray)   :: my_nd_arr, py_input, py_output
  type(object)    :: model, retval
  type(module_py), save :: skl_mod

  real(kind=real64) :: f_input(1), f_output(1)

  arr(1) = 42
  !f_input(1) = 1500

  ierror = forpy_initialize()
  if (ierror /= 0) then
    write (*,*) "Initialisation of forpy failed!!! Can not test. Errorcode = ", ierror
    stop
  endif
 
  !ierror = list_create(my_list)

  ierror = ndarray_create(my_nd_arr, arr)
 
  ! Instead of setting the environment variable PYTHONPATH,
  ! we can add the current directory "." to sys.path
  ierror = get_sys_path(paths)
  if (ierror == 0) then
    ierror = paths%append(".")
    call paths%destroy
  endif
  
  if (ierror /= 0) then
    write(*,*) "Error setting PYTHONPATH. Cannot test...", ierror
    call err_print
    STOP
  endif

  !ierror = import_py(skl_mod, "skl")
  !if (ierror /= 0) then
  !  write(*,*) "Could not import test module 'test_ndarray'. Cannot test..."
  !  STOP
  !endif

  ierror = call_py(retval, skl_mod, "run_regression")
  ierror = cast(my_nd_arr, retval)
  call retval%destroy

  !ierror = my_nd_arr%get_data(arr)

  !ierror = my_list%append(19)
  !ierror = my_list%append("Hello world!")
  !ierror = my_list%append(3.14d0)
  !ierror = print_py(my_list)

  !call my_list%destroy
  call my_nd_arr%destroy

  call forpy_finalize

end program
