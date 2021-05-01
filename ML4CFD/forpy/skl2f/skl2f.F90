program intro_to_forpy
  use forpy_mod
  use iso_fortran_env
  use iso_c_binding
  implicit none

  integer :: ierror
  integer :: arr(1)

  type(list)      :: my_list
  type(ndarray)   :: my_nd_arr
  type(object)    :: model
  type(module_py) :: skl

  real(kind=real64) :: input, output

  arr(1) = 42

  ierror = forpy_initialize()
  ierror = list_create(my_list)

  ierror = ndarray_create(my_nd_arr, arr)

  ierror = my_list%append(19)
  ierror = my_list%append("Hello world!")
  ierror = my_list%append(3.14d0)
  ierror = print_py(my_list)

  call my_list%destroy
  call my_nd_arr%destroy
  call forpy_finalize

end program
