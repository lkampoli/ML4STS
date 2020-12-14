module intrealstring
use iso_c_binding
implicit none

contains

integer (C_INT) function change_integer(n) bind(c)
    implicit none

    integer (C_INT), intent(in) :: n
    integer (C_INT), parameter :: power = 2

    change_integer = n ** power
end function change_integer

end module intrealstring
