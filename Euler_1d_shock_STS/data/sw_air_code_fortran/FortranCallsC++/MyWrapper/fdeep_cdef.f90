! C functions declaration
interface
    function create_fdeep_c(a, b) bind(C, name="create_fdeep")
        use iso_c_binding
        implicit none
        type(c_ptr) :: create_fdeep_c
        integer(c_int), value :: a
        integer(c_int), value :: b
    end function

    subroutine delete_fdeep_c(fdeep) bind(C, name="delete_fdeep")
        use iso_c_binding
        implicit none
        type(c_ptr), value :: fdeep
    end subroutine

    function fdeep_load_c(fdeep, c) bind(C, name="fdeep_load")
        use iso_c_binding
        implicit none
        integer(c_int) :: fdeep_load_c
        ! The const qualification is translated into an intent(in)
        type(c_ptr), intent(in), value :: fdeep
        integer(c_int), value :: c
    end function

    function fdeep_predict_c(fdeep, c) bind(C, name="fdeep_predict")
        use iso_c_binding
        implicit none
        real(c_double) :: fdeep_predict_c
        type(c_ptr), intent(in), value :: fdeep
        real(c_double), value :: c
    end function

    ! void functions maps to subroutines
    subroutine fdeep_speaker_c(str) bind(C, name="fdeep_speaker")
        use iso_c_binding
        implicit none
        character(len=1, kind=C_CHAR), intent(in) :: str(*)
    end subroutine
end interface
