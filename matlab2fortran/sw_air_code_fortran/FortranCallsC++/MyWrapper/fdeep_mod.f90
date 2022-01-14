module libfdeep
    use iso_c_binding

    private
    public :: fdeep, fdeep_speaker

    ! Yes, include is a keyword in Fortran !
    include "fdeep_cdef.f90"

    ! We'll use a Fortan type to represent a C++ class here, in an opaque maner
    type fdeep
        private
        type(c_ptr) :: ptr ! pointer to the Foo class
    contains
        ! We can bind some functions to this type, allowing for a cleaner syntax.
#ifdef __GNUC__
        procedure :: delete => delete_fdeep_polymorph ! Destructor for gfortran
#else
        final :: delete_fdeep ! Destructor
#endif
        ! Function member
        procedure :: load => fdeep_load
        procedure :: predict => fdeep_predict
    end type

    ! This function will act as the constructor for fdeep type
    interface fdeep
        procedure create_fdeep
    end interface

contains ! Implementation of the functions. We just wrap the C function here.
    function create_fdeep(a, b)
        implicit none
        type(fdeep) :: create_fdeep
        integer, intent(in) :: a, b
        create_fdeep%ptr = create_fdeep_c(a, b)
    end function

    subroutine delete_fdeep(this)
        implicit none
        type(fdeep) :: this
        call delete_fdeep_c(this%ptr)
    end subroutine

    ! Bounds procedure needs to take a polymorphic (class) argument
    subroutine delete_fdeep_polymorph(this)
        implicit none
        class(fdeep) :: this
        call delete_fdeep_c(this%ptr)
    end subroutine

    integer function fdeep_load(this, c)
        implicit none
        class(fdeep), intent(in) :: this
        integer, intent(in) :: c
        fdeep_load = fdeep_load_c(this%ptr, c)
    end function

    double precision function fdeep_predict(this, c)
        implicit none
        class(fdeep), intent(in) :: this
        double precision, intent(in) :: c
        fdeep_predict = fdeep_predict_c(this%ptr, c)
    end function

    subroutine fdeep_speaker(str)
        implicit none
        character(len=*), intent(in) :: str
        character(len=1, kind=C_CHAR) :: c_str(len_trim(str) + 1)
        integer :: N, i

        ! Converting Fortran string to C string
        N = len_trim(str)
        do i = 1, N
            c_str(i) = str(i:i)
        end do
        c_str(N + 1) = C_NULL_CHAR

        call fdeep_speaker_c(c_str)
    end subroutine
end module
