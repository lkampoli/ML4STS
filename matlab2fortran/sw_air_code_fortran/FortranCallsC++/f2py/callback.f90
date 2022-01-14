subroutine foo(fun,r)
external fun
integer ( kind = 4 ) i
!real ( kind = 8 ) r
real ( kind = 8 ), intent(out) :: r
r=0.0D+00
do i= 1,5
    r=r+fun(i)
enddo
end

