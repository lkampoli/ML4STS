
 Module ExchangeVT
 implicit none

 contains

 ! VT exchange process
 ! Reference Adamovich et al. 1998
 ! Code fho.f
 ! AB(i) + Csp = AB(f) + Csp

 function kvt_fho(AB, Csp, t, i, f) result(kvt)
 use constants
 use brent_module
 implicit none

 integer, intent(in) :: AB, Csp
 integer, intent(in) :: i, f
 real(WP) :: kvt, x
 real(WP), intent(in) :: t
 real(WP) :: alpha, const, const2
 real(WP) :: Cvt, delE, deltaa
 real(WP) :: depth
 real(WP) :: Enf, Eni
 real(WP) :: ma, mab, mb, mc, mpar, muu
 real(WP) :: om, ome, omexe
 real(WP) :: R00val
 real(WP) :: rate
 real(WP) :: Svt, vm0, x0, Z
 real(WP) :: theta, theta1, sigmaa, phi
 integer :: s, sf
 integer :: ns, nu
 integer :: p1, p2
 integer :: count

 real(wp) :: xzero, fzero
 integer :: iflag

 real(wp), parameter :: ax   = 0.1
 real(wp), parameter :: bx   = 10
 real(wp), parameter :: tole = 1.0e-6_wp

 type,extends(brent_class) :: myfunc_type
 integer :: i = 0 ! function counter
 end type myfunc_type
 type(myfunc_type) :: myfunc

 real ( kind = 8 ), parameter :: fatol = 1.0D-06
 real ( kind = 8 ) fx
 real ( kind = 8 ) fxa
 real ( kind = 8 ) fxb
 integer ( kind = 4 ), parameter :: max_step = 25
 real ( kind = 8 ) xa
 real ( kind = 8 ), parameter :: xatol = 1.0D-06
 real ( kind = 8 ) xb
 real ( kind = 8 ), parameter :: xrtol = 1.0D-06

 if (i == f) then
   write(*,*) "Error, same states!"
   stop
 end if

 p1 = AB
 p2 = Csp

 ome   = om_e(p1)   * 1.4388e-2
 omexe = om_x_e(p1) * 1.4388e-2

 Eni = ome*i - omexe*i*(i+1)
 Enf = ome*f - omexe*f*(f+1)

 ! AB
 mab = m(p1)
 mc  = m(p2)
 ma  = mab * ram_masses(p1,1)
 mb  = mab * ram_masses(p1,2)

 muu = (ma+mb)*mc/(ma+mb+mc)

 alpha = 4e10
 depth = TwoHundred*k
 mpar  = ma*mc/mb/(ma+mb+mc)
 Svt   = Two*mpar/((One+mpar)*(One+mpar))

 R00val = rO(p1,p2) ! m

 sigmaa = pi*R00val*R00val

 ! m^3/sec
 Z = sigmaa*sqrt(Eight*k*t/(pi*muu))

 if (p2 <= 3) then
   nu = 1
 else
   nu = 0
 end if

 delE   = Eni-Enf
 theta  = abs(delE/(i-f)) ! K
 om     = theta*k/h_bar ! sec^-1
 theta1 = FourPiPi*om*om*muu/(alpha*alpha)/k ! K

! Avoiding gamma or factorial function save a lot of time
! TODO: be carefull here ...
 s  = abs(i-f)  ! if we allow only mono-quantum jumps, this is always 1
 sf = 1.        !gamma(s+1.) !gamma(x) = (x-1)! 1! = 1
 ns = max(i,f)  !(gamma(max(i,f)+1.)/gamma(min(i,f)+1.))**(1./s)
! write(*,*) "s, sf, ns = ", min(i,f), max(i,f), ns

 vm0    = (TwoPi*om*s*k*t/alpha/muu)**(f_1o3)
 const  = One/s * (nu+Two*ns**(One/s)/(s+One))*Svt*theta1/theta
 const2 = (ns**(One/s)/(s+1)*Svt*theta1/theta)**Two

 x0     = Two

 call myfunc%set_function(my_func) ! set the function

 !call zeroin:
 myfunc%i = 0
 !call myfunc%find_zero(ax+0.0001_wp,bx/two+0.0001,tole,xzero,fzero,iflag)
 call myfunc%find_zero(ax+0.0001_wp,bx/two+0.1,tole,xzero,fzero,iflag)

 Cvt = xzero

 deltaa = (One-Cvt*Cvt*Cvt)/(Cvt*Cvt*Cvt) * TwoPi*om/alpha/vm0/Cvt
 phi    = f_2oPi * atan(sqrt(Two*depth/muu)/vm0)

 rate   = ns*sqrt(TwoPi/(Three+deltaa))*s**(f_1o3)/(sf*sf)
 rate   = rate * Cvt * (Svt*theta1/theta)**s * (theta1/t)**(f_1o6)
 rate   = rate*exp(-s**(f_2o3)*(theta1/t)**(f_1o3)*(f_1o2*Cvt*Cvt+One/Cvt)*(One-phi)**(f_2o3)-s*(One-Cvt*Cvt*Cvt))
 rate   = rate * exp(theta*s/Two/t)
 kvt    = rate * Z ! m**3/sec

 contains

 function my_func(me,x) result(f)
 implicit none

 class(brent_class),intent(inout) :: me
 real(wp),intent(in) :: x
 real(wp) :: f

 f = x-(One-const*exp(-TwoPi*om/(alpha*vm0*x))-const2*exp(-FourPi*om/(alpha*vm0*x)))**f_1o3

 select type (me)
 class is (myfunc_type)
   me%i = me%i + 1 ! number of function calls
 end select

 end function my_func

 subroutine p_fx ( x, fx )

  implicit none

  real ( kind = 8 ) fx
  real ( kind = 8 ) x

  fx = x-(One-const*exp(-TwoPi*om/(alpha*vm0*x))-const2*exp(-FourPi*om/(alpha*vm0*x)))**f_1o3

  return
 end

 subroutine brent ( fatol, step_max, prob, xatol, xrtol, xa, xb, fxa, fxb )

  implicit none

  real ( kind = 8 ) d
  real ( kind = 8 ) e
  real ( kind = 8 ) fatol !1.0D-06
  real ( kind = 8 ) fxa
  real ( kind = 8 ) fxb
  real ( kind = 8 ) fxc
  integer ( kind = 4 ) step_max ! 25
  integer ( kind = 4 ) prob
  integer ( kind = 4 ) step_num
  real ( kind = 8 ) p
  real ( kind = 8 ) q
  real ( kind = 8 ) r
  real ( kind = 8 ) s
  real ( kind = 8 ) xa
  real ( kind = 8 ) xb
  real ( kind = 8 ) xc
  real ( kind = 8 ) xm
  real ( kind = 8 ) xatol !1.0D-06
  real ( kind = 8 ) xrtol !1.0D-06
  real ( kind = 8 ) xtol
!
!  Initialization.
!
  write ( *, '(a)' ) ' '
  write ( *, '(a)' ) 'BRENT'
  write ( *, '(a)' ) ' '
  write ( *, '(a)' ) &
    '  Step      XA            XB             F(XA)         F(XB)'
  write ( *, '(a)' ) ' '

  step_num = 0

  call p_fx ( xa, fxa )
  call p_fx ( xb, fxb )
!
!  Check that f(ax) and f(bx) have different signs.
!
  if ( ( fxa < 0.0D+00 .and. fxb < 0.0D+00 ) .or. &
       ( 0.0D+00 < fxa .and. 0.0D+00 < fxb ) ) then

    write ( *, '(a)' ) ' '
    write ( *, '(a)' ) 'BRENT - Fatal error!'
    write ( *, '(a)' ) '  F(XA) and F(XB) have same sign.'
    return

  end if

  xc = xa
  fxc = fxa
  d = xb - xa
  e = d

  do

    write ( *, '(2x,i4,2x,2g16.8,2g14.6)' ) step_num, xb, xc, fxb, fxc

    step_num = step_num + 1

    if ( step_max < step_num ) then
      write ( *, '(a)' ) ' '
      write ( *, '(a)' ) '  Maximum number of steps taken.'
      exit
    end if

    if ( abs ( fxc ) < abs ( fxb ) ) then
      xa = xb
      xb = xc
      xc = xa
      fxa = fxb
      fxb = fxc
      fxc = fxa
    end if

    xtol = 2.0D+00 * xrtol * abs ( xb ) + 0.5D+00 * xatol
!
!  XM is the halfwidth of the current change-of-sign interval.
!
    xm = 0.5D+00 * ( xc - xb )

    if ( abs ( xm ) <= xtol ) then
      write ( *, '(a)' ) ' '
      write ( *, '(a)' ) '  Interval small enough for convergence.'
      exit
    end if

    if ( abs ( fxb ) <= fatol ) then
      write ( *, '(a)' ) ' '
      write ( *, '(a)' ) '  Function small enough for convergence.'
      exit
    end if
!
!  See if a bisection is forced.
!
    if ( abs ( e ) < xtol .or. abs ( fxa ) <= abs ( fxb ) ) then

      d = xm
      e = d

    else

      s = fxb / fxa
!
!  Linear interpolation.
!
      if ( xa == xc ) then

        p = 2.0D+00 * xm * s
        q = 1.0D+00 - s
!
!  Inverse quadratic interpolation.
!
      else

        q = fxa / fxc
        r = fxb / fxc
        p = s * ( 2.0D+00 * xm * q * ( q - r ) - ( xb - xa ) * ( r - 1.0D+00 ) )
        q = ( q - 1.0D+00 ) * ( r - 1.0D+00 ) * ( s - 1.0D+00 )

      end if

      if ( 0.0D+00 < p ) then
        q = - q
      else
        p = - p
      end if

      s = e
      e = d

      if ( 3.0D+00 * xm * q - abs ( xtol * q ) <= 2.0D+00 * p .or. &
           abs ( 0.5D+00 * s * q ) <= p ) then
        d = xm
        e = d
      else
        d = p / q
      end if

    end if
!
!  Save in XA, FXA the previous values of XB, FXB.
!
    xa = xb
    fxa = fxb
!
!  Compute the new value of XB, and evaluate the function there.
!
    if ( xtol < abs ( d ) ) then
      xb = xb + d
    else if ( 0.0D+00 < xm ) then
      xb = xb + xtol
    else if ( xm <= 0.0D+00 ) then
      xb = xb - xtol
    end if

    call p_fx ( xb, fxb )
!
!  If the new FXB has the same sign as FXC, then replace XC by XA.
!
    if ( ( 0.0D+00 < fxb .and. 0.0D+00 < fxc ) .or. &
         ( fxb < 0.0D+00 .and. fxc < 0.0D+00 ) ) then
      xc = xa
      fxc = fxa
      d = xb - xa
      e = d
    end if

  end do

  return
 end

 end function kvt_fho

 End module ExchangeVT
