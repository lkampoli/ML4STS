
 Module ExchangeVV
 implicit none

 contains

 ! Code fho.f
 ! AB(i1) + CD(i2) = AB(f1) + CD(f2)

 function kvv_fho(AB, CD, t, i1, i2, f1, f2) result(kvv)
 use constants
 implicit none

 integer, intent(in) :: AB, CD, f1, f2, i1, i2
 real(WP), intent(in) :: t
 real(WP) :: kvv
 real(WP) :: sigmab, lambda, exp_la
 real(WP) :: delE, delE1, delE2
 real(WP) :: Ef1, Ef2, Ei1, Ei2, Evib
 real(WP) :: s, sf, ns1, ns2 ! these are made real not to fail gamma function
 real(WP) :: mab, mcd, mul
 real(WP) :: ome_ab, ome_cd, Omega, omexe_ab, omexe_cd
 real(WP) :: R0l, P1001, th1, th2, Z, z1, z2, G
 integer :: p1, p2

 p1 = AB
 p2 = CD

 ome_ab   = om_e(p1)   * 1.4388e-2
 omexe_ab = om_x_e(p1) * 1.4388e-2
 ome_cd   = om_e(p2)   * 1.4388e-2
 omexe_cd = om_x_e(p2) * 1.4388e-2

 Ei1 = ome_ab*i1 - omexe_ab*i1*(i1+1)
 Ei2 = ome_cd*i2 - omexe_cd*i2*(i2+1)
 Ef1 = ome_ab*f1 - omexe_ab*f1*(f1+1)
 Ef2 = ome_cd*f2 - omexe_cd*f2*(f2+1)

 delE1  = Ei1-Ef1
 delE2  = Ei2-Ef2

 mab    = m(p1)
 mcd    = m(p2)

 mul    = mab*mcd/(mab+mcd)

 R0l    = rO(p1,p2)
 sigmab = pi*R0l*R0l
 Z      = sigmab*sqrt(Eight*k*t/(pi*mul))

! Avoiding gamma or factorial function save a lot of time!
! TODO: be carefull here ... (it may be not general at all!)
 s   = abs(i1-f1)
 sf  = 1 !gamma(s+1)
!ns1 = (gamma(max(i1,f1)+One)/gamma(min(i1,f1)+One))**(One/s)
!ns2 = (gamma(max(i2,f2)+One)/gamma(min(i2,f2)+One))**(One/s)
 ns1 = max(i1,f1)
 ns2 = max(i2,f2)

 if (i1 == f1) then
   th1 = ome_ab*(One-Two*(omexe_ab/ome_ab)*i1) ! K
   z1 = Zero
 else
   th1 = abs(delE1/(i1-f1)) ! K
   z1 = One
 end if

 if (i2 == f2) then
   th2 = ome_cd*(One-Two*(omexe_cd/ome_cd)*i2) ! K
   z2 = Zero
 else
   th2 = abs(delE2/(i2-f2)) ! K
   z2 = One
 end if

 if ((z1 /= 0) .or. (z2 /= 0)) then
   Evib = (th1*z1+th2*z2)/(z1+z2) ! K
 end if

 if ((z1 == 0) .and. (z2 == 0)) then
   Evib = f_1o2*(th1+th2) ! K
 end if

 Omega  = Evib*k/h_bar

 ! (1,0)->(0,1)
 P1001  = Svv*alphaVV*alphaVV*k*t/(Two*Omega*Omega*mul)

 ! Resonance defect correction (Keck, Carrier, Adamovich)
 delE   = delE1+delE2
 lambda = f_2o3*sqrt(FourPiPi*mul*Omega*Omega/(alphaVV*alphaVV*k*t)) * abs(delE)/(Evib*sqrt8*s)
 exp_la = exp(-f_2o3*lambda)
 G      = f_1o2*(Three-exp_la)*exp_la
 kvv    = Z * (ns1*ns2*P1001)**s/sf/(One+Two*ns1*ns2*P1001/(s+One))**(s+One) * G
 end function kvv_fho

 End module ExchangeVV
