# Pressure
set term postscript eps enhanced color 20
set output "profile_P.eps"
set title "Pressure distribution behind a normal shock\nM_{/Symbol \245}=10.99, T_{/Symbol \245}=300.0 K, p_{/Symbol \245}=133.32 Torr"
set xlabel "Distance behind shock (cm)"
set ylabel "Pressure [Pa]"
set xrange [0.0001:200]
set yrange [*:*]
set logscale x
set mytics 2
set key top left
set grid xtics ytics mxtics mytics
plot "matlab/results.dat" u 1:10 t 'matlab'  w l lt 1 lw 2, \
     "rho-p.dat"          u 2:4  t 'fortran' w l lt 2 lw 2

# Density
set term postscript eps enhanced color 20
set output "profile_RHO.eps"
set title "Density distribution behind a normal shock\nM_{/Symbol \245}=10.99, T_{/Symbol \245}=300.0 K, p_{/Symbol \245}=133.32 Torr"
set xlabel "Distance behind shock (cm)"
set ylabel "Density [Kg/m^3]"
set xrange [0.0001:200]
set yrange [*:*]
set logscale x
set mytics 2
set key top left
set grid xtics ytics mxtics mytics
plot "matlab/results.dat" u 1:11 t 'matlab'  w l lt 1 lw 2, \
     "rho-p.dat"          u 2:3  t 'fortran' w l lt 2 lw 2

# Temperature
set term postscript eps enhanced color 20
set output "profile_T.eps"
set title "Temperature distribution behind a normal shock\nM_{/Symbol \245}=10.99, T_{/Symbol \245}=300.0 K, p_{/Symbol \245}=133.32 Torr"
set xlabel "Distance behind shock (cm)"
set ylabel "Temperature [K]"
set xrange [0.0001:200]
set yrange [*:*]
set logscale x
set mytics 2
set key bottom left
set grid xtics ytics mxtics mytics
plot "matlab/results.dat" u 1:2 t 'matlab'  w l lt 1 lw 2, \
     "T.dat"              u 2:3 t 'fortran' w l lt 2 lw 2

# Velocity
set term postscript eps enhanced color 20
set output "profile_V.eps"
set title "Velocity distribution behind a normal shock\nM_{/Symbol \245}=10.99, T_{/Symbol \245}=300.0 K, p_{/Symbol \245}=133.32 Torr"
set xlabel "Distance behind shock (cm)"
set ylabel "Velocity [m/s]"
set xrange [0.0001:200]
set yrange [*:*]
set logscale x
set mytics 2
set key bottom left
set grid xtics ytics mxtics mytics
plot "matlab/results.dat" u 1:3 t 'matlab'  w l lt 1 lw 2, \
     "V.dat"              u 2:3 t 'fortran' w l lt 2 lw 2
#
set ylabel "U_{1} [m/s]"
plot "matlab/results_U.dat" u 1:2 t 'matlab'  w l lt 1 lw 2, \
     "V.dat"                u 2:4 t 'fortran' w l lt 2 lw 2
#
set ylabel "U_{2} [m/s]"
plot "matlab/results_U.dat" u 1:3 t 'matlab'  w l lt 1 lw 2, \
     "V.dat"                u 2:5 t 'fortran' w l lt 2 lw 2
#
set ylabel "U_{3} [m/s]"
plot "matlab/results_U.dat" u 1:4 t 'matlab'  w l lt 1 lw 2, \
     "V.dat"                u 2:5 t 'fortran' w l lt 2 lw 2
#
unset label 1
unset label 2
unset label 3
unset arrow 1
unset arrow 2

# Molar Fractions
set output "profile_moles.eps"
set title "Species concentrations behind a normal shock in air\nM_{/Symbol \245}=10.99, T_{/Symbol \245}=300.0 K, p_{/Symbol \245}=133.32 Torr"
set xlabel "Distance behind shock (cm)"
set ylabel "Mole fractions"
set logscale x
set mytics 10
set xrange [0.01:200]
set yrange [*:*]
set grid nomxtics nomytics
set key bottom left
set title "N_{2} concentrations behind a normal shock in air\nM_{/Symbol \245}=10.99, T_{/Symbol \245}=300.0 K, p_{/Symbol \245}=133.32 Torr"
plot "matlab/results_fractions.dat" u 1:2 title 'matlab'  w l lt 2 lw 2, \
     "fractions.dat"                u 2:3 title 'fortran' w l lt 1 lw 2

set key top right
set title "O_{2} concentrations behind a normal shock in air\nM_{/Symbol \245}=10.99, T_{/Symbol \245}=300.0 K, p_{/Symbol \245}=133.32 Torr"
plot "matlab/results_fractions.dat" u 1:3 title 'matlab'  w l lt 2 lw 2, \
     "fractions.dat"                u 2:4 title 'fortran' w l lt 1 lw 2

set key top left
set title "NO concentrations behind a normal shock in air\nM_{/Symbol \245}=10.99, T_{/Symbol \245}=300.0 K, p_{/Symbol \245}=133.32 Torr"
plot "matlab/results_fractions.dat" u 1:4 title 'matlab'  w l lt 2 lw 2, \
     "fractions.dat"                u 2:5 title 'fortran' w l lt 1 lw 2

set key top left
set title "N concentrations behind a normal shock in air\nM_{/Symbol \245}=10.99, T_{/Symbol \245}=300.0 K, p_{/Symbol \245}=133.32 Torr"
plot "matlab/results_fractions.dat" u 1:5 title 'matlab'  w l lt 2 lw 2, \
     "fractions.dat"                u 2:6 title 'fortran' w l lt 1 lw 2

set key top left
set title "O concentrations behind a normal shock in air\nM_{/Symbol \245}=10.99, T_{/Symbol \245}=300.0 K, p_{/Symbol \245}=133.32 Torr"
plot "matlab/results_fractions.dat" u 1:6 title 'matlab'  w l lt 2 lw 2, \
     "fractions.dat"                u 2:7 title 'fortran' w l lt 1 lw 2

# Energy-Enthalpy
set term postscript eps enhanced color 20
set output "profile_E.eps"
set title "Energy distribution behind a normal shock\nM_{/Symbol \245}=10.99, T_{/Symbol \245}=300.0 K, p_{/Symbol \245}=133.32 Torr"
set xlabel "Distance behind shock (cm)"
set ylabel "Energy"
set xrange [0.0001:200]
set yrange [*:*]
set logscale x
set mytics 2
set key top left
set grid xtics ytics mxtics mytics
plot "matlab/results_E.dat" u 1:2 t 'matlab'  w l lt 1 lw 2, \
     "energy.dat"           u 2:5 t 'fortran' w l lt 2 lw 2

set title "Enthalpy distribution behind a normal shock\nM_{/Symbol \245}=10.99, T_{/Symbol \245}=300.0 K, p_{/Symbol \245}=133.32 Torr"
set ylabel "Enthalpy"
plot "matlab/results_E.dat" u 1:3 t 'matlab'  w l lt 1 lw 2, \
     "energy.dat"           u 2:9 t 'fortran' w l lt 2 lw 2
#
