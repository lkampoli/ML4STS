# Dataset description

## DB4T.dat
Transport properties for 4T model:
write(666,"(46f15.7)") press, T, TVCO2, TVO2, TVCO, x, & ! inputs
                       visc, bulk_visc, lvibr_co2, lvibr_O2, lvibr_CO, ltot, & ! output
                       THDIF(1), THDIF(2), THDIF(3), THDIF(4), THDIF(5), &
                       ((DIFF(i,j), i=1,5),j=1,5)

# shear_viscosity.txt (and the others)
Transport properties with state-to-state model for N2/N mixture
computed with KAPPA, mixture-sts-transport_properties.cpp

Pressure [Pa] Temperature [K] Molecular molar fractions [] Atomic molar fractions [] Shear viscosity [Pa-s]
