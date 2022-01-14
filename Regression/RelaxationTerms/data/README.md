The Matlab code in bi_mix_sts is used to generate the dataset containing the following variables:

x_s, time_s, Temp, rho, p, v, E, ni_n, na_n, RD_mol, RD_at

where:

* x_s: distance from the shock
* time_s: relaxation time in seconds
* Temp: temperature [K]
* rho: density [kg/m^3]
* p: pressure [Pa]
* v: velocity [m/s]
* E: energy [J]
* ni_n: molar fractions for each vibrational level (number densities over total number density)
* na_n: molar fractions for atom species (number density over total number density)
* RD_mol: dissociation/recombination source term for each vibrational species [..]
* RD_at: dissociation/recombination source term for atomic species [..]
*
which are saved in the file solution_DR.dat, considering only the dissociation/recombination processes.
