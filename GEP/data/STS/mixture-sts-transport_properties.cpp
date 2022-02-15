/*! 
    \file: mixture-sts-transport_properties.cpp 
    \brief Computation of transport properties for binary mixtures.
 */

#include <iostream>
#include <fstream>
#include <string>
#include <iomanip> 

#include<vector>
#include <iterator>

#ifdef WINDOWS
#include <direct.h>
#define GetCurrentDir _getcwd
#else
#include <unistd.h>
#define GetCurrentDir getcwd
#endif

#include "kappa.hpp"

using namespace kappa;

std::string GetCurrentWorkingDir( void ) {
  char buff[FILENAME_MAX];
  GetCurrentDir( buff, FILENAME_MAX );
  std::string current_working_dir(buff);
  return current_working_dir;
}

int main(int argc, char** argv) {

  std::cout << "Start test: computation of shear viscosity" << std::endl;

  std::string m_source = std::getenv("KAPPA_DATA_DIRECTORY");
  std::cout << "KAPPA_DATA_DIRECTORY is: " << m_source << '\n';
  std::string particle_source    = m_source + "particles.yaml";
  std::string interaction_source = m_source + "interaction.yaml";
  std::string output_dir = GetCurrentWorkingDir();
  std::cout << "Current directory is: " << output_dir << std::endl;

  std::vector<kappa::Molecule> molecules;
  std::vector<kappa::Atom> atoms;

  std::cout << "Loading particles data" << std::endl;

  kappa::Molecule mol("N2", true, false, particle_source);
  kappa::Atom at("N", particle_source);

  molecules.push_back(mol);
  atoms.push_back(at);

  kappa::Mixture mixture(molecules, atoms, interaction_source, particle_source);
  std::cout << "Mixture created" << std::endl;

  // Temperature [K]
  std::vector<double> T_vals;
  for (int i=0; i<100; i++) { 
    T_vals.push_back(100 + i * 100);
  }

  // Pressure [Pa]
  double pressure = 101325.;
  std::vector<double> P_vals = {1000.,10000.,20000.,30000.,40000.,50000.,60000.,70000.,80000.,90000.,100000.,101325.};

  // Molar fractions [] (over-written below)
  double x_N2 = 0.90;
  double x_N  = 0.10;

  // initialization (over-written below)
  std::vector<arma::vec> mol_ndens;
  mol_ndens.push_back(mixture.Boltzmann_distribution(T_vals[0], x_N2 * pressure / (K_CONST_K * T_vals[0]), mol)); // N2

  arma::vec atom_ndens(2);
  atom_ndens[0] = 0.;
  atom_ndens[1] = 0.;

  // dump result on different files to possibly reduce size
  std::ofstream outf_sv,outf_bv,outf_tc,outf_td,outf_md;

  outf_sv.open(output_dir + "/TRANSPORT_COEFFICIENTS/shear_viscosity.txt");
  outf_sv << std::setw(20) << "Pressure [Pa]";
  outf_sv << std::setw(20) << "Temperature [K]";
  outf_sv << std::setw(20) << "Molecular molar fractions []";
  outf_sv << std::setw(20) << "Atomic molar fractions []";
  outf_sv << std::setw(20) << "Shear viscosity [Pa-s]";
  outf_sv << std::endl;

  outf_bv.open(output_dir + "/TRANSPORT_COEFFICIENTS/bulk_viscosity.txt");
  outf_bv << std::setw(20) << "Pressure [Pa]";
  outf_bv << std::setw(20) << "Temperature [K]";
  outf_bv << std::setw(20) << "Molecular molar fractions []";
  outf_bv << std::setw(20) << "Atomic molar fractions []";
  outf_bv << std::setw(20) << "Bulk viscosity [Pa-s]";
  outf_bv << std::endl;

  outf_tc.open(output_dir + "/TRANSPORT_COEFFICIENTS/thermal_conductivity.txt");
  outf_tc << std::setw(20) << "Pressure [Pa]";
  outf_tc << std::setw(20) << "Temperature [K]";
  outf_tc << std::setw(20) << "Molecular molar fractions []";
  outf_tc << std::setw(20) << "Atomic molar fractions []";
  outf_tc << std::setw(20) << "Thermal conductivity [W/(m⋅K)]";
  outf_tc << std::endl;

  outf_td.open(output_dir + "/TRANSPORT_COEFFICIENTS/thermal_diffusion.txt");
  outf_td << std::setw(20) << "Pressure [Pa]";
  outf_td << std::setw(20) << "Temperature [K]";
  outf_td << std::setw(20) << "Molecular molar fractions []";
  outf_td << std::setw(20) << "Atomic molar fractions []";
  outf_td << std::setw(20) << "Thermal diffusion [m^2/s]";
  outf_td << std::endl;

  outf_md.open(output_dir + "/TRANSPORT_COEFFICIENTS/mass_diffusion.txt");
  outf_md << std::setw(20) << "Pressure [Pa]";
  outf_md << std::setw(20) << "Temperature [K]";
  outf_md << std::setw(20) << "Molecular molar fractions []";
  outf_md << std::setw(20) << "Atomic molar fractions []";
  outf_md << std::setw(20) << "Mass diffusion [m^2/s]";
  outf_md << std::endl;

//  std::vector<models_omega> omega_integral_models = {models_omega::model_omega_rs, models_omega::model_omega_vss,
//                                                     models_omega::model_omega_bornmayer, models_omega::model_omega_lennardjones,
//                                                     models_omega::model_omega_esa};

  x_N2 = 0.0;
  x_N  = 0.0;
  double tot_ndens = 0.0;
  arma::vec thd; // thermo-diffusion arma vector
  arma::mat diff; // diffusion coeffs
  int i, j, k, m, n;
  for (auto P : P_vals) { 
    std::cout << "Pressure = " << P << std::endl;
    for (auto T : T_vals) {
      std::cout << "Temperature = " << T << std::endl;
      tot_ndens =  P / (K_CONST_K * T);
      for (i=0; i<=10; i++) { // molecule N2
        x_N2 = 0. + i * 0.1;
        std::cout << "x_N2 = " << x_N2 << std::endl;
        x_N = 1 - x_N2;
        std::cout << "x_N = " << x_N << std::endl;

        mol_ndens[0] = mixture.Boltzmann_distribution(T, x_N2 * tot_ndens, mol);
        atom_ndens[0] = x_N * tot_ndens;

        mixture.compute_transport_coefficients(T, mol_ndens, atom_ndens, 0, models_omega::model_omega_rs, 0.);

        outf_sv << std::setw(20) << std::setprecision(6) << P;
        outf_bv << std::setw(20) << std::setprecision(6) << P;
        outf_tc << std::setw(20) << std::setprecision(6) << P;
        outf_td << std::setw(20) << std::setprecision(6) << P;
        outf_md << std::setw(20) << std::setprecision(6) << P;

        outf_sv << std::setw(20) << std::setprecision(6) << T;
        outf_bv << std::setw(20) << std::setprecision(6) << T;
        outf_tc << std::setw(20) << std::setprecision(6) << T;
        outf_td << std::setw(20) << std::setprecision(6) << T;
        outf_md << std::setw(20) << std::setprecision(6) << T;

        // molecular molar fractions
        for(std::vector<float>::size_type i = 0; i != mol_ndens[0].size(); i++) {
          outf_sv << std::setw(20) << std::setprecision(6) << mol_ndens[0][i]/tot_ndens;
          outf_bv << std::setw(20) << std::setprecision(6) << mol_ndens[0][i]/tot_ndens;
          outf_tc << std::setw(20) << std::setprecision(6) << mol_ndens[0][i]/tot_ndens;
          outf_td << std::setw(20) << std::setprecision(6) << mol_ndens[0][i]/tot_ndens;
          outf_md << std::setw(20) << std::setprecision(6) << mol_ndens[0][i]/tot_ndens;
        }

        // atomic molar fractions
        outf_sv << std::setw(20) << std::setprecision(6) << atom_ndens[0]/tot_ndens;
        outf_bv << std::setw(20) << std::setprecision(6) << atom_ndens[0]/tot_ndens;
        outf_tc << std::setw(20) << std::setprecision(6) << atom_ndens[0]/tot_ndens;
        outf_td << std::setw(20) << std::setprecision(6) << atom_ndens[0]/tot_ndens;
        outf_md << std::setw(20) << std::setprecision(6) << atom_ndens[0]/tot_ndens;

        outf_sv << std::setw(20) << std::setprecision(6) << mixture.get_shear_viscosity();
        outf_bv << std::setw(20) << std::setprecision(6) << mixture.get_bulk_viscosity();
        outf_tc << std::setw(20) << std::setprecision(6) << mixture.get_thermal_conductivity();
        //outf_td << std::setw(20) << std::setprecision(6) << mixture.get_thermodiffusion();
        //outf_md << std::setw(20) << std::setprecision(6) << mixture.get_diffusion();

	thd = mixture.get_thermodiffusion();
	for (k=0; k<thd.n_elem; k++) {
          outf_td << std::setw(20) << std::setprecision(6) << thd[k];
        }

	diff = mixture.get_diffusion();
	for (m=0; m<diff.n_rows; m++) {
          for (n=0; n<diff.n_cols; n++) {
            outf_md << std::setw(20) << std::setprecision(6) << diff.at(m,n);
          }
        }

        outf_sv << std::endl;
        outf_bv << std::endl;
        outf_tc << std::endl;
        outf_td << std::endl;
        outf_md << std::endl;
      }
    }
  }

  outf_sv.close();
  outf_bv.close();
  outf_tc.close();
  outf_td.close();
  outf_md.close();
  return 0;
}
