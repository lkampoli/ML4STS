/*
   \file transport_coefficients_air5.cpp
   \brief Computation of transport coefficients for air5 mixture in STS approach.
*/

#include <iostream>
#include <fstream>
#include <iomanip>

#ifdef WINDOWS
#include <direct.h>
#define GetCurrentDir _getcwd
#else
#include <unistd.h>
#define GetCurrentDir getcwd
#endif

#include "kappa.hpp"

//#define PY_SSIZE_T_CLEAN
//#include <Python.h>
#include <stdio.h>
#include <stdlib.h>

//    Py_Initialize();
//
//    PyObject * sys = PyImport_ImportModule("sys");
//    PyObject * path = PyObject_GetAttrString(sys, "path");
//    PyList_Append(path, PyString_FromString("."));
//
//    PyObject * ModuleString = PyString_FromString((char*) "py_function");
//    PyObject * Module = PyImport_Import(ModuleString);
//    PyObject * Dict = PyModule_GetDict(Module);
//    PyObject * Func = PyDict_GetItemString(Dict, "python_test");
//    PyObject * Result = PyObject_CallObject(Func, NULL);
//
//    Py_Finalize();
//
//    Py_Initialize();
//    PyRun_SimpleString("import sys; sys.path.append('.')");
//    PyRun_SimpleString("import mytest;");
//    PyRun_SimpleString("print mytest.myabs(2.0)");
//    Py_Finalize();

std::string GetCurrentWorkingDir( void ) {
  char buff[FILENAME_MAX];
  GetCurrentDir( buff, FILENAME_MAX );
  std::string current_working_dir(buff);
  return current_working_dir;
}

using namespace kappa;

int main(int argc, char** argv) {

  std::cout << "Start computation of transport coefficients" << std::endl;
  std::string m_source = std::getenv("KAPPA_DATA_DIRECTORY");
  std::cout << "KAPPA_DATA_DIRECTORY is: " << m_source << '\n';
  std::string particle_source    = m_source + "particles.yaml";
  std::string interaction_source = m_source + "interaction.yaml";
  std::string output_dir = GetCurrentWorkingDir();
  std::cout << "Current directory is: " << output_dir << std::endl;

  std::cout << "Loading particles data" << std::endl;

  //      anharmonic osc.-\      /-rigid rot.
  kappa::Molecule N2("N2", true, false, particle_source);
  kappa::Molecule O2("O2", true, false, particle_source);
  kappa::Molecule NO("NO", true, false, particle_source);
  kappa::Atom N("N",                    particle_source);
  kappa::Atom O("O",                    particle_source);

  std::vector<kappa::Molecule> mol;
  std::vector<kappa::Atom> at;

  mol.push_back(N2);
  mol.push_back(O2);
  mol.push_back(NO);
  at.push_back(N);
  at.push_back(O);

  std::cout << "Finished loading particles data" << std::endl;

  //std::cout << "Molecule's name " << mol.name << std::endl;
  //std::cout << "Atom's name " << at.name << std::endl;
  //std::cout << "Molecule vibrational levels " << mol.num_vibr_levels[0] << std::endl;

  // air5 mixture
  kappa::Mixture mixture(mol, at, interaction_source, particle_source);

  // some check print
  std::cout << "particles: " << mixture.get_n_particles() << std::endl;
  std::cout << "names: " << mixture.get_names() << std::endl;

  int n_samples = 25;

  FILE *initial;
  //double Xi[5][100];
  double Xi[5][n_samples];
  double x1, x2, x3, x4, x5;
  //initial = fopen( "molar_fractions100.out", "r" ) ;
  initial = fopen( "molar_fractions25.out", "r" ) ;
  if (initial == NULL)
  { printf("Pb ouverture fichier" ); }

  //for ( int n = 0 ; n < 101 ; n++ )
  for ( int n = 0 ; n < n_samples; n++ )
  {
    fscanf(initial, "%le %le %le %le %le\n", &x1, &x2, &x3, &x4, &x5);

    //std::cout << n << std::endl;

    Xi[0][n] = x1;
    Xi[1][n] = x2;
    Xi[2][n] = x3;
    Xi[3][n] = x4;
    Xi[4][n] = x5;
    //std::cout << Xi[0][n] << " " << Xi[1][n] << " " << Xi[2][n] << " " << Xi[3][n] << " " << Xi[4][n] << std::endl;
  }
  fclose(initial);

  // spectroscopic constant for diatomic molecules in ground electronic state
  //int vibr_l = 0; // for N2 d0
  //double Re = 1.097E-10; // N2
  //double be = 2.25E-10;
  //double omega_e = 235860; // m^-1 (N2)
  //double mu = 0.028; // kg/mol (N2)
  //double l_alpha = sqrt( 16.863/(omega_e * mu) );
  //double beta = 2.6986E+10; // N2
  //double d0 = Re + be + (9./2.)*beta*l_alpha*l_alpha*exp( 2*sqrt(beta*l_alpha)*(vibr_l - 1) );

  // set a range for temperature
  std::vector<double> T_vals;
  std::vector<double> X_N2_vals, X_O2_vals, X_NO_vals, X_N_vals, X_O_vals;
  std::vector<double> P_vals;
  double tot_ndens;

  //std::vector<double> T_vals = { 500. };

  // vibrational levels
  int i;

  //for (i=1; i<40; i++) { //100-20000 K
  //  T_vals.push_back(0. + i * 500.);
  //}
  for (i=1; i<10; i++) { //100-5000 K
    T_vals.push_back(0. + i * 500.);
  }

  for (i=1; i<=10; i++) { //10-60000 Pa
    P_vals.push_back(0. + i * 1000.);
  }
  //for (i=1; i<=6; i++) { //10-60000 Pa
  //  P_vals.push_back(0. + i * 5000.);
  //}

  //for (i=0; i<=10; i++) { //0.-0.1-1.
  //  X_N2_vals.push_back(i/10.);
  //  X_O2_vals.push_back(i/10.);
  //  X_NO_vals.push_back(i/10.);
  //  X_N_vals.push_back(i/10.);
  //  X_O_vals.push_back(i/10.);
  //}

  //for (i=0; i<=30; i++) { //0.3:0.6:0.01
  //  X_N2_vals.push_back(0.3 + i/100.);
  //}

  //for (i=0; i<=100; i++) { //3.0E-05:5.5E-03
  //  X_O2_vals.push_back(0.00003 + 5*i/100000.);
  //}

  //for (i=0; i<=500; i++) { //7.E-04-7.E-03
  //  X_NO_vals.push_back(0.0007 + 2*i/100000.);
  //}

  //for (i=0; i<=40; i++) { //0.:0.4:0.01
  //  X_N_vals.push_back(0. + i/100.);
  //}

  //for (i=0; i<=20; i++) { //0.2:0.4:0.01
  //  X_O_vals.push_back(0.2 + i/100.);
  //}

  // arma vector for atom number density
  arma::vec atom_ndens(2);

  // vector of arma vector for molecular number density
  std::vector<arma::vec> mol_ndens;

  // assume an initial boltzmann ditribution                    101325.0
  mol_ndens.push_back(mixture.Boltzmann_distribution(T_vals[0], P_vals[0] / (kappa::K_CONST_K * T_vals[0]), N2));
  mol_ndens.push_back(mixture.Boltzmann_distribution(T_vals[0], P_vals[0] / (kappa::K_CONST_K * T_vals[0]), O2));
  mol_ndens.push_back(mixture.Boltzmann_distribution(T_vals[0], P_vals[0] / (kappa::K_CONST_K * T_vals[0]), NO));

  // output files
  std::ofstream outf;
  std::ofstream outfMD;

  //outf.open(output_dir + "/TCs_air5_INPUTS" + ".txt");
  //outf.open(output_dir + "/TCs_air5_MD_full_BIG" + ".txt");
  //outf.open(output_dir + "/TCs_air5_MD_full_SMALL" + ".txt");
  //outf.open(output_dir + "/TCs_air5_MD_full_SMALL_diffXi" + ".txt");
  outf.open(output_dir + "/TCs_air5_MD_full_SMALL_T500_P1000" + ".txt");
  //outfMD.open(output_dir + "/TCs_air5_MD" + ".txt");
  //outfMD.open(output_dir + "/TCs_air5_MD_diag" + ".txt");

  //outf << std::setw(20) << "Temperature [K]";
  //outf << std::setw(20) << "Thermal conduction.";
  //outf << std::setw(20) << "Shear viscosity.";
  //outf << std::setw(20) << "Bulk viscosity.";
  //outf << std::endl;

  //outf << std::setw(20) << "n";
  outf << std::setw(12) << "T";
  outf << std::setw(12) << "P";
  outf << std::setw(12) << "x_N2";
  outf << std::setw(12) << "x_O2";
  outf << std::setw(12) << "x_NO";
  outf << std::setw(12) << "x_N";
  outf << std::setw(12) << "x_O";
  //outf << std::setw(15) << "Eta";
  //outf << std::setw(15) << "Zeta";
  //outf << std::setw(15) << "Lam";
  //outf << std::setw(15) << "TD";
  //for (i=0; i<122; i++) {
  //  outf << std::setw(12) << "["<<i<<"]";
  //}
  //// https://stackoverflow.com/questions/1229900/reformat-in-vim-for-a-nice-column-layout
  for (i=0; i<124; i++) {
    for (int j=0; j<124; j++) {
      outf << std::setw(12) << std::scientific << "Dij["<<i<<"]["<<j<<"]"; // << std::endl;
    }
  }
  outf << std::endl;

  // transport coefficients
  double th_c;    // thermal conductivity
  double sh_v;    // shear viscosity
  double bk_v;    // bulk viscosity
  double bk_o_sh; // ratio bulk/shear viscosity
  arma::vec th_d; // thermal diffusion
  arma::mat diff; // diffusion coeffs.

  //double x_N2 = 1.0,  x_O2 = 0.0, x_NO = 0.0, x_N = 0.0, x_O = 0.0;
  //double x_N2 = 0.90, x_O2 = 0.0, x_NO = 0.0, x_N = 0.1, x_O = 0.0;

  double x_N2, x_O2, x_NO, x_N, x_O;

   for (auto T : T_vals) {
    for (auto P : P_vals) {
     // total number density
     tot_ndens = P / (kappa::K_CONST_K * T);
     //for (int lv=0; lv<100; lv++) {
     //
     system("./run_dirichlet.sh");
     initial = fopen( "molar_fractions25.out", "r" ) ;
     if (initial == NULL)
     { printf("Pb ouverture fichier" ); }
     for ( int n=0; n<n_samples; n++ )
     {
       fscanf(initial, "%le %le %le %le %le\n", &x1, &x2, &x3, &x4, &x5);
       Xi[0][n] = x1; Xi[1][n] = x2; Xi[2][n] = x3; Xi[3][n] = x4; Xi[4][n] = x5;
     }
     fclose(initial);
     //
     for (int lv=0; lv<n_samples; lv++) {

       x_N2 = Xi[0][lv];
       x_O2 = Xi[1][lv];
       x_NO = Xi[2][lv];
       x_N  = Xi[3][lv];
       x_O  = Xi[4][lv];

       //std::cout << Xi[0][lv] << " " << Xi[1][lv] << " " << Xi[2][lv] << " " << Xi[3][lv] << " " << Xi[4][lv] << std::endl;
       std::cout << lv << " " << x_N2 << " " << x_O2 << " " << x_NO << " " << x_N << " " << x_O << std::endl;

     //for (auto x_N2 : X_N2_vals) {
     // for (auto x_O2 : X_O2_vals) {
     //  for (auto x_NO : X_NO_vals) {
     //   for (auto x_N : X_N_vals)   {
     //    for (auto x_O : X_O_vals)   {

          mol_ndens[0] = mixture.Boltzmann_distribution(T, (1. - x_N2) * tot_ndens, N2);
          mol_ndens[1] = mixture.Boltzmann_distribution(T, (1. - x_O2) * tot_ndens, O2);
          mol_ndens[2] = mixture.Boltzmann_distribution(T, (1. - x_NO) * tot_ndens, NO);
          atom_ndens[0] = x_N * tot_ndens;
          atom_ndens[1] = x_O * tot_ndens;

          // computation of transport coefficients
          //mixture.compute_transport_coefficients(T, mol_ndens, atom_ndens, 0, models_omega::model_omega_rs, 0.);
          mixture.compute_transport_coefficients(T, mol_ndens, atom_ndens, 0, models_omega::model_omega_esa, 0.);

          // retrieve shear viscosity coefficients
          //sh_v = mixture.get_shear_viscosity();

          // retrieve bulk viscosity coefficients
          //bk_v = mixture.get_bulk_viscosity();

          // retrieve thermal conductivity coefficients
          //th_c = mixture.get_thermal_conductivity();

          //th_d = mixture.get_thermodiffusion();

          diff = mixture.get_diffusion();
          //std::cout << "n_rows: " << diff.n_rows << std::endl;
          //std::cout << "n_cols: " << diff.n_cols << std::endl;
          //std::cout << "n_elem: " << diff.n_elem << std::endl;

          //outf << std::setw(10) << tot_ndens;
          outf << std::setw(15) << std::scientific << T;
          outf << std::setw(15) << std::scientific << P;
          outf << std::setw(15) << std::scientific << x_N2;
          outf << std::setw(15) << std::scientific << x_O2;
          outf << std::setw(15) << std::scientific << x_NO;
          outf << std::setw(15) << std::scientific << x_N;
          outf << std::setw(15) << std::scientific << x_O;
          //outf << std::setw(15) << std::scientific << th_c;
          //outf << std::setw(15) << std::scientific << sh_v;
          //outf << std::setw(15) << std::scientific << bk_v;
          //for (i=0; i<th_d.n_elem; i++) {
          //  outf << std::setw(15) << std::scientific << th_d[i];
          //}
          //outf << std::endl;
          for (i=0; i<diff.n_rows; i++) {
            for (int j=0; j<diff.n_cols; j++) {
              //if (j>=i) {
                //std::cout << "diff["<<i<<","<<j<<"]" << std::endl;
                //outfMD << std::setw(15) << std::scientific << diff.at(i,j);
                outf << std::setw(15) << std::scientific << diff.at(i,j);
             //}
            }
          }
          outf << std::endl;
          //outfMD << std::endl;

//         }
//        }
//       }
//      }
     }
    }
   }

   outf.close();
   //outfMD.close();

   return 0;
}
