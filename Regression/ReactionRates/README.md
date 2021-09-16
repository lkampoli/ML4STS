How to run regression?
======================
Assuming to be located in 'ReactionRates' directory (which is the most updated),
in order to perform the regression of reaction rates, execute the bash script,
~~~~~~
./run_regression.sh
~~~~~~
this will loop over all the defined processes ("DR" "VT" "VV" "VV2" "ZR")
and algorithms ("DT" "ET" "RF" "SVM" "GB" "HGB" "KN" "KR" "MLP") and for each
combination will run the python script `run_regression.py`.

Note that if such folders do not exist, they should created before running,
otherwise an error will be haulted.

In turn, the script `run_regression.py` will:

* pre-process: load input data and normalize them
* process: perform regression
* post-process: generate figures and output files

INPUTS
======
The original data was computed by Olga in Matlab for a one-dimensional
shock relaxation problem.

As initial conditions, the conditions from experiments were used:

1. Wurster, W. H., Treanor, C. E., and Williams, M. J., 
  "Non-Equilibrium Radiation from Shock-Heated Air," 
  U.S. Army Research Office, Final Rept., 
  Contract DAAL03-88K-0174, 
  Calspan—Univ. of Buffalo Research Center, Buffalo, NY, July 1991.

In free-stream:

* p0 = 2.25 Torr, T0 = 300 K,  
* a) for nN2/n = 95 %,   nO2/n = 5 %,    v0 = 3.87, 3.49, 3.15, 2.97 km/sec,  
* b) for nN2/n = 77.7 %, nO2/n = 22.3 %, v0 = 3.85, 3.52, 3.26, 2.99 km/sec,  
* c) for nN2/n = 60 %,   nO2/n = 40 %,   v0 = 3.85, 3.47, 3.24, 3.06 km/sec.   

The calculation was also done for a wider range of the initial velocity of the shock wave: 2.9 : 0.9 : 3.9.

2. Gorelov V.A., Gladyshev M.K., Kireev A.Y., Yegorov I.V., Plastinin Y.A., Karabadzhak G.F. 
   Experimental and numerical study of nonequilibrium ultraviolet NO and emission in shock layer 
   J. Thermophys. Heat Transfer. 1998. Vol. 12. P. 172-179.

In free-stream:  

* p0 = 0.1 Torr, T0 = 300 K,
* mixture composition - ordinary air (5 components)
* v0 = 5 : 0.2 : 9 km/sec.

Since the data is three-dimensional, the tables were represented in the following files:

* a) `DR_RATES.xlsx`: dissociation and recombination rate coefficients:
 * - rows    - vibrational state of molecule,
 * - columns - temperature,
 * - sheets  - partner in collision;
* b) `ZR_N2_RATES.xlsx` and `ZR_O2_RATES.xlsx`: Zeldovich reaction rate coefficients:
 * - rows    - vibrational state of molecule N2 or O2,
 * - columns - temperature,
 * - sheets  - vibrational state of NO;
* c) `VT_RATES.xlsx`: rate coefficients of VT exchanges:
 * - rows    - vibrational state of molecule,
 * - columns - temperature,
 * - sheets  - partner in collision;
* d) `VV_DOWN_N2_RATES.xlsx`, `VV_DOWN_O2_RATES.xlsx`,`VV_DOWN_NO_RATES.xlsx`: rate coefficients of VV exchanges:
 * - rows    - transition of main molecule down (1->0, 2->1, etc.),
 * - columns - temperature,
 * - sheets  - transition of partner up;
* e) `VV_UP_N2_RATES.xlsx`, `VV_UP_O2_RATES.xlsx`, `VV_UP_NO_RATES.xlsx`
 * - rows    - transition of main molecule up (0->1, 1->2, etc.),
 * - columns - temperature,
 * - sheets  - transition of partner down.

These original files are located in `data`. They have been further processed by
using the `convert.sh` bash script which executes the `libreconverter.py` python script.
In addition, other manipulations may have been done in order to adapt the file format,
such as removal of commas, tabulation, etc. (that's why other scripts are also present).

Recenty, a problem with `libreconverter.py` has been encountered so other tools may be used.

In all calculations were used:

* the Treanor-Marrone model for dissociation with U = D/6k
* Park's parameters for Arrhenius law
* Savelev model for Zeldovich reactions
* FHO for energy transitions

Dimension of rate coefficients is m^3/sec, for recombination is m^6/sec.

The input data is located with respect to the `process`, specifically, in 
the `ReactionRates` directory there are `DR` `VT` `VV` `VV2` `ZR` folders
and in each one there are the state-to-state rates in `data/processes`,
which are the output labels for the supervised regression and
the input features file in `data/Temperatures.csv`.
Thus, this is a single-input multi-output regression problem becase,
we infer the rates for all levels as function of temperature only.
The chosen temperature range is 250:10:10000 K.

OUTPUTS
=======
The output tree structure is automatically generate by the `run_regression.py`
within each process folder and it will consist of of directory for each algorithm.

Thus, there will be a solution directory:

* for each process, 
* for each algorithm and
* for each vibrational level.

At the innermost nested folder there are the following sub-folders:

* figures: where all plots are located
* models: where the regression model in saved (*.sav)
* otufiles: where all output files are dumped (train/test time/errors)
* scalers: where the scalers used for normalization are saved (*.pkl)

Possibly, a `GridSearchCV_results.csv` file reports the GridSearchCV results.

By running 
~~~~~~
./cleanall.sh
~~~~~~
all solutions will be deleted.

SOURCE CODE
===========
The source code consists of 3 files:

* estimators.py: where all the regression esstimators have been encoded
* utils.py: utilities for loading data, directory tree generation, normalization and plotting
* run_regression.py: master for regression
