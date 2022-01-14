# Scikit-learn regression

In this directory, a simple regression is performed using Scikit-learn.
For the sake of example, Random Forest algorithm has been chosen as 
regressor, since it also allows to compute features importance.

Due to preliminary results, it may be intersting and convenient to 
investigate possible MODEL REDUCTIONS approaches, for example by:

 * removing un-important input features 
 * performing PCA or similares techniques
 * finding a more suitable latent representation with auto-enconders

Preliminary results show that for the `shear_viscosity.txt` dataset,
3 features (instead of 51) are sufficient to recover the majority
of information content, providing satisfactory prediction accuracy.

## Dataset 

For this simple test, the `shear_viscosity.txt` file has been used.
Similarly, the `bulk_viscosity.txt` and `thermal_conductivity.txt` 
files can be seamlessly used, by changing line 46 in `RF.py` script.

The file consists of 52 columns and 13200 rows (~14 Mb), where the
first 51 columns represent pressure, temperature and molar fractions
for a simple binary N2/N mixture. The last column is the shear viscosity.

To notice that the database files were produced by running the file 
`mixture-sts-transport_properties.cpp` in KAPPA. In particular, a 
Boltzmann distribution is assigned. The following variables range was
selected:

P = [1000.,10000.,20000.,30000.,40000.,50000.,60000.,70000.,80000.,90000.,100000.,101325.] \
T = 100:1000:100 \
X_N2 = 0.0:1.0:0.1

Note that the database file for mass diffusion `mass_diffusion.txt`
due to size reason has to be downloaded from the following link:
https://mega.nz/file/i7wgmapa#Iqh6eg2pwU07UtPK_tEFqjtvOPia5qQL7Qa_ArC9UKI

## How to run

To perform regression, execute:

~~~~~~~
python RF.py | tee output.log
~~~~~~~

## Results

At the moment, few files are outputted, namely:

 * the comparison of regression and ground truth (KAPPA), `shear.pdf`
 * the features importance, `importance.pdf`
 * the percentage variance as function of number of components, `pca.pdf`

It appears that two components would be enough to capture ~95% of the 
total variance.
