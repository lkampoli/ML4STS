# Shear viscosity
./regression.sh -a DT  -p shear
./regression.sh -a RF  -p shear
./regression.sh -a ET  -p shear
./regression.sh -a GB  -p shear
./regression.sh -a HGB -p shear
./regression.sh -a KN  -p shear
./regression.sh -a KR  -p shear
./regression.sh -a MLP -p shear
./regression.sh -a NN  -p shear
./regression.sh -a SVM -p shear

# Bulk viscosity
./regression.sh -a DT  -p bulk 
./regression.sh -a RF  -p bulk
./regression.sh -a ET  -p bulk
./regression.sh -a GB  -p bulk
./regression.sh -a HGB -p bulk
./regression.sh -a KN  -p bulk
./regression.sh -a KR  -p bulk
./regression.sh -a MLP -p bulk
./regression.sh -a NN  -p bulk
./regression.sh -a SVM -p bulk

# Thermal conductivity
./regression.sh -a DT  -p conductivity 
./regression.sh -a RF  -p conductivity
./regression.sh -a ET  -p conductivity
./regression.sh -a GB  -p conductivity
./regression.sh -a HGB -p conductivity
./regression.sh -a KN  -p conductivity
./regression.sh -a KR  -p conductivity
./regression.sh -a MLP -p conductivity
./regression.sh -a NN  -p conductivity
./regression.sh -a SVM -p conductivity

# Thermal diffusion
./regression.sh -a DT  -p thermal_diffusion
./regression.sh -a RF  -p thermal_diffusion
./regression.sh -a ET  -p thermal_diffusion
./regression.sh -a GB  -p thermal_diffusion
./regression.sh -a HGB -p thermal_diffusion
./regression.sh -a KN  -p thermal_diffusion
./regression.sh -a KR  -p thermal_diffusion
./regression.sh -a MLP -p thermal_diffusion
./regression.sh -a NN  -p thermal_diffusion
./regression.sh -a SVM -p thermal_diffusion

# Mass diffusion
./regression.sh -a DT  -p mass_diffusion
./regression.sh -a RF  -p mass_diffusion
./regression.sh -a ET  -p mass_diffusion
./regression.sh -a GB  -p mass_diffusion
./regression.sh -a HGB -p mass_diffusion
./regression.sh -a KN  -p mass_diffusion
./regression.sh -a KR  -p mass_diffusion
./regression.sh -a MLP -p mass_diffusion
./regression.sh -a NN  -p mass_diffusion
./regression.sh -a SVM -p mass_diffusion
