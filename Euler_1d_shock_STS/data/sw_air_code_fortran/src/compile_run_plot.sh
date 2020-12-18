rm shprot.x *.o *.mod *.eps

gfortran -c kind_module.f90 \
            constants.f90 \
            init_energy.f90 \
            kdis.f90 \
            DR_Module.f90 \
            fann.F03 \
            DR_Module_NN.f90 \
            k_ex_savelev_st.f90 \
            EX_Module.f90 \
            brent_module.f90 \
            kvt_fho.f90 \
            VT_Module.f90 \
            kvv_fho.f90 \
            VV_Module.f90 \
            rpart_fho.f90 \
            odepack.f \
            odepack_sub1.f \
            odepack_sub2.f \
            OpenMP_dvode_f90_m.f90 \
            main.f90

gfortran *.o -lopenblas -lfann -o shprot.x

time ./shprot.x
#./plot
