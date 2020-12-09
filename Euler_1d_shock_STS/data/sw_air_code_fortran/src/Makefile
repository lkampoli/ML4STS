#!/usr/bin/make
#----------------------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------------------------
# shell
SHELL = /bin/bash
#----------------------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------------------------
# no verbose
#$(VERBOSE).SILENT:
#----------------------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------------------------
# User options
COMPILER = intel
DEBUG    = no
F03STD   = no
OPTIMIZE = no
OPENMP   = no
NULi     = no
NULj     = no
NULk     = no
EXTLIBS	 = yes
#----------------------------------------------------------------------------------------------------------------------------------

.PHONY : DEFAULTRULE
DEFAULTRULE: $(DEXE) shprot

.PHONY : help
help:
	@echo
	@echo -e '\033[1;31m Make options of shprot code\033[0m'
	@echo
	@echo -e '\033[1;31m Compiler choice: COMPILER=$(COMPILER)\033[0m\033[1m => default\033[0m'
	@echo -e '\033[1;31m  COMPILER=gnu  \033[0m\033[1m => GNU gfortran          \033[0m'
	@echo -e '\033[1;31m  COMPILER=intel\033[0m\033[1m => Intel Fortran         \033[0m'
	@echo
	@echo -e '\033[1;31m Compiling options\033[0m'
	@echo -e '\033[1;31m  DEBUG=yes(no)    \033[0m\033[1m => on(off) debug                  (default $(DEBUG))\033[0m'
	@echo -e '\033[1;31m  F03STD=yes(no)   \033[0m\033[1m => on(off) check standard fortran (default $(F03STD))\033[0m'
	@echo -e '\033[1;31m  OPTIMIZE=yes(no) \033[0m\033[1m => on(off) optimization           (default $(OPTIMIZE))\033[0m'
	@echo -e '\033[1;31m  OPENMP=yes(no)   \033[0m\033[1m => on(off) OpenMP directives      (default $(OPENMP))\033[0m'
	@echo
	@echo -e '\033[1;31m Provided Rules: default=shprot\033[0m\033[1m => compile the code\033[0m'
	@echo -e '\033[1;31m  cleanobj     =>\033[0m\033[1m cleaning compiled object\033[0m'
	@echo -e '\033[1;31m  cleanmod     =>\033[0m\033[1m cleaning .mod files\033[0m'
	@echo -e '\033[1;31m  cleanmsg     =>\033[0m\033[1m cleaning make-log massage files\033[0m'
	@echo -e '\033[1;31m  clean        =>\033[0m\033[1m running cleanobj, cleanmod and cleanmsg\033[0m'
	@echo -e '\033[1;31m  cleanall     =>\033[0m\033[1m running clean and cleanexe\033[0m'
	@echo -e '\033[1;31m  tar          =>\033[0m\033[1m creating a tar archive of the project\033[0m'
	@echo -e '\033[1;31m  doc          =>\033[0m\033[1m building the documentation\033[0m'
#----------------------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------------------------
# directory & file
DSRC  = ./src/
DOBJ  = ./obj/
DMOD  = ./mod/
DEXE  = ./
VPATH = $(DSRC) $(DOBJ) $(DMOD)
MKDIRS = $(DOBJ) $(DMOD) $(DEXE)
#-----------------------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------------------
# Compiler specific rules

# GNU
WRN_GNU = -fmax-errors=0 -Wall -Wno-array-temporaries -Warray-bounds -Wcharacter-truncation -Wline-truncation -Wconversion-extra \
	  -Wimplicit-interface -Wimplicit-procedure -Wunderflow -Wextra -Wuninitialized
CHK_GNU = -fcheck=all
DEB_GNU = -fmodule-private -ffree-line-length-132 -fimplicit-none -ffpe-trap=invalid,overflow -fbacktrace -fdump-core \
	  -finit-real=nan #-fno-range-check, precision, denormal, underflow
STD_GNU = -std=f2003 -fall-intrinsics
OPT_GNU = -O3
OMP_GNU = -fopenmp

# Intel
WRN_INT = -warn all
CHK_INT = -check all
DEB_INT = -debug all -extend-source 132 -fpe-all=0 -fp-stack-check -fstack-protector-all -ftrapuv -no-ftz -traceback -gen-interfaces
#DEB_INT = -O0 -C -g -p -warn all -check all -debug all -extend-source 132 -fpe-all=0 -fp-stack-check -fstack-protector-all \
	  -ftrapuv -no-ftz -traceback -std03 -heap-arrays -check noarg_temp_created -check nopointers -check noshap \
	  -qopt-report-phase=ipo,loop,vec -gen-interfaces -warn interfaces
STD_INT = -std03
#OPT_INT = -O3 -xHost -fast -qopt-matmul -ipo -lmkl_blas95_lp64 -lmkl_lapack95_lp64 -qopenmp-simd -qopenmp-stubs -vec \
	  -qsimd-serialize-fp-reduction -finline-functions -qopt-report=5 -qopt-report-filter=vec -inline all -ipo-jobs4 \
	  -qopt_report-phase=vec -unroll
OPT_INT = -fast
OMP_INT = -qopenmp
LIB_INT = -mkl #-L/home/lk/AI/ML/FKB/build/lib/libneural.a

# setting rules according user options
ifeq "$(COMPILER)" "gnu"
  FC = gfortran
  #OPTSC = -cpp -c -J$(DMOD) -static -fprotect-parens -fno-realloc-lhs
  OPTSL =
  WRN = $(WRN_GNU)
  CHK = $(CHK_GNU)
  DEB = $(DEB_GNU)
  STD = $(STD_GNU)
  OPT = $(OPT_GNU)
endif
ifeq "$(COMPILER)" "intel"
  FC = ifort -qopenmp
  #OPTSC = -cpp -c -module $(DMOD) -static -assume protect_parens -assume norealloc_lhs -fp-model source -xHost -fast -parallel #-qopenmp
  #OPTSC = -cpp -c -module $(DMOD) -assume protect_parens -assume norealloc_lhs -fp-model source -xHost -fast -parallel #-qopenmp
	#OPTSC = -cpp -c -module $(DMOD) -static -assume protect_parens -assume norealloc_lhs -fp-model source
	#OPTSC = -cpp -c -module $(DMOD) -static -assume protect_parens -assume norealloc_lhs -fp-model fast=2
	OPTSC = -cpp -c -module $(DMOD)
  OPTSL =
  WRN = $(WRN_INT)
  CHK = $(CHK_INT)
  DEB = $(DEB_INT)
  STD = $(STD_INT)
  OPT = $(OPT_INT)
  LIB = $(LIB_INT)
endif
ifeq "$(DEBUG)" "yes"
  PREPROC := $(PREPROC) -DDEBUG
  OPTSC := $(OPTSC) -O0 -C -g $(WRN) $(CHK) $(DEB)
  OPTSL := $(OPTSL) -O0 -C -g $(WRN) $(CHK) $(DEB)
endif
ifeq "$(F03STD)" "yes"
  OPTSC := $(OPTSC) $(STD)
  OPTSL := $(OPTSL) $(STD)
endif
ifeq "$(OPTIMIZE)" "yes"
  OPTSC := $(OPTSC) $(OPT)
  OPTSL := $(OPTSL) $(OPT)
endif
ifeq "$(OPENMP)" "yes"
  PREPROC := $(PREPROC) -DOPENMP
  OPTSC := $(OPTSC) $(OMP)
  OPTSL := $(OPTSL) $(OMP)
endif
OPTSC := $(OPTSC) $(PREPROC)
OPTSL := $(OPTSL) $(PREPROC) $(LIB)

WHICHFC = $(shell which $(FC))

PRINTCHK = "\\033[1;31m Compiler used \\033[0m\\033[1m $(COMPILER) => $(WHICHFC)\\033[0m \n\
            \\033[1;31m Source dir    \\033[0m\\033[1m $(DSRC)\\033[0m \n\
            \\033[1;31m Debug         \\033[0m\\033[1m $(DEBUG)\\033[0m \n\
            \\033[1;31m F-standard    \\033[0m\\033[1m $(F03STD)\\033[0m \n\
            \\033[1;31m Optimize      \\033[0m\\033[1m $(OPTIMIZE)\\033[0m"
#-----------------------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------------------
# auxiliary rules
.PHONY : PRINTINFO
.NOTPARALLEL : PRINTINFO
PRINTINFO:
	@echo | tee make.log
	@echo -e $(PRINTCHK) | tee -a make.log
	@echo | tee -a make.log
	@echo -e "\033[1;31m Compiling options \033[0m" | tee -a make.log
	@echo -e "\033[1m [$(OPTSC)]\033[0m" | tee -a make.log
	@echo | tee -a make.log
	@echo -e "\033[1;31m Linking options \033[0m" | tee -a make.log
	@echo -e "\033[1m [$(OPTSL) ] \033[0m" | tee -a make.log
	@echo | tee -a make.log

.PHONY : $(MKDIRS)
$(MKDIRS):
	@mkdir -p $@

.PHONY : cleaneps
cleaneps:
	@echo -e "\033[1;31m Deleting .eps files \033[0m" | tee make.log
	@rm *.eps

.PHONY : cleanopt
cleanopt:
	@echo -e "\033[1;31m Deleting .optrpt files \033[0m" | tee make.log
	@rm *.optrpt

.PHONY : cleanlog
cleanlog:
	@echo -e "\033[1;31m Deleting .log files \033[0m" | tee make.log
	@rm *.log

.PHONY : cleandat
cleandat:
	@echo -e "\033[1;31m Deleting .dat files \033[0m" | tee make.log
	@rm *.dat

.PHONY : cleanobj
cleanobj:
	@echo -e "\033[1;31m Deleting objects \033[0m" | tee make.log
	@rm -fr $(DOBJ)

.PHONY : cleanmod
cleanmod:
	@echo -e "\033[1;31m Deleting mods \033[0m" | tee -a make.log
	@rm -fr $(DMOD)

.PHONY : cleanexe
cleanexe:
	@echo -e "\033[1;31m Deleting exes \033[0m" | tee -a make.log
	@rm -f $(addprefix $(DEXE),shprot)

.PHONY : cleanmsg
cleanmsg:
	@rm -f diagnostic_messages
	@rm -f error_messages
	@rm -f make.log

.PHONY : clean
clean:
	@rm -rf *.o *.mod *.log *.eps $(DMOD) $(DOBJ) *.dat error_messages  diagnostic_messages *.optrpt
#clean: cleanobj cleanmod cleanmsg

.PHONY : cleanall
cleanall: clean cleanexe cleandat cleanopt cleanlog

.PHONY : tar
tar: cleanall
	@echo -e "\033[1;31m Creating tar archive of the code \033[0m" | tee make.log
	@rm -f shprot
	@mkdir -p shprot
	@cp -rL src makefile shprot/
	@tar czf shprot.tgz shprot
	@rm -rf shprot

.PHONY : doc
doc:
	@echo -e "\033[1;31m Building documentation \033[0m" | tee make.log
	@doxygen .doxygenconfig
#-----------------------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------------------
# rules of linking and compiling
COTEXT  = -e "\033[1;31m Compiling\033[0m\033[1m $(<F)\033[0m"
LITEXT  = -e "\033[1;31m Assembling\033[0m\033[1m $@\033[0m"

$(DEXE)shprot : PRINTINFO $(MKDIRS) $(DOBJ)main.o
	@echo | tee -a make.log
	@echo $(LITEXT) | tee -a make.log
	@$(FC) $(OPTSL) $(DOBJ)*.o $(LIB) -o $@ 1>> diagnostic_messages 2>> error_messages

$(DOBJ)odepack.o : odepack.f
	@echo $(COTEXT) | tee -a make.log
	@$(FC) $(OPTSC) $< -o $@ 1>> diagnostic_messages 2>> error_messages

$(DOBJ)odepack_sub1.o : odepack_sub1.f
	@echo $(COTEXT) | tee -a make.log
	@$(FC) $(OPTSC) $< -o $@ 1>> diagnostic_messages 2>> error_messages

$(DOBJ)odepack_sub2.o : odepack_sub2.f
	@echo $(COTEXT) | tee -a make.log
	@$(FC) $(OPTSC) $< -o $@ 1>> diagnostic_messages 2>> error_messages

$(DOBJ)OpenMP_dvode_f90_m.o : OpenMP_dvode_f90_m.f90
	@echo $(COTEXT) | tee -a make.log
	@$(FC) $(OPTSC) $< -o $@ 1>> diagnostic_messages 2>> error_messages

$(DOBJ)dvode_f90_m.o : dvode_f90_m.f90
	@echo $(COTEXT) | tee -a make.log
	@$(FC) $(OPTSC) $< -o $@ 1>> diagnostic_messages 2>> error_messages

$(DOBJ)kind_module.o : kind_module.f90
	@echo $(COTEXT) | tee -a make.log
	@$(FC) $(OPTSC) $< -o $@ 1>> diagnostic_messages 2>> error_messages

$(DOBJ)constants.o : constants.f90 \
	$(DOBJ)kind_module.o
	@echo $(COTEXT) | tee -a make.log
	@$(FC) $(OPTSC) $< -o $@ 1>> diagnostic_messages 2>> error_messages

$(DOBJ)brent_module.o : brent_module.f90 \
	$(DOBJ)kind_module.o \
	$(DOBJ)constants.o
	@echo $(COTEXT) | tee -a make.log
	@$(FC) $(OPTSC) $< -o $@ 1>> diagnostic_messages 2>> error_messages

$(DOBJ)init_energy.o : init_energy.f90 \
	$(DOBJ)constants.o
	@echo $(COTEXT) | tee -a make.log
	@$(FC) $(OPTSC) $< -o $@ 1>> diagnostic_messages 2>> error_messages

$(DOBJ)DR_Module.o : DR_Module.f90 \
	$(DOBJ)constants.o \
	$(DOBJ)kdis.o
	@echo $(COTEXT) | tee -a make.log
	@$(FC) $(OPTSC) $< -o $@ 1>> diagnostic_messages 2>> error_messages

$(DOBJ)DR_Module_NN.o : DR_Module_NN.f90 \
	$(DOBJ)constants.o
	@echo $(COTEXT) | tee -a make.log
	@$(FC) $(OPTSC) $< -o $@ 1>> diagnostic_messages 2>> error_messages

$(DOBJ)kdis.o : kdis.f90 \
	$(DOBJ)constants.o \
	$(DOBJ)init_energy.o
	@echo $(COTEXT) | tee -a make.log
	@$(FC) $(OPTSC) $< -o $@ 1>> diagnostic_messages 2>> error_messages

$(DOBJ)EX_Module.o : EX_Module.f90 \
	$(DOBJ)constants.o \
	$(DOBJ)k_ex_savelev_st.o
	@echo $(COTEXT) | tee -a make.log
	@$(FC) $(OPTSC) $< -o $@ 1>> diagnostic_messages 2>> error_messages

$(DOBJ)k_ex_savelev_st.o : k_ex_savelev_st.f90 \
	$(DOBJ)constants.o \
	$(DOBJ)init_energy.o
	@echo $(COTEXT) | tee -a make.log
	@$(FC) $(OPTSC) $< -o $@ 1>> diagnostic_messages 2>> error_messages

$(DOBJ)VT_Module.o : VT_Module.f90 \
	$(DOBJ)constants.o \
	$(DOBJ)kvt_fho.o
	@echo $(COTEXT) | tee -a make.log
	@$(FC) $(OPTSC) $< -o $@ 1>> diagnostic_messages 2>> error_messages

$(DOBJ)kvt_fho.o : kvt_fho.f90 \
	$(DOBJ)constants.o \
	$(DOBJ)brent_module.o
	@echo $(COTEXT) | tee -a make.log
	@$(FC) $(OPTSC) $< -o $@ 1>> diagnostic_messages 2>> error_messages

$(DOBJ)VV_Module.o : VV_Module.f90 \
	$(DOBJ)constants.o \
	$(DOBJ)kvv_fho.o
	@echo $(COTEXT) | tee -a make.log
	@$(FC) $(OPTSC) $< -o $@ 1>> diagnostic_messages 2>> error_messages

$(DOBJ)kvv_fho.o : kvv_fho.f90 \
	$(DOBJ)constants.o
	@echo $(COTEXT) | tee -a make.log
	@$(FC) $(OPTSC) $< -o $@ 1>> diagnostic_messages 2>> error_messages

### FKB
$(DOBJ)mod_kinds.o : FKB/src/lib/mod_kinds.F90
	@echo $(COTEXT) | tee -a make.log
	@$(FC) $(OPTSC) $< -o $@ 1>> diagnostic_messages 2>> error_messages

$(DOBJ)mod_io.o : FKB/src/lib/mod_io.F90 \
	$(DOBJ)mod_kinds.o
	@echo $(COTEXT) | tee -a make.log
	@$(FC) $(OPTSC) $< -o $@ 1>> diagnostic_messages 2>> error_messages

$(DOBJ)mod_activation.o : FKB/src/lib/mod_activation.F90 \
	$(DOBJ)mod_kinds.o
	@echo $(COTEXT) | tee -a make.log
	@$(FC) $(OPTSC) $< -o $@ 1>> diagnostic_messages 2>> error_messages

$(DOBJ)mod_layer.o : FKB/src/lib/mod_layer.F90 \
	$(DOBJ)mod_kinds.o \
	$(DOBJ)mod_activation.o
	@echo $(COTEXT) | tee -a make.log
	@$(FC) $(OPTSC) $< -o $@ 1>> diagnostic_messages 2>> error_messages

$(DOBJ)mod_batchnorm_layer.o : FKB/src/lib/mod_batchnorm_layer.F90 \
	$(DOBJ)mod_kinds.o \
	$(DOBJ)mod_activation.o
	@echo $(COTEXT) | tee -a make.log
	@$(FC) $(OPTSC) $< -o $@ 1>> diagnostic_messages 2>> error_messages

$(DOBJ)mod_dropout_layer.o : FKB/src/lib/mod_dropout_layer.F90 \
	$(DOBJ)mod_kinds.o \
	$(DOBJ)mod_layer.o
	@echo $(COTEXT) | tee -a make.log
	@$(FC) $(OPTSC) $< -o $@ 1>> diagnostic_messages 2>> error_messages

$(DOBJ)mod_batchnorm_layer.o : FKB/src/lib/mod_batchnorm_layer.F90 \
	$(DOBJ)mod_kinds.o \
	$(DOBJ)mod_layer.o
	@echo $(COTEXT) | tee -a make.log
	@$(FC) $(OPTSC) $< -o $@ 1>> diagnostic_messages 2>> error_messages

$(DOBJ)mod_random.o : FKB/src/lib/mod_random.F90 \
	$(DOBJ)mod_kinds.o
	@echo $(COTEXT) | tee -a make.log
	@$(FC) $(OPTSC) $< -o $@ 1>> diagnostic_messages 2>> error_messages

$(DOBJ)mod_dense_layer.o : FKB/src/lib/mod_dense_layer.F90 \
	$(DOBJ)mod_kinds.o \
	$(DOBJ)mod_activation.o \
	$(DOBJ)mod_random.o \
	$(DOBJ)mod_layer.o
	@echo $(COTEXT) | tee -a make.log
	@$(FC) $(OPTSC) $< -o $@ 1>> diagnostic_messages 2>> error_messages

$(DOBJ)mod_parallel.o : FKB/src/lib/mod_parallel.F90 \
	$(DOBJ)mod_kinds.o
	@echo $(COTEXT) | tee -a make.log
	@$(FC) $(OPTSC) $< -o $@ 1>> diagnostic_messages 2>> error_messages

$(DOBJ)mod_network.o : FKB/src/lib/mod_network.F90 \
	$(DOBJ)mod_kinds.o \
	$(DOBJ)mod_layer.o \
	$(DOBJ)mod_dropout_layer.o \
	$(DOBJ)mod_batchnorm_layer.o \
	$(DOBJ)mod_dense_layer.o \
	$(DOBJ)mod_parallel.o
	@echo $(COTEXT) | tee -a make.log
	@$(FC) $(OPTSC) $< -o $@ 1>> diagnostic_messages 2>> error_messages

$(DOBJ)mod_ensemble.o : FKB/src/lib/mod_ensemble.F90 \
	$(DOBJ)mod_kinds.o \
	$(DOBJ)mod_random.o \
	$(DOBJ)mod_network.o
	@echo $(COTEXT) | tee -a make.log
	@$(FC) $(OPTSC) $< -o $@ 1>> diagnostic_messages 2>> error_messages

$(DOBJ)rpart_fho.o : rpart_fho.f90 \
	$(DOBJ)constants.o \
	$(DOBJ)init_energy.o \
	$(DOBJ)DR_Module.o \
	$(DOBJ)DR_Module_NN.o \
	$(DOBJ)EX_Module.o \
	$(DOBJ)VT_Module.o \
	$(DOBJ)VV_Module.o
	@echo $(COTEXT) | tee -a make.log
	@$(FC) $(OPTSC) $< -o $@ 1>> diagnostic_messages 2>> error_messages

$(DOBJ)main.o : main.f90 \
	$(DOBJ)kind_module.o \
	$(DOBJ)mod_kinds.o \
	$(DOBJ)mod_network.o \
	$(DOBJ)mod_ensemble.o \
	$(DOBJ)constants.o \
	$(DOBJ)init_energy.o \
	$(DOBJ)rpart_fho.o \
	$(DOBJ)odepack.o \
	$(DOBJ)odepack_sub1.o \
	$(DOBJ)odepack_sub2.o \
	$(DOBJ)OpenMP_dvode_f90_m.o
	@echo $(COTEXT) | tee -a make.log
	@$(FC) $(OPTSC) $< -o $@ 1>> diagnostic_messages 2>> error_messages

#-----------------------------------------------------------------------------------------------------------------------------------
