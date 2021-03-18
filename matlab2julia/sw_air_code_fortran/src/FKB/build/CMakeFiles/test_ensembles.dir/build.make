# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/lk/Public/ML4STS/Euler_1d_shock_STS/data/sw_air_code_fortran/src/FKB

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/lk/Public/ML4STS/Euler_1d_shock_STS/data/sw_air_code_fortran/src/FKB/build

# Include any dependencies generated for this target.
include CMakeFiles/test_ensembles.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/test_ensembles.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/test_ensembles.dir/flags.make

CMakeFiles/test_ensembles.dir/src/tests/test_ensembles.F90.o: CMakeFiles/test_ensembles.dir/flags.make
CMakeFiles/test_ensembles.dir/src/tests/test_ensembles.F90.o: ../src/tests/test_ensembles.F90
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lk/Public/ML4STS/Euler_1d_shock_STS/data/sw_air_code_fortran/src/FKB/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building Fortran object CMakeFiles/test_ensembles.dir/src/tests/test_ensembles.F90.o"
	/usr/bin/gfortran $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -c /home/lk/Public/ML4STS/Euler_1d_shock_STS/data/sw_air_code_fortran/src/FKB/src/tests/test_ensembles.F90 -o CMakeFiles/test_ensembles.dir/src/tests/test_ensembles.F90.o

CMakeFiles/test_ensembles.dir/src/tests/test_ensembles.F90.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing Fortran source to CMakeFiles/test_ensembles.dir/src/tests/test_ensembles.F90.i"
	/usr/bin/gfortran $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -E /home/lk/Public/ML4STS/Euler_1d_shock_STS/data/sw_air_code_fortran/src/FKB/src/tests/test_ensembles.F90 > CMakeFiles/test_ensembles.dir/src/tests/test_ensembles.F90.i

CMakeFiles/test_ensembles.dir/src/tests/test_ensembles.F90.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling Fortran source to assembly CMakeFiles/test_ensembles.dir/src/tests/test_ensembles.F90.s"
	/usr/bin/gfortran $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -S /home/lk/Public/ML4STS/Euler_1d_shock_STS/data/sw_air_code_fortran/src/FKB/src/tests/test_ensembles.F90 -o CMakeFiles/test_ensembles.dir/src/tests/test_ensembles.F90.s

# Object files for target test_ensembles
test_ensembles_OBJECTS = \
"CMakeFiles/test_ensembles.dir/src/tests/test_ensembles.F90.o"

# External object files for target test_ensembles
test_ensembles_EXTERNAL_OBJECTS =

bin/test_ensembles: CMakeFiles/test_ensembles.dir/src/tests/test_ensembles.F90.o
bin/test_ensembles: CMakeFiles/test_ensembles.dir/build.make
bin/test_ensembles: lib/libneural.a
bin/test_ensembles: CMakeFiles/test_ensembles.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/lk/Public/ML4STS/Euler_1d_shock_STS/data/sw_air_code_fortran/src/FKB/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking Fortran executable bin/test_ensembles"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_ensembles.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test_ensembles.dir/build: bin/test_ensembles

.PHONY : CMakeFiles/test_ensembles.dir/build

CMakeFiles/test_ensembles.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/test_ensembles.dir/cmake_clean.cmake
.PHONY : CMakeFiles/test_ensembles.dir/clean

CMakeFiles/test_ensembles.dir/depend:
	cd /home/lk/Public/ML4STS/Euler_1d_shock_STS/data/sw_air_code_fortran/src/FKB/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/lk/Public/ML4STS/Euler_1d_shock_STS/data/sw_air_code_fortran/src/FKB /home/lk/Public/ML4STS/Euler_1d_shock_STS/data/sw_air_code_fortran/src/FKB /home/lk/Public/ML4STS/Euler_1d_shock_STS/data/sw_air_code_fortran/src/FKB/build /home/lk/Public/ML4STS/Euler_1d_shock_STS/data/sw_air_code_fortran/src/FKB/build /home/lk/Public/ML4STS/Euler_1d_shock_STS/data/sw_air_code_fortran/src/FKB/build/CMakeFiles/test_ensembles.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/test_ensembles.dir/depend

