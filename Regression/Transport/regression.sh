#!/bin/bash

# ----------------------------------
# Colors
# ----------------------------------
NOCOLOR='\033[0m'
RED='\033[0;31m'
GREEN='\033[0;32m'
ORANGE='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
LIGHTGRAY='\033[0;37m'
DARKGRAY='\033[1;30m'
LIGHTRED='\033[1;31m'
LIGHTGREEN='\033[1;32m'
YELLOW='\033[1;33m'
LIGHTBLUE='\033[1;34m'
LIGHTPURPLE='\033[1;35m'
LIGHTCYAN='\033[1;36m'
WHITE='\033[1;37m'
bold=$(tput bold)
normal=$(tput sgr0)

helpFunction()
{
   echo ""
   echo -e "${bold} Usage: $0 ${normal}-a ${RED}Algorithm${NOCOLOR} -p ${BLUE}Property${NOCOLOR}"
   echo -e "${RED}\t-a Algorithm: DT, RF, ET, GP, kNN, SVM, KR, GB, HGB, MLP, NN ${NOCOLOR}"
   echo -e "${BLUE}\t-p Property: shear, bulk, conductivity, thermo_diffusion, mass_diffusion ${NOCOLOR}"
   echo ""
   echo ""
   exit 1 # Exit script after printing help
}

while getopts ":a:p:" arg
do
  case $arg in
    a) Algorithm=$(echo $OPTARG | tr '[:lower:]' '[:upper:]');;
    p) Property=$( echo $OPTARG | tr '[:upper:]' '[:lower:]');;
    ?) helpFunction;; # Print helpFunction in case parameter is non-existent
  esac
done
echo ""
echo -e "Algorithm: $Algorithm\n"
echo -e "Property: $Property\n"

# Print helpFunction in case parameters are empty
if [ -z "$Algorithm" ] || [ -z "$Property" ]
then
   echo -e "${YELLOW} Some or all of the parameters are empty! ${NOCOLOR}";
   helpFunction
fi

# Redirection
 echo "Move to $Property/src/$Algorithm ..."
 cd $Property/src/$Algorithm
 echo "Regression of $Property with $Algorithm algorithm ..." 
 ./regression.py
 cd ../../..
 echo "Done!"
