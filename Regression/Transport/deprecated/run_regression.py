#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

def main():
    parser = argparse.ArgumentParser(description='transport coefficients regression')

    parser.add_argument('-p', '--property', type=str,
                        choices=['shear', 'bulk', 'conductivity', 'thermo_diffusion', 'mass_diffusion'],
                        default='shear,bulk,conductivity,thermo_diffusion,mass_diffusion',
                        help='Comma-separated names of properties whose regression is performed')

    parser.add_argument('-a', '--algorithm', type=str,
                        choices=['DT', 'RF', 'ET', 'GP', 'kNN', 'SVM', 'KR', 'GB', 'HGB'],
                        default='DT',
                        help='regression algorithm')

    args = parser.parse_args()
    print(args)

    property = args.property.split(',')
    print(property)

    algorithm = args.algorithm.split(',')
    print(algorithm)

    if algorithm == 'DT':
         from shear.src.DT.regressor import a
         print('calling ...')
#    	 a()


if __name__ == "__main__":
    main()
