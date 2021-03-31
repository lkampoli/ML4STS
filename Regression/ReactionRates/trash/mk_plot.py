#!/usr/bin/env python
# -*- coding: utf-8 -*

import matplotlib.pyplot as plt

def draw_plot(x_test_dim, y_test_dim, y_regr_dim, figure, data):
   
   plt.scatter(x_test_dim, y_test_dim[:,5], s=2, c='k', marker='o', label='Matlab')
   plt.scatter(x_test_dim, y_regr_dim[:,5], s=2, c='purple', marker='+', label='DT, i=5')

   plt.scatter(x_test_dim, y_test_dim[:,10], s=2, c='k', marker='o', label='Matlab')
   plt.scatter(x_test_dim, y_regr_dim[:,10], s=2, c='r', marker='+', label='DT, i=10')

   plt.scatter(x_test_dim, y_test_dim[:,15], s=2, c='k', marker='o', label='Matlab')
   plt.scatter(x_test_dim, y_regr_dim[:,15], s=2, c='c', marker='+', label='DT, i=15')

   plt.scatter(x_test_dim, y_test_dim[:,20], s=2, c='k', marker='o', label='Matlab')
   plt.scatter(x_test_dim, y_regr_dim[:,20], s=2, c='g', marker='+', label='DT, i=20')

   plt.scatter(x_test_dim, y_test_dim[:,25], s=2, c='k', marker='o', label='Matlab')
   plt.scatter(x_test_dim, y_regr_dim[:,25], s=2, c='y', marker='+', label='DT, i=25')

   plt.scatter(x_test_dim, y_test_dim[:,30], s=2, c='k', marker='o', label='Matlab')
   plt.scatter(x_test_dim, y_regr_dim[:,30], s=2, c='b', marker='+', label='DT, i=30')

   plt.scatter(x_test_dim, y_test_dim[:,35], s=2, c='k', marker='o', label='Matlab')
   plt.scatter(x_test_dim, y_regr_dim[:,35], s=2, c='m', marker='+', label='DT, i=35')

   #plt.ylabel(r'$\eta$ [Pa·s]')
   plt.xlabel('T [K] ')
   plt.legend()
   plt.tight_layout()
   plt.savefig(figure+"/regression_MO_"+data+'.pdf')
   plt.show()
   plt.close()
