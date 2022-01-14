/*
Fast Artificial Neural Network Library (fann)
Copyright (C) 2003-2016 Steffen Nissen (steffen.fann@gmail.com)

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

#include <stdio.h>
#include "fann.h"

int main()
{
	const float desired_error = (const float) 0.00001;
	const unsigned int max_epochs = 10000;
	const unsigned int epochs_between_reports = 10;
	struct fann_train_data *train_data, *test_data;

  const unsigned int num_input = 1;
  const unsigned int num_output = 1; //126;
	const unsigned int num_layers = 3;
	const unsigned int num_neurons_hidden = 50;

	unsigned int i = 0;

  //struct fann *ann = fann_create_standard(num_layers, num_input, num_neurons_hidden, num_output);
  //struct fann *ann = fann_create_standard(5, 1, 50, 50, 50, 126);
  //struct fann *ann = fann_create_standard(5, 1, 50, 50, 50, 1);
  struct fann *ann = fann_create_standard(3, 1, 50, 1);

  fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
  //fann_set_activation_function_hidden(ann, FANN_SIGMOID);

  //fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);
	//fann_set_activation_function_output(ann, FANN_SIGMOID);
  fann_set_activation_function_output(ann, FANN_LINEAR);

  fann_set_train_error_function(ann, FANN_ERRORFUNC_LINEAR);
  //fann_set_train_error_function(ann, FANN_ERRORFUNC_TANH);

  //fann_set_training_algorithm(ann, FANN_TRAIN_RPROP);
  //fann_set_training_algorithm(ann, FANN_TRAIN_SARPROP);
  //fann_set_training_algorithm(ann, FANN_TRAIN_BATCH);
  //fann_set_training_algorithm(ann, FANN_TRAIN_QUICKPROP);
	//fann_set_training_algorithm(ann, FANN_TRAIN_INCREMENTAL);

  //fann_set_learning_momentum(ann, 0.7);

  //fann_set_learning_rate(ann, 0.001);

	//train_data = fann_read_train_from_file("../datasets/dataXY12mod.dat");
	train_data = fann_read_train_from_file("../datasets/dataXY12modT.dat");

  fann_set_scaling_params(ann, train_data,
       -1,  /* New input minimum  */
        1,  /* New input maximum  */
       -1,  /* New output minimum */
        1); /* New output maximum */

  //fann_scale_train_data(train_data, -1, 1);

  //fann_scale_train(ann, train_data);

  //fann_shuffle_train_data(train_data);

  fann_train_on_data(ann, train_data,                      max_epochs, epochs_between_reports, desired_error);
  //fann_train_on_file(ann, "../datasets/dataXY12mod.dat", max_epochs, epochs_between_reports, desired_error);

	fann_save(ann, "inference_ODE_XY_train.net");
  fann_destroy_train(train_data);
	fann_destroy(ann);

	return 0;
}
