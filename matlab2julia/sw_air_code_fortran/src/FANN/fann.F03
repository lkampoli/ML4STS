!
!    fortran 2003 module interface for the Fast Artificial Neural Network
!    Library
!
!    Copyright (C) 2016 C. Ringeval
!
!    This program is free software: you can redistribute it and/or modify
!    it under the terms of the GNU General Public License as published by
!    the Free Software Foundation, either version 3 of the License, or
!    (at your option) any later version.
!
!    This program is distributed in the hope that it will be useful,
!    but WITHOUT ANY WARRANTY; without even the implied warranty of
!    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
!    GNU General Public License for more details.
!
!    You should have received a copy of the GNU General Public License
!    along with this program.  If not, see <http://www.gnu.org/licenses/>.
!
!    email: chris@ringeval.info

module fann
  use, intrinsic :: iso_c_binding

#ifdef DOUBLEFANN
  integer, parameter :: FANN_TYPE = C_DOUBLE
#elif defined FIXEDFANN
  integer, parameter :: FANN_TYPE = C_INT
#else
  integer, parameter :: FANN_TYPE = C_FLOAT
#endif


  type, bind(C) :: fann_connection
     integer(C_INT) :: from_neuron
     integer(C_INT) :: to_neuron
#ifdef FIXEDFANN
     integer(FANN_TYPE) :: weight
#else
     real(FANN_TYPE) :: weight
#endif
  end type fann_connection


  enum, bind(C)
     enumerator :: FANN_TRAIN_INCREMENTAL=0, &
          FANN_TRAIN_BATCH, &
          FANN_TRAIN_RPROP, &
          FANN_TRAIN_QUICKPROP, &
          FANN_TRAIN_SARPROP
  end enum

  enum, bind(C)
     enumerator :: FANN_LINEAR = 0, &
          FANN_THRESHOLD, &
          FANN_THRESHOLD_SYMMETRIC, &
          FANN_SIGMOID, &
          FANN_SIGMOID_STEPWISE, &
          FANN_SIGMOID_SYMMETRIC, &
          FANN_SIGMOID_SYMMETRIC_STEPWISE, &
          FANN_GAUSSIAN, &
          FANN_GAUSSIAN_SYMMETRIC, &
          FANN_GAUSSIAN_STEPWISE, &
          FANN_ELLIOT, &
          FANN_ELLIOT_SYMMETRIC, &
          FANN_LINEAR_PIECE, &
          FANN_LINEAR_PIECE_SYMMETRIC, &
          FANN_SIN_SYMMETRIC, &
          FANN_COS_SYMMETRIC, &
          FANN_SIN, &
          FANN_COS
  end enum


  enum, bind(C)
     enumerator :: FANN_ERRORFUNC_LINEAR = 0, &
          FANN_ERRORFUNC_TANH
  end enum


  enum, bind(C)
     enumerator :: FANN_STOPFUNC_MSE = 0, &
          FANN_STOPFUNC_BIT
  end enum

  enum, bind(C)
     enumerator :: FANN_NETTYPE_LAYER = 0, &
          FANN_NETTYPE_SHORTCUT
  end enum




  interface 


     function fann_create_from_file(configuration_file) bind(C)
       import C_PTR, C_CHAR
       implicit none
       type(C_PTR) :: fann_create_from_file
       character(C_CHAR), dimension(*) :: configuration_file
     end function fann_create_from_file
   


     function fann_save(ann,configuration_file) bind(C)
       import C_PTR, C_INT, C_CHAR
       implicit none
       integer(C_INT) :: fann_save
       type(C_PTR), value :: ann
       character(C_CHAR), dimension(*) :: configuration_file
     end function fann_save



     function fann_save_to_fixed(ann,configuration_file) bind(C)
       import C_PTR, C_INT, C_CHAR
       implicit none
       integer(C_INT) :: fann_save_to_fixed
       type(C_PTR), value :: ann
       character(C_CHAR), dimension(*) :: configuration_file
     end function fann_save_to_fixed


     
     function fann_create_standard_array(num_layers,layers) bind(C)
       import C_PTR, C_INT
       implicit none         
       type(C_PTR) :: fann_create_standard_array
       integer(C_INT), value :: num_layers
       integer(C_INT), dimension(*) :: layers
     end function fann_create_standard_array



     function fann_create_sparse_array(connection_rate,num_layers,layers) bind(C)
       import C_PTR, C_INT, C_FLOAT
       implicit none         
       type(C_PTR) :: fann_create_sparse_array
       real(C_FLOAT), value :: connection_rate
       integer(C_INT), value :: num_layers
       integer(C_INT), dimension(*) :: layers
     end function fann_create_sparse_array



     function fann_create_shortcut_array(num_layers,layers) bind(C)
       import C_PTR, C_INT
       implicit none         
       type(C_PTR) :: fann_create_shortcut_array
       integer(C_INT), value :: num_layers
       integer(C_INT), dimension(*) :: layers
     end function fann_create_shortcut_array



     subroutine fann_destroy(ann) bind(C)
       import C_PTR
       implicit none
       type(C_PTR), value :: ann
     end subroutine fann_destroy



     function fann_copy(ann) bind(C)
       import C_PTR
       implicit none
       type(C_PTR), value :: ann
       type(C_PTR) :: fann_copy       
     end function fann_copy



     function fann_run(ann,input) bind(C)
       import C_PTR, FANN_TYPE
       implicit none
       type(C_PTR) :: fann_run
       type(C_PTR), value :: ann
#ifdef FIXEDFANN
       integer(FANN_TYPE), dimension(*) :: input
#else
       real(FANN_TYPE), dimension(*) :: input
#endif
     end function fann_run



     subroutine fann_randomize_weights(ann,min_weight,max_weight) bind(C)
       import C_PTR, C_DOUBLE, C_FLOAT, FANN_TYPE
       implicit none
       type(C_PTR), value :: ann
#ifdef FIXEDFANN
       integer(FANN_TYPE), value :: min_weight, max_weight
#else
       real(FANN_TYPE), value :: min_weight, max_weight
#endif
     end subroutine fann_randomize_weights



     subroutine fann_init_weights(ann,data) bind(C)
       import C_PTR
       implicit none
       type(C_PTR), value :: ann, data
     end subroutine fann_init_weights



     subroutine fann_print_connections(ann) bind(C)
       import C_PTR
       implicit none
       type(C_PTR), value :: ann    
     end subroutine fann_print_connections



     subroutine fann_print_parameters(ann) bind(C)
       import C_PTR
       implicit none
       type(C_PTR), value :: ann
     end subroutine fann_print_parameters



     function fann_get_num_input(ann) bind(C)
       import C_PTR, C_INT
       implicit none
       integer(C_INT) :: fann_get_num_input
       type(C_PTR), value :: ann
     end function fann_get_num_input



     function fann_get_num_output(ann) bind(C)
       import C_PTR, C_INT
       implicit none
       integer(C_INT) :: fann_get_num_output
       type(C_PTR), value :: ann
     end function fann_get_num_output



     function fann_get_total_neurons(ann) bind(C)
       import C_PTR, C_INT
       implicit none
       integer(C_INT) :: fann_get_total_neurons
       type(C_PTR), value :: ann
     end function fann_get_total_neurons



     function fann_get_total_connections(ann) bind(C)
       import C_PTR, C_INT
       implicit none
       integer(C_INT) :: fann_get_total_connections
       type(C_PTR), value :: ann
     end function fann_get_total_connections



     function fann_get_network_type(ann) bind(C)
       import C_PTR, C_INT
       implicit none
       integer(C_INT) :: fann_get_network_type
       type(C_PTR), value :: ann
     end function fann_get_network_type



     function fann_get_connection_rate(ann) bind(C)
       import C_PTR, C_FLOAT
       implicit none
       real(C_FLOAT) :: fann_get_connection_rate
       type(C_PTR), value :: ann
     end function fann_get_connection_rate



     function fann_get_num_layers(ann) bind(C)
       import C_PTR, C_INT
       implicit none
       integer(C_INT) :: fann_get_num_layers
       type(C_PTR), value :: ann
     end function fann_get_num_layers



     subroutine fann_get_layer_array(ann,layers) bind(C)
       import C_PTR, C_INT
       implicit none
       type(C_PTR), value :: ann
       integer(C_INT), dimension(*) :: layers
     end subroutine fann_get_layer_array



     subroutine fann_get_bias_array(ann,biases) bind(C)
       import C_PTR, C_INT
       implicit none
       type(C_PTR), value :: ann
       integer(C_INT), dimension(*) :: biases
     end subroutine fann_get_bias_array



     subroutine fann_get_connection_array(ann,connections) bind(C)
       import C_PTR, fann_connection
       implicit none
       type(C_PTR), value :: ann
       type(fann_connection), dimension(*) :: connections
     end subroutine fann_get_connection_array



     subroutine fann_set_weight_array(ann,connections,num_connections) bind(C)
       import C_PTR, C_INT, fann_connection
       implicit none
       type(C_PTR), value :: ann
       type(fann_connection), dimension(*) :: connections
       integer(C_INT), value :: num_connections
     end subroutine fann_set_weight_array



     subroutine fann_set_weight(ann,from_neuron,to_neuron,weight) bind(C)
       import C_PTR, C_INT, FANN_TYPE
       implicit none
       type(C_PTR), value :: ann
       integer(C_INT), value :: from_neuron, to_neuron
#ifdef FIXEDFANN
       integer(FANN_TYPE), value :: weight
#else
       real(FANN_TYPE), value :: weight
#endif
     end subroutine fann_set_weight


#ifdef FIXEDFANN
     function fann_get_decimal_point(ann) bind(C)
       import C_PTR, C_INT
       implicit none
       integer(C_INT) :: fann_get_decimal_point
       type(C_PTR), value :: ann
     end function fann_get_decimal_point



     function fann_get_multiplier(ann) bind(C)
       import C_PTR, C_INT
       implicit none
       integer(C_INT) :: fann_get_multiplier
       type(C_PTR), value :: ann
     end function fann_get_multiplier
#endif


     subroutine fann_train(ann,input,output) bind(C)
       import C_PTR, FANN_TYPE
       implicit none
       type(C_PTR), value :: ann
#ifdef FIXEDFANN
       integer(FANN_TYPE), dimension(*) :: input, output
#else
       real(FANN_TYPE), dimension(*) :: input, output
#endif      
     end subroutine fann_train



     function fann_test(ann,input,desired_output) bind(C)
       import C_PTR, FANN_TYPE
       implicit none
       type(C_PTR) :: fann_test
       type(C_PTR), value :: ann
#ifdef FIXEDFANN
       integer(FANN_TYPE), dimension(*) :: input
       integer(FANN_TYPE), dimension(*) :: desired_output
#else
       real(FANN_TYPE), dimension(*) :: input
       real(FANN_TYPE), dimension(*) :: desired_output
#endif
     end function fann_test



     function fann_get_mse(ann) bind(C, name='fann_get_MSE')
       import C_PTR, C_FLOAT
       implicit none
       real(C_FLOAT) :: fann_get_mse
       type(C_PTR), value :: ann
     end function fann_get_mse



     function fann_get_bit_fail(ann) bind(C)
       import C_PTR, C_INT
       implicit none
       integer(C_INT) :: fann_get_bit_fail
       type(C_PTR), value :: ann
     end function fann_get_bit_fail



     subroutine fann_reset_mse(ann) bind(C, name='fann_reset_MSE')
       import C_PTR
       implicit none
       type(C_PTR), value :: ann
     end subroutine fann_reset_mse



     subroutine fann_train_on_data(ann,data,max_epochs,epochs_between_reports,desired_error) bind(C)
       import C_PTR, C_FLOAT, C_INT
       implicit none
       type(C_PTR), value :: ann, data
       integer(C_INT), value :: max_epochs, epochs_between_reports
       real(C_FLOAT), value :: desired_error
     end subroutine fann_train_on_data



     subroutine fann_train_on_file(ann,chain,max_epochs,epochs_between_reports,desired_error) bind(C)
       import C_PTR, C_FLOAT, C_INT, C_CHAR
       implicit none
       type(C_PTR), value :: ann
       character(C_CHAR), dimension(*) :: chain
       integer(C_INT), value :: max_epochs, epochs_between_reports
       real(C_FLOAT), value :: desired_error
     end subroutine fann_train_on_file



     function fann_train_epoch(ann,data) bind(C)
       import C_PTR, C_FLOAT
       implicit none
       type(C_PTR), value :: ann, data
       real(C_FLOAT) :: fann_train_epoch
     end function fann_train_epoch



     function fann_test_data(ann,data) bind(C)
       import C_PTR, C_FLOAT
       implicit none
       type(C_PTR), value :: ann, data
       real(C_FLOAT) :: fann_test_data
     end function fann_test_data



     function fann_read_train_from_file(chain) bind(C)
       import C_PTR, C_CHAR
       implicit none
       type(C_PTR) :: fann_read_train_from_file
       character(C_CHAR), dimension(*) :: chain
     end function fann_read_train_from_file



     function fann_create_train(num_data, num_input, num_output) bind(C)
       import C_PTR, C_INT
       implicit none
       type(C_PTR) :: fann_create_train
       integer(C_INT), value :: num_data, num_input, num_output
     end function fann_create_train



     function fann_create_train_from_callback(num_data, num_input, num_output, user_function) bind(C)
       import C_PTR, C_INT, C_FUNPTR
       implicit none
       type(C_PTR) :: fann_create_train_from_callback
       integer(C_INT), value :: num_data, num_input, num_output
       type(C_FUNPTR), value :: user_function
     end function fann_create_train_from_callback



     subroutine fann_destroy_train(data) bind(C)
       import C_PTR
       implicit none
       type(C_PTR), value :: data       
     end subroutine fann_destroy_train



     subroutine fann_shuffle_train_data(data) bind(C)
       import C_PTR
       implicit none
       type(C_PTR), value :: data
     end subroutine fann_shuffle_train_data



     subroutine fann_scale_train(ann,data) bind(C)
       import C_PTR
       implicit none
       type(C_PTR), value :: ann, data
     end subroutine fann_scale_train
     


     subroutine fann_descale_train(ann,data) bind(C)
       import C_PTR
       implicit none
       type(C_PTR), value :: ann, data
     end subroutine fann_descale_train



     function fann_set_input_scaling_params(ann,data,new_input_min,new_input_max) bind(C)
       import C_PTR, C_INT, C_FLOAT
       implicit none
       integer(C_INT) :: fann_set_input_scaling_params
       type(C_PTR), value :: ann, data
       real(C_FLOAT), value :: new_input_min,new_input_max
     end function fann_set_input_scaling_params



     function fann_set_output_scaling_params(ann,data,new_output_min,new_output_max) bind(C)
       import C_PTR, C_INT, C_FLOAT
       implicit none
       integer(C_INT) :: fann_set_output_scaling_params
       type(C_PTR), value :: ann, data
       real(C_FLOAT), value :: new_output_min,new_output_max
     end function fann_set_output_scaling_params



     function fann_set_scaling_params(ann,data,new_input_min,new_input_max &
          ,new_output_min,new_output_max) bind(C)
       import C_PTR, C_INT, C_FLOAT
       implicit none
       integer(C_INT) :: fann_set_scaling_params
       type(C_PTR), value :: ann, data
       real(C_FLOAT), value :: new_input_min, new_input_max
       real(C_FLOAT), value :: new_output_min, new_output_max
     end function fann_set_scaling_params



     function fann_clear_scaling_params(ann) bind(C)
       import C_PTR, C_INT
       implicit none
       integer(C_INT) :: fann_clear_scaling_params
       type(C_PTR), value :: ann
     end function fann_clear_scaling_params



     subroutine fann_scale_input(ann,input_vector) bind(C)
       import C_PTR, FANN_TYPE
       implicit none
       type(C_PTR), value :: ann
#ifdef FIXEDFANN
       integer(FANN_TYPE), dimension(*) :: input_vector
#else
       real(FANN_TYPE), dimension(*) :: input_vector
#endif    
     end subroutine fann_scale_input



     subroutine fann_scale_output(ann,output_vector) bind(C)
       import C_PTR, FANN_TYPE
       implicit none
       type(C_PTR), value :: ann
#ifdef FIXEDFANN
       integer(FANN_TYPE), dimension(*) :: output_vector
#else
       real(FANN_TYPE), dimension(*) :: output_vector
#endif      
     end subroutine fann_scale_output

     
     
     subroutine fann_descale_input(ann,input_vector) bind(C)
       import C_PTR, FANN_TYPE
       implicit none
       type(C_PTR), value :: ann
#ifdef FIXEDFANN
       integer(FANN_TYPE), dimension(*) :: input_vector
#else
       real(FANN_TYPE), dimension(*) :: input_vector
#endif      
     end subroutine fann_descale_input



     subroutine fann_descale_output(ann,output_vector) bind(C)
       import C_PTR, FANN_TYPE
       implicit none
       type(C_PTR), value :: ann
#ifdef FIXEDFANN
       integer(FANN_TYPE), dimension(*) :: output_vector
#else
       real(FANN_TYPE), dimension(*) :: output_vector
#endif      
     end subroutine fann_descale_output



     subroutine fann_scale_input_train_data(data,new_min,new_max) bind(C)
       import C_PTR, FANN_TYPE
       implicit none
       type(C_PTR), value :: data
#ifdef FIXEDFANN
       integer(FANN_TYPE), value :: new_min
       integer(FANN_TYPE), value :: new_max
#else
       real(FANN_TYPE), value :: new_min
       real(FANN_TYPE), value :: new_max
#endif       
     end subroutine fann_scale_input_train_data



     subroutine fann_scale_output_train_data(data,new_min,new_max) bind(C)
       import C_PTR, FANN_TYPE
       implicit none
       type(C_PTR), value :: data
#ifdef FIXEDFANN
       integer(FANN_TYPE), value :: new_min
       integer(FANN_TYPE), value :: new_max
#else
       real(FANN_TYPE), value :: new_min
       real(FANN_TYPE), value :: new_max
#endif       
     end subroutine fann_scale_output_train_data



    subroutine fann_scale_train_data(data,new_min,new_max) bind(C)
      import C_PTR, FANN_TYPE
      implicit none
      type(C_PTR), value :: data
#ifdef FIXEDFANN
      integer(FANN_TYPE), value :: new_min
      integer(FANN_TYPE), value :: new_max
#else
      real(FANN_TYPE), value :: new_min
      real(FANN_TYPE), value :: new_max
#endif       
    end subroutine fann_scale_train_data



    function fann_merge_train_data(data1,data2) bind(C)
      import C_PTR
      implicit none
      type(C_PTR) :: fann_merge_train_data
      type(C_PTR), value :: data1, data2
    end function fann_merge_train_data



    function fann_duplicate_train_data(data) bind(C)
      import C_PTR
      implicit none
      type(C_PTR) :: fann_duplicate_train_data
      type(C_PTR), value :: data
    end function fann_duplicate_train_data
    


    function fann_subset_train_data(data,pos,length) bind(C)
      import C_PTR, C_INT
      implicit none
      type(C_PTR) :: fann_subset_train_data
      type(C_PTR), value :: data
      integer(C_INT), value :: pos, length
    end function fann_subset_train_data



    function fann_length_train_data(data) bind(C)
      import C_PTR, C_INT
      implicit none
      integer(C_INT) :: fann_length_train_data
      type(C_PTR), value :: data
    end function fann_length_train_data



    function fann_num_input_train_data(data) bind(C)
       import C_PTR, C_INT
       implicit none
       type(C_PTR), value :: data
       integer(C_INT) :: fann_num_input_train_data
     end function fann_num_input_train_data



     function fann_num_output_train_data(data) bind(C)
       import C_PTR, C_INT
       implicit none
       type(C_PTR), value :: data
       integer(C_INT) :: fann_num_output_train_data
     end function fann_num_output_train_data



    subroutine fann_save_train(data,chain) bind(C)
       import C_PTR, C_CHAR
       implicit none
       type(C_PTR), value :: data
       character(C_CHAR), dimension(*) :: chain
     end subroutine fann_save_train


          
     function fann_save_train_to_fixed(data,filename,decimal_point) bind(C)
       import C_PTR, C_INT, C_CHAR
       implicit none
       integer(C_INT) :: fann_save_train_to_fixed
       type(C_PTR), value :: data
       character(C_CHAR), dimension(*) :: filename
       integer(C_INT), value :: decimal_point
     end function fann_save_train_to_fixed



     function fann_get_training_algorithm(ann) bind(C)
       import C_PTR, C_INT
       implicit none
       integer(C_INT) :: fann_get_training_algorithm
       type(C_PTR), value :: ann
     end function fann_get_training_algorithm



     subroutine fann_set_training_algorithm(ann, training_algorithm) bind(C)
       import C_PTR, C_INT
       implicit none
       type(C_PTR), value :: ann
       integer(C_INT), value :: training_algorithm
     end subroutine fann_set_training_algorithm



     function fann_get_learning_rate(ann) bind(C)
       import C_PTR, C_FLOAT
       implicit none
       real(C_FLOAT) :: fann_get_learning_rate
       type(C_PTR), value :: ann       
     end function fann_get_learning_rate



     subroutine fann_set_learning_rate(ann,learning_rate) bind(C)
       import C_PTR, C_FLOAT
       implicit none
       type(C_PTR), value :: ann
       real(C_FLOAT), value :: learning_rate
     end subroutine fann_set_learning_rate



     function fann_get_learning_momentum(ann) bind(C)
       import C_PTR, C_FLOAT
       type(C_PTR), value :: ann
       real(C_FLOAT) :: fann_get_learning_momentum
     end function fann_get_learning_momentum



     subroutine fann_set_learning_momentum(ann,learning_momentum) bind(C)
       import C_PTR, C_FLOAT
       implicit none
       type(C_PTR), value :: ann
       real(C_FLOAT), value :: learning_momentum
     end subroutine fann_set_learning_momentum



     function fann_get_activation_function(ann,layer,neuron) bind(C)
       import C_PTR, C_INT
       implicit none
       integer(C_INT) :: fann_get_activation_function
       type(C_PTR), value :: ann
       integer(C_INT), value :: layer, neuron       
     end function fann_get_activation_function



     subroutine fann_set_activation_function(ann,activation_function,layer,neuron) bind(C)
       import C_PTR, C_INT
       implicit none
       type(C_PTR), value :: ann
       integer(C_INT), value :: activation_function
       integer(C_INT), value :: layer, neuron       
     end subroutine fann_set_activation_function



     subroutine fann_set_activation_function_layer(ann,activation_function,layer) bind(C)
       import C_PTR, C_INT
       implicit none
       type(C_PTR), value :: ann
       integer(C_INT), value :: activation_function
       integer(C_INT), value :: layer
     end subroutine fann_set_activation_function_layer



     subroutine fann_set_activation_function_hidden(ann,activation_function) bind(C)
       import C_PTR, C_INT
       implicit none
       type(C_PTR), value :: ann
       integer(C_INT), value :: activation_function
     end subroutine fann_set_activation_function_hidden



     subroutine fann_set_activation_function_output(ann,activation_function) bind(C)
       import C_PTR, C_INT
       implicit none
       type(C_PTR), value :: ann
       integer(C_INT), value :: activation_function
     end subroutine fann_set_activation_function_output



     function fann_get_activation_steepness(ann,layer,neuron) bind(C)
       import FANN_TYPE, C_INT, C_PTR
       implicit none
#ifdef FIXEDFANN
       integer(FANN_TYPE) :: fann_get_activation_steepness
#else
       real(FANN_TYPE) :: fann_get_activation_steepness
#endif

       type(C_PTR), value :: ann
       integer(C_INT), value :: layer, neuron
     end function fann_get_activation_steepness


     subroutine fann_set_activation_steepness(ann,steepness,layer,neuron) bind(C)
       import FANN_TYPE, C_INT, C_PTR
       implicit none
#ifdef FIXEDFANN
       integer(FANN_TYPE), value :: steepness
#else
       real(FANN_TYPE), value :: steepness
#endif
       type(C_PTR), value :: ann
       integer(C_INT), value :: layer, neuron
     end subroutine  fann_set_activation_steepness



     subroutine fann_set_activation_steepness_layer(ann,steepness,layer) bind(C)
       import FANN_TYPE, C_INT, C_PTR
       implicit none
#ifdef FIXEDFANN
       integer(FANN_TYPE), value :: steepness
#else
       real(FANN_TYPE), value :: steepness
#endif
       type(C_PTR), value :: ann
       integer(C_INT), value :: layer
     end subroutine  fann_set_activation_steepness_layer



     subroutine fann_set_activation_steepness_hidden(ann,steepness) bind(C)
       import FANN_TYPE, C_PTR
       implicit none
#ifdef FIXEDFANN
       integer(FANN_TYPE), value :: steepness
#else
       real(FANN_TYPE), value :: steepness
#endif
       type(C_PTR), value :: ann
     end subroutine  fann_set_activation_steepness_hidden



     subroutine fann_set_activation_steepness_output(ann,steepness) bind(C)
       import FANN_TYPE, C_PTR
       implicit none
#ifdef FIXEDFANN
       integer(FANN_TYPE), value :: steepness
#else
       real(FANN_TYPE), value :: steepness
#endif
       type(C_PTR), value :: ann
     end subroutine  fann_set_activation_steepness_output



     function fann_get_train_error_function(ann) bind(C)
       import C_PTR, C_INT
       implicit none
       integer(C_INT) :: fann_get_train_error_function
       type(C_PTR), value :: ann
     end function  fann_get_train_error_function



     subroutine fann_set_train_error_function(ann,train_error_function) bind(C)
       import C_PTR, C_INT
       implicit none
       type(C_PTR), value :: ann
       integer(C_INT), value :: train_error_function
     end subroutine fann_set_train_error_function



     function fann_get_train_stop_function(ann) bind(C)
       import C_PTR, C_INT
       implicit none
       integer(C_INT) :: fann_get_train_stop_function
       type(C_PTR), value :: ann
     end function  fann_get_train_stop_function



     subroutine fann_set_train_stop_function(ann,train_stop_function) bind(C)
       import C_PTR, C_INT
       implicit none
       type(C_PTR), value :: ann
       integer(C_INT), value :: train_stop_function
     end subroutine fann_set_train_stop_function



     function fann_get_bit_fail_limit(ann) bind(C)
       import C_PTR, FANN_TYPE
       implicit none
#ifdef FIXEDFANN
       integer(FANN_TYPE) :: fann_get_bit_fail_limit
#else
       real(FANN_TYPE) :: fann_get_bit_fail_limit
#endif
       type(C_PTR), value :: ann
     end function fann_get_bit_fail_limit



     subroutine fann_set_bit_fail_limit(ann,bit_fail_limit) bind(C)
       import C_PTR, FANN_TYPE
       implicit none
       type(C_PTR), value :: ann
#ifdef FIXEDFANN
       integer(FANN_TYPE), value :: bit_fail_limit
#else
       real(FANN_TYPE), value :: bit_fail_limit
#endif
     end subroutine fann_set_bit_fail_limit



     function fann_get_quickprop_decay(ann) bind(C)
       import C_PTR, C_FLOAT
       implicit none
       real(C_FLOAT) :: fann_get_quickprop_decay
       type(C_PTR), value :: ann
     end function fann_get_quickprop_decay



     subroutine fann_set_quickprop_decay(ann,quickprop_decay) bind(C)
       import C_PTR, C_FLOAT
       implicit none
       type(C_PTR), value :: ann
       real(C_FLOAT), value :: quickprop_decay
     end subroutine fann_set_quickprop_decay



     function fann_get_quickprop_mu(ann) bind(C)
       import C_PTR, C_FLOAT
       implicit none
       real(C_FLOAT) :: fann_get_quickprop_mu
       type(C_PTR), value ::  ann
     end function fann_get_quickprop_mu



     subroutine fann_set_quickprop_mu(ann,quickprop_mu) bind(C)
       import C_PTR, C_FLOAT
       implicit none
       type(C_PTR), value :: ann
       real(C_FLOAT), value :: quickprop_mu
     end subroutine fann_set_quickprop_mu



     function fann_get_rprop_increase_factor(ann) bind(C)
       import C_PTR, C_FLOAT
       implicit none
       real(C_FLOAT) :: fann_get_rprop_increase_factor
       type(C_PTR), value :: ann
     end function fann_get_rprop_increase_factor



     subroutine fann_set_rprop_increase_factor(ann,rprop_increase_factor) bind(C)
       import C_PTR, C_FLOAT
       implicit none
       type(C_PTR), value :: ann
       real(C_FLOAT), value :: rprop_increase_factor
     end subroutine fann_set_rprop_increase_factor



     function fann_get_rprop_decrease_factor(ann) bind(C)
       import C_PTR, C_FLOAT
       implicit none
       real(C_FLOAT) :: fann_get_rprop_decrease_factor
       type(C_PTR), value :: ann
     end function fann_get_rprop_decrease_factor



     subroutine fann_set_rprop_decrease_factor(ann,rprop_decrease_factor) bind(C)
       import C_PTR, C_FLOAT
       implicit none
       type(C_PTR), value :: ann
       real(C_FLOAT), value :: rprop_decrease_factor
     end subroutine fann_set_rprop_decrease_factor



     function fann_get_rprop_delta_min(ann) bind(C)
       import C_PTR, C_FLOAT
       implicit none
       real(C_FLOAT) :: fann_get_rprop_delta_min
       type(C_PTR), value :: ann
     end function fann_get_rprop_delta_min



     subroutine fann_set_rprop_delta_min(ann,rprop_delta_min) bind(C)
       import C_PTR, C_FLOAT
       implicit none
       type(C_PTR), value :: ann
       real(C_FLOAT), value :: rprop_delta_min
     end subroutine fann_set_rprop_delta_min



     function fann_get_rprop_delta_max(ann) bind(C)
       import C_PTR, C_FLOAT
       implicit none
       real(C_FLOAT) :: fann_get_rprop_delta_max
       type(C_PTR), value :: ann
     end function fann_get_rprop_delta_max



     subroutine fann_set_rprop_delta_max(ann,rprop_delta_max) bind(C)
       import C_PTR, C_FLOAT
       implicit none
       type(C_PTR), value :: ann
       real(C_FLOAT), value :: rprop_delta_max
     end subroutine fann_set_rprop_delta_max



     function fann_get_rprop_delta_zero(ann) bind(C)
       import C_PTR, C_FLOAT
       implicit none
       real(C_FLOAT) :: fann_get_rprop_delta_zero
       type(C_PTR), value :: ann
     end function fann_get_rprop_delta_zero



     subroutine fann_set_rprop_delta_zero(ann,rprop_delta_zero) bind(C)
       import C_PTR, C_FLOAT
       implicit none
       type(C_PTR), value :: ann
       real(C_FLOAT), value :: rprop_delta_zero
     end subroutine fann_set_rprop_delta_zero



     function fann_get_sarprop_weight_decay_shift(ann) bind(C)
       import C_PTR, C_FLOAT
       implicit none
       real(C_FLOAT) :: fann_get_sarprop_weight_decay_shift
       type(C_PTR), value :: ann
     end function fann_get_sarprop_weight_decay_shift



     subroutine fann_set_sarprop_weight_decay_shift(ann,sarprop_weight_decay_shift) bind(C)
       import C_PTR, C_FLOAT
       implicit none
       type(C_PTR), value :: ann
       real(C_FLOAT), value :: sarprop_weight_decay_shift
     end subroutine fann_set_sarprop_weight_decay_shift



     function fann_get_sarprop_step_error_threshold_factor(ann) bind(C)
       import C_PTR, C_FLOAT
       implicit none
       real(C_FLOAT) :: fann_get_sarprop_step_error_threshold_factor
       type(C_PTR), value :: ann
     end function fann_get_sarprop_step_error_threshold_factor



     subroutine fann_set_sarprop_step_error_threshold_factor(ann,sarprop_step_error_threshold_factor) bind(C)
       import C_PTR, C_FLOAT
       implicit none
       type(C_PTR), value :: ann
       real(C_FLOAT), value :: sarprop_step_error_threshold_factor
     end subroutine fann_set_sarprop_step_error_threshold_factor



     function fann_get_sarprop_step_error_shift(ann) bind(C)
       import C_PTR, C_FLOAT
       implicit none
       real(C_FLOAT) :: fann_get_sarprop_step_error_shift
       type(C_PTR), value :: ann
     end function fann_get_sarprop_step_error_shift



     subroutine fann_set_sarprop_step_error_shift(ann,sarprop_step_error_shift) bind(C)
       import C_PTR, C_FLOAT
       implicit none
       type(C_PTR), value :: ann
       real(C_FLOAT), value :: sarprop_step_error_shift
     end subroutine fann_set_sarprop_step_error_shift



     function fann_get_sarprop_temperature(ann) bind(C)
       import C_PTR, C_FLOAT
       implicit none
       real(C_FLOAT) :: fann_get_sarprop_temperature
       type(C_PTR), value :: ann
     end function fann_get_sarprop_temperature



     subroutine fann_set_sarprop_temperature(ann,sarprop_temperature) bind(C)
       import C_PTR, C_FLOAT
       implicit none
       type(C_PTR), value :: ann
       real(C_FLOAT), value :: sarprop_temperature
     end subroutine fann_set_sarprop_temperature



     subroutine fann_cascadetrain_on_data(ann,data,max_neurons,neurons_between_reports &
          ,desired_error) bind(C)
       import C_PTR, C_INT, C_FLOAT
       implicit none
       type(C_PTR), value :: ann, data
       integer(C_INT), value :: max_neurons, neurons_between_reports
       real(C_FLOAT), value :: desired_error
     end subroutine fann_cascadetrain_on_data



     subroutine fann_cascadetrain_on_file(ann,filename,max_neurons,neurons_between_reports &
          ,desired_error) bind(C)
       import C_PTR, C_INT, C_FLOAT, C_CHAR
       implicit none
       type(C_PTR), value :: ann
       character(C_CHAR), dimension(*) :: filename
       integer(C_INT), value :: max_neurons, neurons_between_reports
       real(C_FLOAT), value :: desired_error
     end subroutine fann_cascadetrain_on_file



     function fann_get_cascade_output_change_fraction(ann) bind(C)
       import C_PTR, C_FLOAT
       implicit none
       real(C_FLOAT) :: fann_get_cascade_output_change_fraction
       type(C_PTR), value :: ann
     end function fann_get_cascade_output_change_fraction



     subroutine fann_set_cascade_output_change_fraction(ann,cascade_output_change_fraction) bind(C)
       import C_PTR, C_FLOAT
       implicit none
       type(C_PTR), value :: ann
       real(C_FLOAT), value :: cascade_output_change_fraction
     end subroutine fann_set_cascade_output_change_fraction



     function fann_get_cascade_output_stagnation_epochs(ann) bind(C)
       import C_PTR, C_INT
       implicit none
       integer(C_INT) :: fann_get_cascade_output_stagnation_epochs
       type(C_PTR), value :: ann
     end function fann_get_cascade_output_stagnation_epochs



     subroutine fann_set_cascade_output_stagnation_epochs(ann,cascade_output_stagnation_epochs) bind(C)
       import C_PTR, C_INT
       implicit none
       type(C_PTR), value :: ann
       integer(C_INT), value :: cascade_output_stagnation_epochs
     end subroutine fann_set_cascade_output_stagnation_epochs
     

     
     function fann_get_cascade_candidate_change_fraction(ann) bind(C)
       import C_PTR, C_FLOAT
       implicit none
       real(C_FLOAT) :: fann_get_cascade_candidate_change_fraction
       type(C_PTR), value :: ann
     end function fann_get_cascade_candidate_change_fraction



     subroutine fann_set_cascade_candidate_change_fraction(ann,cascade_candidate_change_fraction) bind(C)
       import C_PTR, C_FLOAT
       implicit none       
       type(C_PTR), value :: ann
       real(C_FLOAT), value :: cascade_candidate_change_fraction
     end subroutine fann_set_cascade_candidate_change_fraction



     function fann_get_cascade_candidate_stagnation_epochs(ann) bind(C)
       import C_PTR, C_INT
       implicit none
       integer(C_INT) :: fann_get_cascade_candidate_stagnation_epochs
       type(C_PTR), value :: ann
     end function fann_get_cascade_candidate_stagnation_epochs



     subroutine fann_set_cascade_candidate_stagnation_epochs(ann,cascade_candidate_stagnation_epochs) bind(C)
       import C_PTR, C_INT
       implicit none
       type(C_PTR), value :: ann
       integer(C_INT), value :: cascade_candidate_stagnation_epochs
     end subroutine fann_set_cascade_candidate_stagnation_epochs



     function fann_get_cascade_weight_multiplier(ann) bind(C)
       import C_PTR, FANN_TYPE
       implicit none
#ifdef FIXEDFANN
       integer(FANN_TYPE) :: fann_get_cascade_weight_multiplier
#else
       real(FANN_TYPE) :: fann_get_cascade_weight_multiplier
#endif
       type(C_PTR), value :: ann     
     end function fann_get_cascade_weight_multiplier



     subroutine fann_set_cascade_weight_multiplier(ann,cascade_weight_multiplier) bind(C)
       import C_PTR, FANN_TYPE
       implicit none
       type(C_PTR), value :: ann
#ifdef FIXEDFANN
       integer(FANN_TYPE), value :: cascade_weight_multiplier
#else
       real(FANN_TYPE), value :: cascade_weight_multiplier
#endif
     end subroutine fann_set_cascade_weight_multiplier



     function fann_get_cascade_candidate_limit(ann) bind(C)
       import C_PTR, FANN_TYPE
       implicit none
#ifdef FIXEDFANN
       integer(FANN_TYPE) :: fann_get_cascade_candidate_limit
#else
       real(FANN_TYPE) :: fann_get_cascade_candidate_limit
#endif
       type(C_PTR), value :: ann     
     end function fann_get_cascade_candidate_limit
       


     subroutine fann_set_cascade_candidate_limit(ann,cascade_candidate_limit) bind(C)
       import C_PTR, FANN_TYPE
       implicit none
       type(C_PTR), value :: ann
#ifdef FIXEDFANN
       integer(FANN_TYPE), value :: cascade_candidate_limit
#else
       real(FANN_TYPE), value :: cascade_candidate_limit
#endif
     end subroutine fann_set_cascade_candidate_limit



     function fann_get_cascade_max_out_epochs(ann) bind(C)
       import C_PTR, C_INT
       implicit none
       integer(C_INT) :: fann_get_cascade_max_out_epochs
       type(C_PTR), value :: ann
     end function fann_get_cascade_max_out_epochs



     subroutine fann_set_cascade_max_out_epochs(ann,cascade_max_out_epochs) bind(C)
       import C_PTR, C_INT
       implicit none
       type(C_PTR), value :: ann
       integer(C_INT), value :: cascade_max_out_epochs
     end subroutine fann_set_cascade_max_out_epochs



     function fann_get_cascade_min_out_epochs(ann) bind(C)
       import C_PTR, C_INT
       implicit none
       integer(C_INT) :: fann_get_cascade_min_out_epochs
       type(C_PTR), value :: ann
     end function fann_get_cascade_min_out_epochs



     subroutine fann_set_cascade_min_out_epochs(ann,cascade_min_out_epochs) bind(C)
       import C_PTR, C_INT
       implicit none
       type(C_PTR), value :: ann
       integer(C_INT), value :: cascade_min_out_epochs
     end subroutine fann_set_cascade_min_out_epochs



     function fann_get_cascade_max_cand_epochs(ann) bind(C)
       import C_PTR, C_INT
       implicit none
       integer(C_INT) :: fann_get_cascade_max_cand_epochs
       type(C_PTR), value :: ann
     end function fann_get_cascade_max_cand_epochs



     subroutine fann_set_cascade_max_cand_epochs(ann,cascade_max_cand_epochs) bind(C)
       import C_PTR, C_INT
       implicit none
       type(C_PTR), value :: ann
       integer(C_INT), value :: cascade_max_cand_epochs
     end subroutine fann_set_cascade_max_cand_epochs



     function fann_get_cascade_min_cand_epochs(ann) bind(C)
       import C_PTR, C_INT
       implicit none
       integer(C_INT) :: fann_get_cascade_min_cand_epochs
       type(C_PTR), value :: ann
     end function fann_get_cascade_min_cand_epochs



     subroutine fann_set_cascade_min_cand_epochs(ann,cascade_min_cand_epochs) bind(C)
       import C_PTR, C_INT
       implicit none
       type(C_PTR), value :: ann
       integer(C_INT), value :: cascade_min_cand_epochs
     end subroutine fann_set_cascade_min_cand_epochs

     

     function fann_get_cascade_num_candidates(ann) bind(C)
       import C_PTR, C_INT
       implicit none
       integer(C_INT) :: fann_get_cascade_num_candidates
       type(C_PTR), value :: ann
     end function fann_get_cascade_num_candidates



     function fann_get_cascade_activation_functions_count(ann) bind(C)
       import C_PTR, C_INT
       implicit none
       integer(C_INT) :: fann_get_cascade_activation_functions_count
       type(C_PTR), value :: ann
     end function fann_get_cascade_activation_functions_count



     function fann_get_cascade_activation_functions(ann) bind(C)
       import C_PTR, C_INT
       implicit none
       type(C_PTR) :: fann_get_cascade_activation_functions
       type(C_PTR), value :: ann
     end function fann_get_cascade_activation_functions



     subroutine fann_set_cascade_activation_functions(ann,cascade_activation_functions &
          ,cascade_activation_functions_count) bind(C)
       import C_PTR, C_INT
       implicit none
       type(C_PTR), value :: ann
       integer(C_INT), dimension(*) :: cascade_activation_functions
       integer(C_INT), value :: cascade_activation_functions_count
     end subroutine fann_set_cascade_activation_functions



     function fann_get_cascade_activation_steepnesses_count(ann) bind(C)
       import C_PTR, C_INT
       implicit none
       integer(C_INT) :: fann_get_cascade_activation_steepnesses_count
       type(C_PTR), value :: ann
     end function fann_get_cascade_activation_steepnesses_count



     function fann_get_cascade_activation_steepnesses(ann) bind(C)
       import C_PTR, FANN_TYPE
       implicit none
       type(C_PTR) :: fann_get_cascade_activation_steepnesses
       type(C_PTR), value :: ann
     end function fann_get_cascade_activation_steepnesses
     


     subroutine fann_set_cascade_activation_steepnesses(ann,cascade_activation_steepnesses &
          ,cascade_activation_steepnesses_count) bind(C)
       import C_PTR, C_INT, FANN_TYPE
       implicit none
       type(C_PTR), value :: ann
#ifdef FIXEDFANN
       integer(FANN_TYPE), dimension(*) :: cascade_activation_steepnesses
#else
       real(FANN_TYPE), dimension(*) :: cascade_activation_steepnesses
#endif
       integer(C_INT), value :: cascade_activation_steepnesses_count
     end subroutine fann_set_cascade_activation_steepnesses



     function fann_get_cascade_num_candidate_groups(ann) bind(C)
       import C_PTR, C_INT
       implicit none
       integer(C_INT) :: fann_get_cascade_num_candidate_groups
       type(C_PTR), value :: ann
     end function fann_get_cascade_num_candidate_groups

       

     subroutine fann_set_cascade_num_candidate_groups(ann,cascade_num_candidate_groups) bind(C)
       import C_PTR, C_INT
       implicit none
       type(C_PTR), value :: ann
       integer(C_INT), value :: cascade_num_candidate_groups
     end subroutine fann_set_cascade_num_candidate_groups



#ifdef FANN23

     function fann_create_train_array(num_data,num_input,input,num_output,output) bind(C)
       import C_PTR, FANN_TYPE
       implicit none
       type(C_PTR) :: fann_create_train_array
       integer(C_INT), value :: num_data, num_input, num_output
#ifdef FIXEDFANN
       integer(FANN_TYPE), dimension(*) :: input,output
#else
       real(FANN_TYPE), dimension(*) :: input, output
#endif

     end function fann_create_train_array


     function fann_get_train_input(data,position) bind(C)
       import C_PTR, C_INT, FANN_TYPE
       implicit none
#ifdef FIXEDFANN
       integer(FANN_TYPE) :: fann_get_train_input
#else
       real(FANN_TYPE) :: fann_get_train_input
#endif
       type(C_PTR), value :: data
       integer(C_INT), value :: position
     end function fann_get_train_input



     function fann_get_train_output(data,position) bind(C)
       import C_PTR, C_INT, FANN_TYPE
       implicit none
#ifdef FIXEDFANN
       integer(FANN_TYPE) :: fann_get_train_output
#else
       real(FANN_TYPE) :: fann_get_train_output
#endif
       type(C_PTR), value :: data
       integer(C_INT), value :: position
     end function fann_get_train_output

     

     function fann_get_min_train_input(data) bind(C)
       import C_PTR, C_INT, FANN_TYPE
       implicit none
#ifdef FIXEDFANN
       integer(FANN_TYPE) :: fann_get_min_train_input
#else
       real(FANN_TYPE) :: fann_get_min_train_input
#endif
       type(C_PTR), value :: data
     end function fann_get_min_train_input



     function fann_get_max_train_input(data) bind(C)
       import C_PTR, C_INT, FANN_TYPE
       implicit none
#ifdef FIXEDFANN
       integer(FANN_TYPE) :: fann_get_max_train_input
#else
       real(FANN_TYPE) :: fann_get_max_train_input
#endif
       type(C_PTR), value :: data
     end function fann_get_max_train_input

    
     
     function fann_get_min_train_output(data) bind(C)
       import C_PTR, C_INT, FANN_TYPE
       implicit none
#ifdef FIXEDFANN
       integer(FANN_TYPE) :: fann_get_min_train_output
#else
       real(FANN_TYPE) :: fann_get_min_train_output
#endif
       type(C_PTR), value :: data
     end function fann_get_min_train_output



     function fann_get_max_train_output(data) bind(C)
       import C_PTR, C_INT, FANN_TYPE
       implicit none
#ifdef FIXEDFANN
       integer(FANN_TYPE) :: fann_get_max_train_output
#else
       real(FANN_TYPE) :: fann_get_max_train_output
#endif
       type(C_PTR), value :: data
     end function fann_get_max_train_output


#endif



    
  end interface



contains

!some wrappers and utility functions based on fortran types


  function f_c_string(fname)
    implicit none
    character(len=*), intent(in) :: fname
    character(kind=C_CHAR, len=len(fname)+1) :: f_c_string
    integer :: i,n

    n = len(fname)
    do i=1,n
       f_c_string(i:i) = fname(i:i)
    enddo
    f_c_string(n+1:n+1)=C_NULL_CHAR

  end function f_c_string



  function f_fann_run(ann,input)    
    implicit none
#ifdef FIXEDFANN
    integer(FANN_TYPE), dimension(:), intent(in) :: input
    integer(FANN_TYPE), dimension(:), pointer :: f_fann_run
#else
    real(FANN_TYPE), dimension(:), intent(in) :: input
    real(FANN_TYPE), dimension(:), pointer :: f_fann_run
#endif
    type(C_PTR) :: ann
    type(C_PTR) :: this

    this = fann_run(ann,input)

    call C_F_POINTER(this,f_fann_run,shape=[fann_get_num_output(ann)])

  end function f_fann_run



  function f_fann_get_cascade_activation_functions(ann)
    implicit none
    integer(C_INT), dimension(:), pointer :: f_fann_get_cascade_activation_functions
    type(C_PTR) :: ann
    type(C_PTR) :: this

    this = fann_get_cascade_activation_functions(ann)

    call C_F_POINTER(this,f_fann_get_cascade_activation_functions &
         ,shape=[fann_get_cascade_activation_functions_count(ann)])

  end function f_fann_get_cascade_activation_functions


  function f_fann_get_cascade_activation_steepnesses(ann)
    implicit none
#ifdef FIXEDFANN
    integer(FANN_TYPE), dimension(:), pointer :: f_fann_get_cascade_activation_steepnesses
#else
    real(FANN_TYPE), dimension(:), pointer :: f_fann_get_cascade_activation_steepnesses
#endif
    type(C_PTR) :: ann
    type(C_PTR) :: this

    this = fann_get_cascade_activation_steepnesses(ann)

    call C_F_POINTER(this,f_fann_get_cascade_activation_steepnesses &
         ,shape=[fann_get_cascade_activation_steepnesses_count(ann)])

  end function f_fann_get_cascade_activation_steepnesses



  function enum_training_algorithm(algoname) result(algoid)
    implicit none
    character(len=*), intent(in) :: algoname
    integer(C_INT) :: algoid

    select case (algoname)

    case ('FANN_TRAIN_INCREMENTAL')
       algoid = FANN_TRAIN_INCREMENTAL

    case ('FANN_TRAIN_BATCH')
       algoid = FANN_TRAIN_BATCH

    case ('FANN_TRAIN_RPROP')
       algoid = FANN_TRAIN_RPROP

    case ('FANN_TRAIN_QUICKPROP')
       algoid = FANN_TRAIN_QUICKPROP

    case ('FANN_TRAIN_SARPROP')
       algoid = FANN_TRAIN_SARPROP

    case default
       stop 'enum_training_algorithm: name not found!'

    end select

  end function enum_training_algorithm



  function enum_activation_function(actiname) result(actifunc)
    implicit none
    character(len=*), intent(in) :: actiname
    integer(C_INT) :: actifunc

    select case (actiname)

    case ('FANN_LINEAR')
       actifunc = FANN_LINEAR
    case ('FANN_THRESHOLD')
       actifunc = FANN_THRESHOLD
    case ('FANN_THRESHOLD_SYMMETRIC')
       actifunc = FANN_THRESHOLD_SYMMETRIC
    case ('FANN_SIGMOID')
       actifunc = FANN_SIGMOID
    case ('FANN_SIGMOID_STEPWISE')
       actifunc = FANN_SIGMOID_STEPWISE
    case ('FANN_SIGMOID_SYMMETRIC')
       actifunc = FANN_SIGMOID_SYMMETRIC
    case ('FANN_SIGMOID_SYMMETRIC_STEPWISE')
       actifunc = FANN_SIGMOID_SYMMETRIC_STEPWISE
    case ('FANN_GAUSSIAN')
       actifunc = FANN_GAUSSIAN
    case ('FANN_GAUSSIAN_SYMMETRIC')
       actifunc = FANN_GAUSSIAN_SYMMETRIC
    case ('FANN_GAUSSIAN_STEPWISE')
       actifunc = FANN_GAUSSIAN_STEPWISE
    case ('FANN_ELLIOT')
       actifunc = FANN_ELLIOT
    case ('FANN_ELLIOT_SYMMETRIC')
       actifunc = FANN_ELLIOT_SYMMETRIC
    case ('FANN_LINEAR_PIECE')
       actifunc = FANN_LINEAR_PIECE
    case ('FANN_LINEAR_PIECE_SYMMETRIC')
       actifunc = FANN_LINEAR_PIECE_SYMMETRIC
    case ('FANN_SIN_SYMMETRIC')
       actifunc = FANN_SIN_SYMMETRIC
    case ('FANN_COS_SYMMETRIC')
       actifunc = FANN_COS_SYMMETRIC
    case ('FANN_SIN')
       actifunc = FANN_SIN
    case ('FANN_COS')
       actifunc = FANN_COS
    case default
       stop 'enum_activation_function: name not found!'
    end select

  end function enum_activation_function



  function enum_error_function(errorname) result(errorfunc)
    implicit none
    character(len=*), intent(in) :: errorname
    integer(C_INT) :: errorfunc

    select case (errorname)

    case ('FANN_ERRORFUNC_LINEAR')
       errorfunc = FANN_ERRORFUNC_LINEAR
    case ('FANN_ERRORFUNC_TANH')
       errorfunc = FANN_ERRORFUNC_TANH
    case default
       stop 'enum_error_function: name not found!'
    end select

  end function enum_error_function



  function enum_stop_function(stopname) result(stopfunc)
    implicit none
    character(len=*), intent(in) :: stopname
    integer(C_INT) :: stopfunc

    select case (stopname)

    case ('FANN_STOPFUNC_MSE')
       stopfunc = FANN_STOPFUNC_MSE
    case ('FANN_STOPFUNC_BIT')
       stopfunc = FANN_STOPFUNC_BIT
    case default
       stop 'enum_stop_function: name not found!'
    end select

  end function enum_stop_function



  function enum_network_type(netname) result(nettype)
    implicit none
    character(len=*), intent(in) :: netname
    integer(C_INT) :: nettype

    select case (netname)

    case ('FANN_NETTYPE_LAYER')
       nettype = FANN_NETTYPE_LAYER
    case ('FANN_NETTYPE_SHORTCUT')
       nettype = FANN_NETTYPE_SHORTCUT
    case default
       stop 'enum_network_type: name not found!'
    end select

  end function enum_network_type


end module fann
