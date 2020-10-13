 echo "Running DecisionTree ... "
 cd DecisionTree                           ; py2nb regression.py ; cd ..
 echo "Running ExtraTrees ... "
 cd ExtraTrees                             ; py2nb regression.py ; cd ..
 echo "Running GaussianProcess ... "
#cd GaussianProcess                        ; py2nb regression.py ; cd ..
 echo "Running GradientBoosting ... "
 cd GradientBoosting                       ; py2nb regression.py ; cd ..
 echo "Running HistGradientBoosting ... "
 cd HistGradientBoosting                   ; py2nb regression.py ; cd ..
 echo "Running KernelRidge ... "
 cd KernelRidge                            ; py2nb regression.py ; cd ..
 echo "Running KNeighbor ... "
 cd KNeighbor                              ; py2nb regression.py ; cd ..
 echo "Running MultiLayerPerceptron ... "
 cd MultiLayerPerceptron                   ; py2nb regression.py ; cd ..
 echo "Running RandomForest ... "
 cd RandomForest                           ; py2nb regression.py ; cd ..
 echo "Running SupportVectorMachines ... "
 cd SupportVectorMachines                  ; py2nb regression.py ; cd ..
