 echo "Running DecisionTree ... "
 cd DecisionTree;          python3 regression.py;  cd ..
 echo "Running ExtraTrees ... "
 cd ExtraTrees;            python3 regression.py;  cd ..
 echo "Running GaussianProcess ... "
#cd GaussianProcess;       python3 regression.py;  cd ..
 echo "Running GradientBoosting ... "
 cd GradientBoosting;      python3 regression.py;  cd ..
 echo "Running HistGradientBoosting ... "
 cd HistGradientBoosting;  python3 regression.py;  cd ..
 echo "Running KernelRidge ... "
 cd KernelRidge; python3   regression.py;          cd ..
 echo "Running KNeighbor ... "
 cd KNeighbor; python3     regression.py;          cd ..
 echo "Running MultiLayerPerceptron ... "
 cd MultiLayerPerceptron;  python3 regression.py;  cd ..
 echo "Running RandomForest ... "
 cd RandomForest;          python3 regression.py;  cd ..
 echo "Running SupportVectorMachines ... "
 cd SupportVectorMachines; python3 regression.py;  cd ..
