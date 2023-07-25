#!/usr/bin/bash
check_exit(){
    local exit_stat=$1
    if [ $exit_stat -ne 0 ]; then
	exit
    fi
}

# Set up the testing
num=13
test="/home/kpapad/UG_thesis/Thesis/Bdt/src/TestSmeared.py"
dataset="WPhiJets_M60M5080Deltas_Application_Smeared12"
#dataset="WPhiJets_M60M5080Deltas_Application"
modelname="WPhiJets_M60M5080Deltas"
model="myModel"$modelname"_conf"$num

# Setup plotting
plot="/home/kpapad/UG_thesis/Thesis/Bdt/src/plotResultsSmeared.py"
prediction=$dataset"Pred"$num
output_plot=$dataset$num"BDTplot.pdf"

# Test and plot
output_test=$dataset"Pred"$num
python $test $dataset $model $output_test 
ret=$?
check_exit $ret
python $plot $prediction $output_plot
ret=$?
check_exit $ret
