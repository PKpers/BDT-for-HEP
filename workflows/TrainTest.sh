#!/usr/bin/bash
check_exit(){
    local exit_stat=$1
    if [ $exit_stat -ne 0 ]; then
	exit
    fi
}

train_model() {
    local train_module=$1
    local path_data=$2
    local setdata=$3
    local cfg=$4
    local mdl=$5
    python $train_module $path_data $setdata $cfg $mdl
    ret=$?
    check_exit $ret
}

test_model(){
    local test_module=$1
    local path_data=$2
    local setdata=$3
    local mdl=$4
    local out=$5
    python $test $path_data $setdata $mdl $out
    ret=$?
    check_exit $ret
}

DIR="$( dirname -- "${BASH_SOURCE[0]}"; )";   # Get the directory name
DIR="$( realpath -e -- "$DIR"; )"; 
base="$(dirname "$DIR")" # Get the root of the progect 

data_path="/home/kpapad/UG_thesis/Thesis/share/SimuData/"
train=$base"/src/Training.py"
test=$base"/src/Testing.py"
dataset="WPhiJets_M200M100300Deltas"

k=1
config="training_conf"$k".yaml"
model="myModel_"$dataset"_conf"$k # to be used as output for train_model and input for test_model
output=$dataset"PConf"$k"Pred" # the output of the test_model

#train_model $train $data_path $dataset $config $model
test_model $test $data_path $dataset $model $output

