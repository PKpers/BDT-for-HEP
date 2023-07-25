#!/usr/bin/bash
plot="/home/kpapad/UG_thesis/Thesis/Bdt/src/plotResults.py"
predictions=()
outputs=()
dataset="WPhiJets_M60M5080Deltas"
k=13
for (( i=$k ; i<=$k ; i++));
do
    prediction=$dataset"PConf"$i"Pred"
    output=$dataset"PConf"$i"BDTplot.pdf"
    python $plot $prediction $output
done
exit
# add the feature importance plot to the plots pdf
_path="/home/kpapad/UG_thesis/Thesis/Bdt/out/Plots/"
feat_imp=$_path"feature_importance.pdf"
plots=$_path$output
if [ -f $feat_imp ]; then
    echo "merging feature importance plot with the rest in "$plots"..."
    mutool merge -o $plots $plots* $feat_imp 1 
    rm $feat_imp
    echo "done"
fi
