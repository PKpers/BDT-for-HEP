#!/usr/bin/python
import ROOT
import numpy as np
from Training import load_data, get_project_root 
import sys
from sklearn.metrics import roc_curve, auc

def make_roc_curve(true_score, predictions, weights, out) -> float :
    '''
    Plots a roc curve and saves it to a file
    args:
    true_score(ndarray): The true score of the data
    predictions(ndarray): The model's prediction on data
    weights(ndarray): The data's weights
    out: /path/to/rocPlot/file.root
    '''
    fpr, tpr, _ = roc_curve(true_score, predictions , sample_weight=weights)
    auc_score = auc(fpr, tpr)

    # Plot and save the roc curve in a root file. The user doesn't need to worry abt this as the plotter file will read this and plot it in pdf 
    gFile = ROOT.TFile(out, "Recreate") 
    g = ROOT.TGraph(len(fpr), fpr, tpr)
    g.Write('roc_curve')
    gFile.Close()
    return auc_score


if len(sys.argv) != 5:
    print("Usage: {} {} {} {} {}" \
        .format(sys.argv[0], "/path/to/TrainingTesting/data/", "trainingTesting dataset name", "model name", "prediction name(output)"))
    exit(-1)
#

base = get_project_root()
base = str(base)

# Input settings
input_dir = sys.argv[1]
fileName = sys.argv[2]
TreeName = "myTree"

# Load data
infiles = [
    ('{}{}{}.root'.format(input_dir, fileName+str("_SIG_"), p),
     '{}{}{}.root'.format(input_dir, fileName+str("_BKG_"), p))
    for p in ('Test', 'Train')
]

xTest, yTest_true, wTest, _= load_data(infiles[0][0], infiles[0][1]) # Load the testing data set 
xTrain, yTrain_true, wTrain, _= load_data(infiles[1][0], infiles[1][1]) # Load th training data set

# Make sure the model's root file exists
modelName = sys.argv[3]
modelName =  modelName if ".root" in modelName else modelName + ".root"
modelPath = base +'/out/Models/'
model = modelPath + modelName

print("Started testing using data from {} with using the following model: {}".format(fileName, modelName))

if (ROOT.gSystem.AccessPathName(model)) :
    ROOT.Info(sys.argv[0], model+" does not exist")
    exit(-1)

# load the model 
bdt = ROOT.TMVA.Experimental.RBDT[""]("myBDT", model)
test = bdt.GetVariableImportance()
# Apply the model to the testing and training data
y_pred = [ bdt.Compute(x) for x in (xTest, xTrain) ]

# Output settings: Where to save the output of the model
outFileName = sys.argv[4]
outNameTest = outFileName + "test.root"
outNameTrain = outFileName + "train.root"
output_dir = base + '/out/Predictions/'
outFileTest, outFileTrain =[ output_dir + name for name in (outNameTest, outNameTrain) ]

# calculate the auc score and write the roc curve to a file
rocFileTest = base + "/out/Plots/" + outFileName + "Test_roc_curve.root" 
rocFileTrain = base + "/out/Plots/" + outFileName + "Train_roc_curve.root" 
test_auc = make_roc_curve(yTest_true, y_pred[0], wTest, rocFileTest)
train_auc = make_roc_curve(yTrain_true, y_pred[1], wTrain, rocFileTrain)

# Write the model's output on root files
print("exporting predicted data to {} and {} ...".format(outFileTest, outFileTrain))

dfTest = ROOT.RDF.FromNumpy( 
    { "yTest_pred" :      np.array(y_pred[0]), 
      "yTest_true" :      np.array(yTest_true), }
).Snapshot(TreeName, outFileTest) 

dfTrain = ROOT.RDF.FromNumpy(
    { "yTrain_pred" :        np.array(y_pred[1]),
      "yTrain_true" :        np.array(yTrain_true), }
).Snapshot(TreeName, outFileTrain)

print('done')
print(modelName,' auc score on testing data : ', test_auc)

