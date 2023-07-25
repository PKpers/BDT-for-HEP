#!/usr/bin/python
import ROOT
import sys
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from asdict import read_as_dict
from Training import load_data
from sklearn.model_selection import GridSearchCV

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: {} {} {}".format(sys.argv[0], "training_dataset", "training_config"))
        exit(-1)

    # Load data
    # Load the training data 
    dataset = sys.argv[1]
    inpath = '/home/kpapad/UG_thesis/Thesis/share/SimuData/'
    sig_filename = inpath+ "{}_SIG_Train.root".format(dataset)
    bkg_filename = inpath+ "{}_BKG_Train.root".format(dataset)
    x, y, w, num_all= load_data(sig_filename, bkg_filename)
    
    # Load training config
    config_dir="/home/kpapad/UG_thesis/Thesis/Bdt/config/"
    training_config = config_dir + sys.argv[2]
    print("loading training configuration: {}".format(training_config))

    training_config = read_as_dict(training_config)
    param = list( training_config.items() )
    print('Training started with data from {} and {}'.format(sig_filename, bkg_filename)) 

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=0)
    
    # Create the list for validation set
    dval = [(X_val, y_val)]

    # First tune two important parameters
    bdt = XGBClassifier(**training_config)
    training_set = [x, y, w]

    tune_params={
        #'scale_pos_weight' : [0.5, 0.8, 1, 1.1]
        #'n_estimators' : [1000, 1500, 2000, 2500, 3000],
        #'learning_rate' : [0.01, 0.03, 0.05, 0.07, 0.09]
        'reg_alpha' : [i/100 for i in range(1, 10, 2)], 
        'reg_lambda' : range(1, 10, 2),
        #'reg_alpha' : [0.0, 0.1, 0.2]
        #'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100],
        #'reg_lambda':[1e-5, 1e-2, 0.1, 1, 100],
        #'reg_lambda': range(50, 100, 10) 
        #'reg_lambda' : range(1, 10, 2) 
        #'max_depth': [6, 7, 8, 9],
        #'min_child_weight' : [6,7,8,9]
        #'learning_rate': [0.4, 0.5, 0.6],
        # 'subsample':[i/10.0 for i in range(0,10, 2)],
        #'colsample_bytree':[i/10.0 for i in range(0,10, 2)]
        #'gamma' : [i/10.0 for i in range(0,5)] 
        #'objective': ['binary:logistic'],
        #'early_stopping_rounds': [10],
    }
    grid = GridSearchCV(bdt, tune_params, cv=5, scoring='accuracy', n_jobs=4)
    grid.fit(X_train, y_train, eval_set=dval, verbose=True )

    best_bdt = grid.best_estimator_
    print('Search done, best params: ', grid.best_params_)
    
    # Save model in TMVA format
    outpath = '/home/kpapad/UG_thesis/Thesis/Bdt/out/Models/'
    modelname = sys.argv[2][13:-5]
    modelFile = outpath +"myModel{}_conf{}.root".format(dataset, modelname)
    print('Training, done')
    print('Saving model at {}'.format(modelFile))
    ROOT.TMVA.Experimental.SaveXGBoost(best_bdt,"myBDT", modelFile ,  num_inputs=num_all)
    print('done')
    
    
