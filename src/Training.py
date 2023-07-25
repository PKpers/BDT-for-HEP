#!/usr/bin/python
import ROOT
from pathlib import Path

def pair_permute(__list__):
    '''
    combine the items of a list
    in pairs of two.
    order doesn't matter
    repetition allowed
    '''
    permutations = []
    for i, xi in enumerate(__list__):
        if type(xi) is str:
            p = [xi+xj for xj in __list__[i:]]
        #
        permutations += p
    #
    return permutations

def make_variable_names(signal_filename, background_filename):
    '''
    Generates a list of variable names based on the given signal and background filenames.
    Args:
        signal_filename (str): The name of the signal file.
        background_filename (str): The name of the background file.
    '''
    is_pxyz_signal = "Pxyz" in signal_filename
    is_pxyz_background = "Pxyz" in background_filename

    if is_pxyz_signal and is_pxyz_background:
        names = ("Px", "Py", "Pz")
    else:
        names = ("Pt", "Eta", "Phi")

    variables = [name + str(j) for j in range(1, 3) for name in names]

    is_perm_signal = 'Perm' in signal_filename
    is_perm_background = 'Perm' in background_filename
    is_deltas_signal = 'Deltas' in signal_filename
    is_deltas_background = 'Deltas' in background_filename

    if is_perm_signal and is_perm_background:
        variables = pair_permute(variables)
    elif is_deltas_signal and is_deltas_background:
        variables = ["Pt1", "Pt2", "DeltaPhi", "DeltaR", "DeltaEta"]

    return variables

def make_bar_graph(variables, feat_imp):
    from matplotlib import pyplot as plt 

    plt.bar(variables, feat_imp)
    plt.ylabel('Featrue importance', fontsize = 15)
    plt.tick_params(labelsize='large', width=3)
    plt.savefig("/home/kpapad/UG_thesis/Thesis/Bdt/out/Plots/feature_importance_hm.pdf")
    return

def load_data(signal_filename, background_filename):
    import numpy as np
    # Read data from ROOT files
    data_sig = ROOT.RDataFrame("tree", signal_filename).AsNumpy()
    data_bkg = ROOT.RDataFrame("tree", background_filename).AsNumpy()

    # create the variable names
    variables = make_variable_names(signal_filename, background_filename) 
    
    # Convert inputs to format readable by machine learning tools
    x_sig = np.vstack([data_sig[var] for var in variables]).T
    x_bkg = np.vstack([data_bkg[var] for var in variables]).T
    
    x = np.vstack([x_sig, x_bkg]).astype(np.float32)
    # Create labels
    # number of events. After transposing , we have an object of that form [ [], [], ..., [] ]
    # each sub array corresponds to the number of event: [ [1st event contents ], [2nd, ] ... ]
    # spape method, returns the lentgh of each dimention of the array.
    # In our case the array is 2 dimentional. To describe the position of an element in the array,
    # we need one index to specify the number of the event the the element is in
    # and another one to specify its position inside the event(eg the jth element of the ith event)
    # so x_sig.shape is a tuple whose elements is number of events and number of elements in each event
    # from that tuple we take the 0th element which is the number of events 
    
    num_sig = x_sig.shape[0]#same as len(np.array) returns error
    num_bkg = x_bkg.shape[0]
    y = np.hstack([np.ones(num_sig), np.zeros(num_bkg)]).astype(np.float32)

   # ones(n): return an array of shape n filled with 1
   # zeros(n): return an array of shape n filled with 0
   # [np.array(), np.array()] -hstack-> np.array[contents of the two arrays merged]
   
    # Compute weights balancing both classes
    num_all = num_sig + num_bkg
    w = np.hstack([np.ones(num_sig) * num_all / num_sig, np.ones(num_bkg) * num_all / num_bkg])\
          .astype(np.float32)
    # asign the same weight in all sig events and the same in all bkg events. wsig != wbkg.
    # np.ones it is used to create an array of diemntion = num_bkg
    return x, y, w, num_all

def get_project_root() -> Path:
    return Path(__file__).parent.parent

def load_config(config_dir, config_fname):
    conf_file = config_dir + config_fname
    with open(conf_file, 'r') as file:
        cfg = yaml.safe_load(file)
    #
    hyper_params = cfg['hyper_params']
    return hyper_params



    
 # ======================================================================= #
 # ========================== MAIN FUNCTION ================================= #
 # ======================================================================= #
if __name__ == "__main__":
    import sys
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split
    import yaml
    
    if len(sys.argv) != 5:
        print("Usage: {} {} {} {} {}" \
              .format(sys.argv[0], "/path/to/training/data/", "training dataset name", "training config name", "model name"))
        exit(-1)
    #

    base = get_project_root() # get the path to root 
    base = str(base)
    
    # Load the training data 
    dataset_dir = sys.argv[1]
    dataset = sys.argv[2]
    sig_filename = dataset_dir+ "{}_SIG_Train.root".format(dataset)
    bkg_filename = dataset_dir+ "{}_BKG_Train.root".format(dataset)
    x, y, w, num_all= load_data(sig_filename, bkg_filename)
    
    # Load training config
    cfg_fname = sys.argv[3]
    cfg_path = base + "/config/"
    training_config = load_config(cfg_path, cfg_fname)

    print("loading training configuration: {}".format(training_config))
    
    
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.20, random_state=0)
    
    # Create the list for validation set
    dval = [(X_val, y_val)]

    print('Training started with data from {} and {}'.format(sig_filename, bkg_filename)) 

    # Train the classifier
    bdt = XGBClassifier(**training_config)
    bdt.fit(X_train, y_train, eval_set=dval, verbose=True)
    
    # Save model in TMVA format
    outpath = base + "/out/Models/"
    modelname = sys.argv[4]
    modelname = modelname if ".root" in modelname else modelname + ".root" 
    modelFile = outpath + modelname

    print('Training, done')
    print('Saving model at {}'.format(modelFile))

    ROOT.TMVA.Experimental.SaveXGBoost(bdt,"myBDT", modelFile,  num_inputs=num_all)
    print('done')
    
