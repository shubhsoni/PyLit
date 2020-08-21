import pandas as pd
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import streamlit as st
import pycaret.regression as pr
import time
from pycaret.datasets import get_data
import io
import shap
st.set_option('deprecation.showfileUploaderEncoding', False)


def print_log():
    time.sleep(2)
    with open('logs.log','rb') as log:
        st.write(log.readlines())
        
@st.cache()
def use_dummy_data():
    '''get dummy data for modeling'''
    return get_data('diamond', verbose=False)

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def gen_pandas_profile(dataset):
    #time.sleep(1)
    return ProfileReport(dataset)

# split train test data
@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def split_train_test(data, test_perc = 33, random_state = 42):
    '''Split data into training and testing datasets... validation data will be used from train 
    data itself during modeling.
    keep target variable intact in the data, setup method will automaticaly treat
    feature and target variables separately by specifying target name.
    
    param data: (pandas df) dataset
    param test_perc: (float) percentage of data to be used for testing
    param random_state: (int) (optional) set seed for reproducible split
    returns: (tuple) train_data and test_data '''
    #time.sleep(1)
    train_perc = (100 - test_perc)/100
    train_data = data.sample(frac=train_perc, random_state=random_state).reset_index(drop=True)
    test_data = data.drop(train_data.index).reset_index(drop=True)

    # print('Data for Modeling: ' + str(train_data.shape))
    # print('Unseen Data For Predictions ' + str(test_data.shape))
    return train_data, test_data

#model library
#               'ID          Name      '
#               '--------    ----------'     
def model_library_():
    model_library = {'lr'       : 'Linear Regression',                   
                    'lasso'    : 'Lasso Regression'   ,             
                    'ridge'    : 'Ridge Regression '   ,            
                    'en'       : 'Elastic Net'          ,         
                    'lar'      : 'Least Angle Regression',                  
                    'llar'     : 'Lasso Least Angle Regression',                   
                    'omp'      : 'Orthogonal Matching Pursuit' ,                    
                    'br'       : 'Bayesian Ridge'              ,    
                    'ard'      : 'Automatic Relevance Determination',                  
                    'par'      : 'Passive Aggressive Regressor'     ,              
                    'ransac'   : 'Random Sample Consensus'      ,
                    'tr'       : 'TheilSen Regressor'            ,      
                    'huber'    : 'Huber Regressor'                 ,              
                    'kr'       : 'Kernel Ridge'                ,                    
                    'svm'      : 'Support Vector Machine'   ,                        
                    'knn'      : 'K Neighbors Regressor'  ,                        
                    'dt'       : 'Decision Tree'     ,                               
                    'rf'       : 'Random Forest'    ,                               
                    'et'       : 'Extra Trees Regressor' ,                           
                    'ada'      : 'AdaBoost Regressor' ,                            
                    'gbr'      : 'Gradient Boosting Regressor' ,                              
                    'mlp'      : 'Multi Level Perceptron',                         
                    'xgboost'  : 'Extreme Gradient Boosting' ,                  
                    'lightgbm' : 'Light Gradient Boosting' ,                   
                    'catboost' : 'CatBoost Regressor' , 
                    }
    return model_library

# Plot            Name                             
# ------          ---------                       
def plot_library_():
    plot_library =  {'residuals'    :'Residuals Plot',
                    'error'         :'Prediction Error Plot',
                    'cooks'         :'Cooks Distance Plot'   ,                      
                    'rfe'           :'Recursive Feat. Selection',                     
                    'learning'      :'Learning Curve' ,                          
                    'vc'            :'Validation Curve',                               
                    'manifold'      :'Manifold Learning',                        
                    'feature'       :'Feature Importance',                        
                    'parameter'     :'Model Hyperparameter'
    }
    return plot_library

@st.cache
def run_setup(**kwargs):
    pr.setup(**kwargs, silent = True, verbose=False)

@st.cache(allow_output_mutation=True)
def compare_model_fn(whitelist=None):
    bestmodels = pr.compare_models(whitelist=whitelist, verbose=False)
    return bestmodels

@st.cache
def base_model_fn(clf):
    base_model = pr.create_model(estimator=clf, verbose=False)
    return base_model

@st.cache
def tune_model_fn(base_model):
    tm = pr.tune_model(base_model, verbose=False)
    return tm

@st.cache
def interpret_model_fn(modellist=None,bestmodels=None, plottype='residuals',feature=None):
    if 'catboost' in str(type(bestmodels)):
        return pr.interpret_model(bestmodels, feature=feat)
    else:
        if isinstance(bestmodels, list):
            clf = bestmodels[modellist.index(showplot)]
        else:
            clf = bestmodels
        return pr.plot_model(clf, plot=plottype, verbose=False)
