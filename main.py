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

st.title('Modeling App')

st.sidebar.subheader('Step1: Load data')
select_data = st.sidebar.radio('Select data', ('Use dummy data', 'Browse'))

@st.cache()
def use_dummy_data():
    return get_data('diamond', verbose=False)

if select_data == 'Use dummy data':
    dataset = use_dummy_data()
    #st.write('Using Diamond dataset...')
else: #select_data == 'Browse':
    file_buffer = st.sidebar.file_uploader('Browse data',type = ['txt','csv'])
    dataset = pd.DataFrame(file_buffer)


if st.sidebar.checkbox('Display data snapshot'):
    st.write(dataset.head())
'Data shape: ',dataset.shape

'''## Generate descriptive report of data'''
@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def gen_pandas_profile(dataset):
    #time.sleep(1)
    return ProfileReport(dataset)

if st.checkbox('Generate report'):
    st_profile_report(gen_pandas_profile(dataset))
    

st.sidebar.subheader('Step 2: Select test data fraction')
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


'''## Train test split'''
test_perc = st.sidebar.slider('testing data percentage:', 10, 33)#, format='%')
'test data percentage:', test_perc, '%'

train_data, test_data = split_train_test(dataset, test_perc = test_perc, random_state = 42)
'train_data shape:', train_data.shape, 'test_data shape:', test_data.shape

st.sidebar.subheader('Step 3: Model setup')
collist = tuple({col for col in train_data.columns})
target = st.sidebar.selectbox('target:', collist)
'Selected target variable:', target

experiment_id = st.sidebar.text_input('Enter experiment id', max_chars=3)
if not experiment_id:
    st.warning('Specify experiment id (use numbers)')
    st.stop()

if experiment_id is not None and experiment_id != '':
    if not isinstance(int(experiment_id), int):
        st.warning('Please enter a number')
        st.stop()

'experiment_id:', experiment_id

experiment_name = st.sidebar.text_input('Enter experiment name (optional)')
if experiment_name:
    st.write('experiment name: ', experiment_name)
else:
    experiment_name = None

if st.sidebar.checkbox('Show model setup documentation.'):
    st.text('---------------------------------')
    st.write('Refer to Parameters in below documentation \
              from PyCaret for setting up advance controls\
              for modeling  :', pr.setup.__doc__)

advance_setup = st.sidebar.checkbox('Show advance controls')

if advance_setup:
    normalize = st.sidebar.checkbox('Normalize (training data)')
    transformation = st.sidebar.checkbox('Transform (training data)')
    transform_target = st.sidebar.checkbox('Transform target')
    combine_rare_levels = st.sidebar.checkbox('Combine rare levels (for categorical data)')
    if combine_rare_levels:
        rare_level_threshold = st.sidebar.slider('Rare level threshold %', 2, 5)
        rare_level_threshold /= 100
    else:
        rare_level_threshold = None
    remove_multicollinearity = st.sidebar.checkbox('Remove multicollinearity')
    if remove_multicollinearity:
        multicollinearity_threshold = st.sidebar.slider('Multicollinearity threshold (R2)', 80, 99)
        multicollinearity_threshold /= 100
    else:
        multicollinearity_threshold = None
    bin_numeric_features = st.sidebar.checkbox('Bin numeric feature')
    if bin_numeric_features:
        bin_feats = list(collist)
        bin_feats.remove(target)
        selected_bin_feats = st.sidebar.multiselect('Select features', tuple(bin_feats))
    else:
        selected_bin_feats = [None]

if advance_setup:
    setup_dict = {'data': train_data,
                'target' : target, 
                'session_id': int(experiment_id), 
                'experiment_name':experiment_name,
                'normalize' : normalize,
                'transformation' : transformation,
                'transform_target' : transform_target, 
                'combine_rare_levels' : combine_rare_levels, 
                'rare_level_threshold' : rare_level_threshold,
                'remove_multicollinearity' : remove_multicollinearity,
                'multicollinearity_threshold' : multicollinearity_threshold,
                'bin_numeric_features' : list(selected_bin_feats) #['Carat Weight']
                }
else:
    setup_dict = {'data': train_data,
              'target' : target, 
             'session_id': int(experiment_id),
             'experiment_name': experiment_name,
             'normalize' : False,
             'transformation' : False,
             'transform_target' : False, 
             'combine_rare_levels' : False, 
             'rare_level_threshold' : 0.1,
             'remove_multicollinearity' : False,
             'multicollinearity_threshold' : 0.9,
             'bin_numeric_features' : None #['Carat Weight']
             }

#               'ID          Name      '
#               '--------    ----------'     
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
model_library_option = [m[1].strip() for m in model_library.items()]

setupready = False
model_radio = None
confirm = False

modeling_trial = st.radio('Modeling trial: ',('Create base model', 'Tune selected model'))
if modeling_trial=='Create base model':
    model_radio = st.checkbox('Build all available models')
    if model_radio:
        if st.button('Train models'):
            with st.spinner('Running setup.'):
                pr.setup(**setup_dict, silent = True, verbose=False)
            st.success('Setup completed succesfully.')
            with st.spinner('Training models...'):
                bestmodels = pr.compare_models(verbose=False)
            results = pr.pull()
            modellist = results.Model.tolist()
            st.write(results)

            # st.write('Training selected models:', model_to_build)
            # results = pr.compare_models(whitelist=model_to_build)
            # st.write(results)
    else:
        model_to_build = st.multiselect('Select models to build:', model_library_option)
        if model_to_build:
            if isinstance(model_to_build, list):
                model_to_build_ = [k for k,v in list(model_library.items()) if v in model_to_build]
                with st.spinner('Running setup.'):
                    pr.setup(**setup_dict, silent = True, verbose=False)
                st.success('Setup completed succesfully.')
                with st.spinner('Training models...'):
                    bestmodels = pr.compare_models(whitelist=model_to_build_,  verbose=False)
                results = pr.pull()
                modellist = results.Model.tolist()
                st.text('Training results:')
                st.write(results)
                showplot = st.selectbox('Show plot for :', modellist)
                if showplot:
                    if 'catboost' in str(type(bestmodels)):
                        pr.interpret_model(bestmodels)
                        st.pyplot()
                    else:
                        if isinstance(bestmodels, list):
                            clf = bestmodels[modellist.index(showplot)]
                        else:
                            clf = bestmodels
                        pr.plot_model(clf, verbose=False)
                        st.pyplot()

if modeling_trial == 'Tune selected model':
    model_to_tune = st.selectbox('Select model to tune:', model_library_option)
    if model_to_tune:
        with st.spinner('Creating base model:'):
            if isinstance(model_to_tune, str):
                model_to_tune_ = [k for k,v in list(model_library.items()) if v in model_to_tune]
                base_model = pr.create_model(model_to_tune_[0], verbose=False)
        with st.spinner('Tuning hyper-parameter of base model'):
            tm = pr.tune_model(base_model, verbose=False)
            results = pr.pull()
            if 'catboost' in str(type(tm)):
                pr.interpret_model(tm)
                st.pyplot()
            else:
                pr.plot_model(tm, verbose=False)
                st.pyplot()
            st.write(results)
            
