################################## IMPORTS ##################################

from pymongo import MongoClient
import pandas as pd

from database.mongo import *
from data_analysis.analysis import *
from data_analysis.dataset import *
import warnings

# Ignore all FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)



###################################### CONNECTION ##############################################

# Database
database_name = "TFG"

# Remote execution
host = 'localhost'
port = 27017

db, connection = connect_db(database_name, host, port)

############################# READING DATA  ##################################

# List with all excel files
excel_files =   [
                "ECGx_60s_21Nov2023_15_37.xlsx",
                "ECGy_60s_21Nov2023_15_38.xlsx",
                "EDA_60s_21Nov2023_16_32.xlsx",
                "PAT_60s_21Nov2023_17_42.xlsx",
                "PRV_60s_21Nov2023_17_41.xlsx",
                "PYROS_Salivettes.xls",
                "RESP_60s_21Nov2023_15_38.xlsx",
                "TEMP_60s_23Nov2023_00_55.xlsx",
                "Pacients.xlsx",
                "Professionals.xlsx",
                "sessio.xlsx",
                "Tests.xlsx",
                ]           

base_path = "C:/Users/onasa/Documents/3r2n_semestre/BDnR/Pycharm/TFG/Data/"

# Dictionary to store DataFrames
dfs = {}

# Read Excel files into DataFrames
for file in excel_files:
    file_path = f"{base_path}{file}"
    df_name = f"df_{file.split('.')[0]}"
    dfs[df_name] = pd.read_excel(file_path)

# Access DataFrames using the original variable names
df_ECGx = dfs["df_ECGx_60s_21Nov2023_15_37"]
df_ECGy = dfs["df_ECGy_60s_21Nov2023_15_38"]
df_EDA = dfs["df_EDA_60s_21Nov2023_16_32"]
df_PAT = dfs["df_PAT_60s_21Nov2023_17_42"]
df_PRV = dfs["df_PRV_60s_21Nov2023_17_41"]
df_RESP = dfs["df_RESP_60s_21Nov2023_15_38"]
df_TEMP = dfs["df_TEMP_60s_23Nov2023_00_55"]
df_saliv = dfs["df_PYROS_Salivettes"]
df_pacients = dfs["df_Pacients"]
df_professionals = dfs["df_Professionals"]
df_sessio = dfs["df_sessio"]
df_test = dfs["df_Tests"]

for df in [df_ECGx, df_ECGy, df_EDA, df_PAT, df_PRV, df_RESP, df_TEMP, df_saliv]:
    df['stage'].replace({'BASAL_COGNITIVE_TASK': 1, 'BASAL': 1, 'BASELINE':1, 'STRESS_COGNITIVE_TASK': 2, 'STRESS':2, 'CONTROL_COGNITIVE_TASK':3, 'CONTROL':3}, inplace=True)
    df.sort_values(by=["ID", "stage"], inplace=True)

measures = [df_ECGx, df_ECGy, df_EDA, df_PAT, df_PRV, df_RESP, df_TEMP, df_saliv]
names = ["ECGx", "ECGy", "EDA", "PAT", "PRV", "RESP", "TEMP", "saliv"]

############################# TRANSFER DATA WITH MONGODB ################################## 

# ---------------------------- TRANSFER DATA USUARI --------------------------- 

collection_users = users_collection(db, df_pacients, df_professionals)

# ---------------------------- TRANSFER DATA SESSIO --------------------------- 

collection_sessio, sessio_data = session_collection(db, collection_users, df_sessio)

# ---------------------------- TRANSFER DATA TEST PSICOMÈTRIC -----------------

collection_psico = test_collection(db, df_test, sessio_data)

# ---------------------------- TRANSFER DATA SENYAL --------------------------- 

collection_senyal = signal_collection(db, measures, names)

# ---------------------------- TRANSFER DATA MESURA ---------------------------

collection_mesura = measure_collection(db, measures, names)


###################################### INITIAL DATA PROCESSING #######################################

# --------------- Columns and Nulls Handling for ECGx ---------------

df_ECGx.drop(columns='ECGname', inplace=True)
df_ECGx.drop(columns='overlap', inplace=True)

# We fill nulls with predicted values
features1 = []
for i in df_ECGx.columns:
  if(i != 'ID' and i != 'rPHF'and i != 'rPLF' and i != 'rPLFn' and i != 'rPHFn' and i != 'rLF_HF' and i!= 'Fr' and i != 'Pk'):
    features1.append(i)

features2 = []
for i in df_ECGx.columns:
  if(i != 'ID' and i != 'rLF_HF' and i!= 'Fr' and i != 'Pk'):
    features2.append(i)

predict_nulls(df_ECGx, features1, targets = ['rPHF','rPLF','rPLFn','rPHFn','rLF_HF'])
predict_nulls(df_ECGx, features2, ['Fr', 'Pk'])


# --------------- Columns and Nulls Handling for ECGy ---------------

df_ECGy.drop(columns='ECGname', inplace=True)
df_ECGy.drop(columns='overlap', inplace=True)

# We fill nulls with predicted values
targets = ['rPHF','rPLF','rPLFn','rPHFn','rLF_HF']
features1 = [feature for feature in features1 if feature not in targets]
predict_nulls(df_ECGy, features1, targets)

targets = ['Fr', 'Pk']
features2 = [feature for feature in features2 if feature not in targets]
predict_nulls(df_ECGy, features2, ['Fr', 'Pk'])


# --------------- Nulls Handling for PAT ---------------

# We fill nulls with the mean --> not enough data to predict
null_for_mean_PAT(df_PAT)


# --------------- Columns and Nulls Handling for PRV ---------------

df_PRV.drop(columns='PRVoverlap', inplace=True)

# We fill nulls with the predicted values
targets = ['PRVrPHF','PRVrPLFn','PRVrPHFn','PRVrLF_HF','PRVrPLf', 'Fr', 'Pk']
features_prv = [feature for feature in df_PRV.columns if feature not in targets and feature != 'ID']
predict_nulls(df_PRV, features_prv, targets)


# --------------- Columns Handling for TEMP ---------------

df_TEMP.drop(columns='session', inplace=True)


################################# DATASET UNIFICATION ##################################

df_list = [df_ECGx, df_ECGy, df_EDA, df_PAT, df_PRV, df_RESP, df_TEMP, df_test, df_saliv]
var_names = ["ECGx", "ECGy", "EDA", "PAT", "PRV", "RESP", "TEMP", "test", "saliv"]

# Calculate the mean of the columns grouped by 'ID' and 'stage' -> obtain 1 value per variable in each state (by taking the mean)
df_list = mean_by_group(df_list)

for df in df_list:
    if "onset" in df.columns:
        df.drop(columns='onset', inplace=True)
    if "offset" in df.columns:
        df.drop(columns='offset', inplace=True)
    if "length" in df.columns:
        df.drop(columns='length', inplace=True)

# Changing column names from DataFrames
change_col_names(df_list, var_names)

# For each session there must be stages 1, 2 or 3. If one misses, we add the corresponding row with the values of the mean of the stage
df_list_updated = add_missing_rows(df_list)

for df in df_list_updated:
    df.sort_values(by=["ID", "stage"], inplace=True)

merged_df = merge_data(df_list_updated)


######################################### NORMALIZATION ##########################################

norm_merged_df = norm(merged_df)


######################################### PCA ####################################################

# Separate independent variables (X) from the dependent variable (y)
X = norm_merged_df.drop('test_VASS', axis=1)  # Drop the 'VASS' column to get independent variables
y = norm_merged_df['test_VASS']

# Apply PCA
Pca_application(X.select_dtypes(include=['float64', 'int64']), y, False, 0.8) # Change bool to True for plots



######################################### CORRELATIONS ###########################################

# Correlations between all variables
correlations = norm_merged_df.select_dtypes(include=['float64', 'int64']).corr()

# Search the most correlated variables with target (more than 0.25)
best_corr, reduced_corr = best_corr_target(correlations, target = 'test_VASS')

plot_heatmap(correlations, best_corr, reduced_corr, all = False, reduced = False, best = False)

best_corr.remove('ECGy_rPHF')
best_corr.remove('TEMP_Min')
best_corr.remove('TEMP_Med')

data_corr = dataset_cor(norm_merged_df, best_corr)

print(data_corr.corr()['test_VASS'])

plot_pairplot(data_corr, show = False)

# --------------- Try log transformation ---------------

log_df = merged_df.select_dtypes(include=['float64', 'int64']).corr().applymap(lambda x: np.log(x + 10))
log_correlations = log_df.select_dtypes(include=['float64', 'int64']).corr()
log_best_corr, log_reduced_corr = best_corr_target(log_correlations, target = 'test_VASS', min_correlation = 0.5)
log_data_corr = dataset_cor(log_df, log_best_corr)

plot_pairplot(log_data_corr, show = False)



# --------------- plots 3d para ver relaciones ---------------

plot_3d_rels(data_corr, best_corr, show = False)


######################################### MODELS NO DATA AUGMENTATION ############################

best_corr.remove('test_VASS')

param_sets = {
    'RF': {'n_estimators': [50, 100, 150, 200, 250], 'max_depth': [10, 15]},
    'GRADBOOST': {'n_estimators': [50, 100, 150, 200, 250], 'learning_rate': [0.1, 0.5], 'max_depth': [3, 5]},
    'SVR': {'C': [0.01, 0.1, 1, 10, 100], 'epsilon': [0.01, 0.1, 0.2, 0.5, 1]}
}

regressors = {
    'LINEAR': perform_regression(norm_merged_df, best_corr, 'LINEAR', show_process=False, plot_pred = False),
    'LASSO': perform_regression(norm_merged_df, best_corr, 'LASSO', show_process=False, plot_pred = False),
    'BAYES': perform_regression(norm_merged_df, best_corr, 'BAYES', show_process=False, plot_pred = False),
    'RF': best_params_rf(param_sets['RF']['n_estimators'], param_sets['RF']['max_depth'], norm_merged_df, best_corr, False),
    'GRADBOOST': best_params_gb(param_sets['GRADBOOST']['n_estimators'], param_sets['GRADBOOST']['learning_rate'], param_sets['GRADBOOST']['max_depth'], norm_merged_df, best_corr, False),
    'SVR': best_params_svr(param_sets['SVR']['C'], param_sets['SVR']['epsilon'], norm_merged_df, best_corr, False)
}

result_dicts = {'r2_list': {}, 'mse_list': {}, 'acc_list': {}}

for regressor in regressors:
    r2_max, error_min, acc_max = regressors[regressor]
    result_dicts['r2_list'][regressor] = r2_max
    result_dicts['mse_list'][regressor] = error_min
    result_dicts['acc_list'][regressor] = acc_max

plot_metrics(result_dicts)

######################################### WITH DATA AUGMENTATION #####################################

merged_df = new_patients(merged_df)
norm_merged_df = norm(merged_df)

param_sets = {
    'RF': {'n_estimators': [50, 100, 150, 200, 250], 'max_depth': [10, 15]},
    'GRADBOOST': {'n_estimators': [50, 100, 150, 200, 250], 'learning_rate': [0.1, 0.5], 'max_depth': [3, 5]},
    'SVR': {'C': [0.01, 0.1, 1, 10, 100], 'epsilon': [0.01, 0.1, 0.2, 0.5, 1]}
}

regressors = {
    'LINEAR': perform_regression(norm_merged_df, best_corr, 'LINEAR', show_process=False, plot_pred = False),
    'LASSO': perform_regression(norm_merged_df, best_corr, 'LASSO', show_process=False, plot_pred = False),
    'BAYES': perform_regression(norm_merged_df, best_corr, 'BAYES', show_process=False, plot_pred = False),
    'RF': best_params_rf(param_sets['RF']['n_estimators'], param_sets['RF']['max_depth'], norm_merged_df, best_corr, False),
    'GRADBOOST': best_params_gb(param_sets['GRADBOOST']['n_estimators'], param_sets['GRADBOOST']['learning_rate'], param_sets['GRADBOOST']['max_depth'], norm_merged_df, best_corr, False),
    'SVR': best_params_svr(param_sets['SVR']['C'], param_sets['SVR']['epsilon'], norm_merged_df, best_corr, False)
}

result_dicts_augment = {'r2_list': {}, 'mse_list': {}, 'acc_list': {}}

for regressor in regressors:
    r2_max_aug, error_min_aug, acc_max_aug = regressors[regressor]
    result_dicts_augment['r2_list'][regressor] = r2_max_aug
    result_dicts_augment['mse_list'][regressor] = error_min_aug
    result_dicts_augment['acc_list'][regressor] = acc_max_aug

plot_metrics(result_dicts_augment)


###################################### CLOSE CONNECTION ##########################################

connection.close()