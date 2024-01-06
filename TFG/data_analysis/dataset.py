from functools import reduce
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import itertools
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math
from statistics import mean



############################# FUNCTIONS FOR DATA CLEANING ##################################

def change_col_names(df_list, var_names):
    """
    Input:

    df_list (list of DataFrames): List of DataFrames to change column names.
    var_names (list): List of new variable names.

    Description: Changes column names in each DataFrame in the input list based on the provided variable names.

    Output: None.
    """

    for df, var_name in zip(df_list, var_names):
        new_columns = [f"{var_name}_{col}" if col not in ['stage', 'ID'] else col for col in df.columns]  # new col names
        df.columns = new_columns  



def merge_data(df_list_updated):
    """
    Input:

    df_list_updated (list): List of DataFrames to be merged.

    Description: Merges DataFrames in the input list on 'ID' and 'stage'.

    Output: Returns a merged DataFrame.
    """

    merged_df = reduce(lambda left, right: pd.merge(left, right, on=['ID', 'stage'], how='inner'), df_list_updated)

    if merged_df.isnull().values.any():
        print("Dataframe contains nulls.")
    else:
        print("Dataframe without nulls!!! You did amazinggggg.")

    return merged_df



def mean_by_group(df_list):
    """
    Input:

    df_list (list of DataFrames): List of DataFrames.

    Description: Calculates the mean for each group (ID and stage) in each DataFrame in the input list.

    Output: Returns a list of new DataFrames where means are calculated for each group.
    """

    new_dfs = []

    for df in df_list:
        new_dfs.append(df.groupby(['ID', 'stage'], as_index=False).mean())

    return new_dfs



def norm(merged_df):
    """
    Input:

    merged_df (DataFrame): DataFrame to be normalized.

    Description: Normalizes numerical columns in the input DataFrame using Min-Max scaling.

    Output: Returns a normalized DataFrame.
    """

    min_max_scaler = MinMaxScaler()
    norm_merged_df = pd.DataFrame(min_max_scaler.fit_transform(merged_df.iloc[:, 1:]),
                                  columns=merged_df.iloc[:, 1:].columns)
    norm_merged_df['ID'] = merged_df['ID']
    print("Normalization done :D")

    return norm_merged_df



def dataset_cor(norm_merged_df, best_corr):
    """
    Input:
    norm_merged_df (DataFrame): The input DataFrame containing normalized data.
    best_corr (list): List of selected columns to be retained in the dataset.

    Description: Returns a new DataFrame with only the columns specified in best_corr.

    Output: A subset of norm_merged_df containing only the columns specified in best_corr.
    """
    
    dataset_cor = norm_merged_df.copy()

    for i in norm_merged_df:
        if i not in best_corr:
            dataset_cor = dataset_cor.drop(i, axis = 1)

    return dataset_cor


def new_patients(merged_df):
    """
    Input:
    merged_df (DataFrame): The input DataFrame containing patient data.

    Description: Generates new patient data by averaging existing rows and adding standard deviations.

    Output: A DataFrame containing the original and generated patient data.
    """

    dfs_por_stage = {}

    grupos = merged_df.groupby('stage')

    for stage, grupo_df in grupos:
        dfs_por_stage[stage] = grupo_df

    df_stage_1 = dfs_por_stage.get(1)
    df_stage_2 = dfs_por_stage.get(2)
    df_stage_3 = dfs_por_stage.get(3)

    dfs = [df_stage_1, df_stage_2, df_stage_3]

    stage = 1

    for df in dfs:
        new_df = []
        last_num = int((merged_df['ID'].iloc[-1])[-2:])

        for i in range(0, len(df), 2):
            if i + 1 < len(df):
                rows_to_average = df.iloc[i:i+2]
                media = rows_to_average.drop(['ID', 'stage'], axis=1).mean()
                new_row = {'ID': f'PYROS{(last_num) + 1}', 'stage': stage, **media}
                missing_df = pd.DataFrame([new_row])
                new_df.append(missing_df)
                last_num += 1

        for i in range(0, len(df), 3):
            if i + 1 < len(df):
                rows_to_average = df.iloc[i:i+3]
                media = rows_to_average.drop(['ID', 'stage'], axis=1).mean()
                new_row = {'ID': f'PYROS{(last_num) + 1}', 'stage': stage, **media}
                missing_df = pd.DataFrame([new_row])
                new_df.append(missing_df)
                last_num += 1

        std_dev = df.drop(['ID', 'stage'], axis=1).std()
        print(std_dev)

        for i in range(len(df)):
            row = df.iloc[i]
            new_row_values = row.drop(['ID', 'stage']) + std_dev
            new_row = {'ID': f'PYROS{(last_num) + 1}', 'stage': 1, **new_row_values}
            missing_df = pd.DataFrame([new_row])
            new_df.append(missing_df)
            last_num += 1


        df = pd.concat([df] + new_df, ignore_index=True)    
        return df


def predict_nulls(dataset, features, targets):
    """
    Input:

    dataset (DataFrame): The dataset containing missing values.
    features (list): List of feature columns used for prediction.
    targets (list): List of target columns with missing values to be predicted.

    Description: For each target column with missing values, the function uses linear regression to predict the missing values based on the provided features.

    Output: None.
    """

    for target in targets:
        df_ECGx_subset = dataset[dataset[target].notna()]

        # Selecct  (features) and (target)
        X = df_ECGx_subset[features]
        y = df_ECGx_subset[target]

        # Divide in train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        modelo_regresion = LinearRegression()

        modelo_regresion.fit(X_train, y_train)

        predicciones = modelo_regresion.predict(X_test)

        mse = mean_squared_error(y_test, predicciones)
        # print(f'Mean Squared Error: {mse}')

        df_ECGx_subset = dataset[dataset[target].isna()]
        predicciones_nuevas = modelo_regresion.predict(df_ECGx_subset[features])

        indices_filas_con_nans = dataset[dataset[target].isna()].index

        dataset.loc[indices_filas_con_nans, target] = predicciones_nuevas
        dataset[target]
        features.append(target)



def null_for_mean_PAT(df_PAT):
    """
    Input:

    df_PAT (DataFrame): DataFrame containing 'mPAT' and 'stdPAT' columns with missing values.

    Description: Fills missing values in 'mPAT' and 'stdPAT' columns by calculating means for each stage group.

    Output: None.
    """

    # Create three lists for groups stage 2 and 3
    grupos = df_PAT.groupby('stage')['mPAT'].apply(list)
    lista_stage_2_mPAT = grupos.get(2, [])
    lista_stage_2_mPAT = [x for x in lista_stage_2_mPAT if not math.isnan(x)]
    lista_stage_3_mPAT = grupos.get(3, [])
    lista_stage_3_mPAT = [x for x in lista_stage_3_mPAT if not math.isnan(x)]

    grupos = df_PAT.groupby('stage')['stdPAT'].apply(list)
    lista_stage_2_stdPAT = grupos.get(2, [])
    lista_stage_2_stdPAT = [x for x in lista_stage_2_stdPAT if not math.isnan(x)]

    lista_stage_3_stdPAT = grupos.get(3, [])
    lista_stage_3_stdPAT = [x for x in lista_stage_3_stdPAT if not math.isnan(x)]

    mean_2_mPAT = mean(lista_stage_2_mPAT)
    mean_3_mPAT = mean(lista_stage_3_mPAT)
    mean_2_sPAT = mean(lista_stage_2_stdPAT)
    mean_3_sPAT = mean(lista_stage_3_stdPAT)

    for index, row in df_PAT.iterrows():
        stage = row['stage']
        if pd.isna(row['mPAT']):
            if stage == 2:
                df_PAT.loc[index, 'mPAT'] = mean_2_mPAT
            else:
                df_PAT.loc[index, 'mPAT'] = mean_3_mPAT
        if pd.isna(row['stdPAT']):
            if stage == 3:
                df_PAT.loc[index, 'stdPAT'] = mean_3_sPAT
            else:
                df_PAT.loc[index, 'stdPAT'] = mean_2_sPAT



def add_missing_rows(df_list):
    """
    Input:

    df_list (list of DataFrames): List of DataFrames containing data with potential missing rows.

    Description: Adds missing rows to each DataFrame in the input list based on missing combinations of 'ID' and 'stage'.
                 If missing rows are added, the function calculates the mean value for each measure column for the corresponding
                 stage and fills the missing values with it.

    Output: List of DataFrames with added missing rows.
    """

    df_list_updated = []

    for df in df_list:
        # Iterate over each stage (1, 2, 3)
        for state in [1, 2, 3]:
            # Filter the DataFrame by stage
            state_df = df[df['stage'] == state]
            # Get unique IDs in the original DataFrame
            all_ids = df['ID'].unique()
            # Get unique IDs in the filtered DataFrame
            state_ids = state_df['ID'].unique()
            # Create all possible combinations of stage and ID
            all_combinations = list(itertools.product([state], all_ids))
            # Find the combinations that are not present in the filtered DataFrame
            missing_combinations = set(all_combinations) - set(zip([state] * len(state_ids), state_ids))

            # Create a list of DataFrames for the missing combinations
            missing_dfs = []
            for missing_combination in missing_combinations:
                stage, missing_id = missing_combination
                new_row = {'ID': missing_id, 'stage': stage}

                # Calculate the means of other columns and add them to the row
                for col in df.columns:
                    if col not in ['ID', 'stage']:
                        new_row[col] = df[(df['stage'] == stage)][col].mean()

                # Convert the new row to a DataFrame
                missing_df = pd.DataFrame([new_row])

                # Append the missing DataFrame to the list
                missing_dfs.append(missing_df)

            # Concatenate the original DataFrame and the missing DataFrames
            df = pd.concat([df] + missing_dfs, ignore_index=True)

        df_list_updated.append(df)

    return df_list_updated










