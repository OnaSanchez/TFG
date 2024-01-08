import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LinearRegression, Lasso, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn import model_selection
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import random
from itertools import combinations
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import uuid
import os


############################# FUNCTIONS INITIAL DATA PROCESSING ##################################

def Pca_application(X, y, plot=True, target_explained_variance = 0.8):
    """
    Input:
        X (DataFrame): Input data matrix or DataFrame.
        y (DataFrame): Target variable.
        plot (bool, optional): Whether to plot the results.
        target_explained_variance (float, optional): Target cumulative explained variance ratio.

    Description: Apply Principal Component Analysis (PCA) to the input data and visualize the results.

    Output: None
    """

    pca = PCA()
    X_reduced = pca.fit_transform(X)

    #define cross validation method
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

    regr = LinearRegression()
    mse = []

    # Calculate MSE with only the intercept
    score = -1*model_selection.cross_val_score(regr,
            np.ones((len(X_reduced),1)), y, cv=cv,
            scoring='neg_mean_squared_error').mean()    
    mse.append(score)

    # Calculate MSE using cross-validation, adding one component at a time
    for i in np.arange(1, 16):
        score = -1*model_selection.cross_val_score(regr,
                X_reduced[:,:i], y, cv=cv, scoring='neg_mean_squared_error').mean()
        mse.append(score)
        
    # Visualize the explained variance by each principal component
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_explained_variance = explained_variance_ratio.cumsum()

    # Choose the number of components that explain var% of the variance
    num_components = len(cumulative_explained_variance[cumulative_explained_variance < target_explained_variance]) + 1

    if plot:
        # Plot the explained variance
        plt.plot(range(len(explained_variance_ratio)), explained_variance_ratio, marker='o')
        plt.xlabel('Principal Component Index')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Explained Variance for Each Principal Component')
        plt.show()

        # Plot the cumulative variance
        plt.plot(cumulative_explained_variance)
        plt.axvline(x=num_components, color='r', linestyle='--', label=f'{num_components} Components')
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Cumulative Explained Variance by Principal Components')
        plt.show()

        # Plot cross-validation results    
        plt.plot(mse)
        plt.xlabel('Number of Principal Components')
        plt.ylabel('MSE')
        plt.title('test_VASS')
        plt.show()

    print("Number of components chosen:", num_components)

    

def best_corr_target(correlation, target, min_correlation = 0.25):
    """
    Input:
        correlation (DataFrame): The correlation matrix.
        target (str): The target variable for which correlations are evaluated.
        min_correlation (float, optional): The minimum absolute correlation coefficient to consider.

    Description:
        Identify variables that have a correlation coefficient greater than the specified threshold with the target variable.
        This function extracts variable names that meet the correlation criteria.

    Output:
    - names_cor: A list of variable names with correlation coefficients greater than the threshold.
    - reduced_cor: A list of variable names with correlation coefficients greater than 0.45
                            (excluding those with a correlation coefficient of 1).
    """

    l = []
    names_cor = [target]
    l_red = []
    reduced_cor = [target]

    for i in range(len(correlation[target])):
        if abs(correlation[target][i]) > 0.15 and correlation[target][i] != 1:
            l_red.append(i)
            reduced_cor.append(list(correlation.keys())[i])

        if abs(correlation[target][i]) > min_correlation and correlation[target][i] != 1:
            l.append(i)
            names_cor.append(list(correlation.keys())[i])

    print(correlation[target][l])    
    
    return names_cor, reduced_cor



############################# FUNCTIONS DATA MODELING ##################################


def split_data(x, y, train_ratio=0.8):

    """
    Input:
        x (numpy.ndarray): The feature matrix.
        y (numpy.ndarray): The target variable.
        train_ratio (float, optional): The ratio of data to be used for training.

    Description: Split the input data into training and validation sets based on the specified ratio.

    Output:
        - x_train (numpy.ndarray): Feature matrix for training.
        - y_train (numpy.ndarray): Target variable for training.
        - x_val (numpy.ndarray): Feature matrix for validation.
        - y_val (numpy.ndarray): Target variable for validation.
    """
     
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    n_train = int(np.floor(x.shape[0]*train_ratio))
    indices_train = indices[:n_train]
    indices_val = indices[n_train:] 
    x_train = x[indices_train, :]
    y_train = y[indices_train]
    x_val = x[indices_val, :]
    y_val = y[indices_val]
    return x_train, y_train, x_val, y_val



def mse(y_true, y_pred):
    """
    Input:
        y_true (array-like): The true values of the target variable.
        y_pred (array-like): The predicted values of the target variable.

    Description:
        Calculate the Mean Squared Error (MSE) between the true and predicted values.

    Output:
        The MSE
    """
    
    return mean_squared_error(y_true, y_pred)



def regression(x, y):
    """
    Input:
        x (array-like): The feature matrix.
        y (array-like): The target variable.

    Description: Train a linear regression model.

    Output:
        LinearRegression: The trained linear regression model.
    """

    regr = LinearRegression()
    regr.fit(x, y)
    return regr



def lasso(x, y, a = 0.1):
    """
    Input:
        x (array-like): The feature matrix.
        y (array-like): The target variable.
        a (float): The regularization strength (alpha) for Lasso regression.

    Description: Train a lasso regression model.

    Output:
        LassoRegression: The trained lasoo regression model.
    """
    regr = Lasso(alpha=a)
    regr.fit(x, y)
    return regr



def Bayes(x, y, t = 1e-6):
    """
    Input:
        x (array-like): The feature matrix.
        y (array-like): The target variable.
        t (float): Tolerance for stopping criterion. Default is 1e-6.

    Description:
        Train a Bayesian Ridge regression model.

    Output:
        BayesianRidge: The trained Bayesian Ridge regression model.
    """
    regr = BayesianRidge(tol= t)
    regr.fit(x, y)
    return regr



def combination(A, n_conj):
    """
    Input:
        A (list): The input iterable.
        n_conj (int): The size of each combination.

    Description: Generate combinations of size n_conj from the elements in A.

    Output:
        A list containing all combinations of size n_conj.
    """
    temp = combinations(A, n_conj)
    aux = []
    for i in temp:
        aux.append(list(i))
    return aux



def perform_regression(norm_merged_df, best_corr, regressor, show_process = False, plot_pred = True):
    """
    Input:
        norm_merged_df (DataFrame): The normalized and merged DataFrame containing patient data.
        best_corr (list): A list of selected features for regression.
        regressor (str): The type of regression model ('LINEAR', 'LASSO', 'BAYES').
        show_process (bool): Whether to print intermediate results during the process. Default is False.
        plot_pred (bool): Whether to plot predictions. Default is True.

    Description:
        Perform regression using the specified features and regression model. Evaluate performance metrics such as
        Mean Squared Error (MSE), R2 score, and a custom accuracy metric.

    Output:
        - r2_max: Maximum R2 score achieved.
        - error_min: Minimum Mean Squared Error (MSE) achieved.
        - acc_max: Maximum Custom Accuracy achieved.
    """

    y = np.array(norm_merged_df['test_VASS'])
    rep = 1000

    error_min = np.inf
    pos_e = 0
    r2_max = 0
    pos_r = 0
    acc_max = 0

    l =  combination(best_corr,1) + combination(best_corr,2) + combination(best_corr,3) + combination(best_corr,4) + combination(best_corr,5) + [best_corr] 

    for j in l:
        dt_std_cp = norm_merged_df.copy()

        for i in dt_std_cp:
            if i not in j:
                dt_std_cp = dt_std_cp.drop(i, axis = 1)

        x = dt_std_cp.to_numpy()
        error = 0
        r2 = 0
        inter = 0
        acc = 0
        coef = [0 for k in range(len(j))]
        

        for k in range(rep):
            x_train, y_train, x_val, y_val = split_data(x, y)

            if regressor == 'LINEAR':
                regr = regression(x_train, y_train)
            if regressor == 'LASSO':
                regr = lasso(x_train, y_train)
            if regressor == 'BAYES':
                regr = Bayes(x_train, y_train)

            error += mse(np.exp(y_val), np.exp(regr.predict(x_val)))
            r2 += r2_score(y_val, regr.predict(x_val))
            coef += regr.coef_
            inter += regr.intercept_
            # Calculate the custom metric: percentage of predictions within a threshold
            acc += custom_accuracy(y_val, regr.predict(x_val))
            
        if show_process:
            print(j)
            print("Mean squeared error: ", error/rep)
            print("R2 score: ", r2/rep)
            print("Coef: ", coef/rep)
            print("Intercept: ", inter/rep)
            print("Custom Accuracy: ", acc/rep)
            print("--------------------------------------")
        
        if r2/rep > r2_max:
            r2_max = r2/rep
            pos_r = j
        if error/rep < error_min:
            error_min = error/rep
            pos_e = j
        if acc/rep > acc_max:
            acc_max = acc/rep
            pos_acc = j

        if plot_pred: 
            plot_predictions(y_val, regr.predict(x_val), model_line=regr.predict(x_val), title=f"Predicciones para {j}")

    print("Pel model de regressió: ", regressor)
    print("MILLOR R2 SCORE DE TOTS: ", pos_r, r2_max)  
    print("MILLOR MSE DE TOTS: ", pos_e, error_min)
    print("Mejor Custom Accuracy de todos: ", pos_acc, acc_max)
        
    return r2_max, error_min, acc_max



def custom_accuracy(real_values, predicted_values):
    """
    Input:
        real_values (array-like): The true values of the target variable.
            The actual target values.
        predicted_values (array-like): The predicted values of the target variable.
            The values predicted by a model.

    Description: Calculate custom accuracy as the percentage accuracy based on Mean Absolute Error (MAE).

    Output: The custom accuracy as a percentage.
    """

    mae = mean_absolute_error(real_values, predicted_values)
    percentage_accuracy = 100 - (mae / real_values.mean()) * 100
    return percentage_accuracy



def perform_regression_2(norm_merged_df, best_corr, show_process = False, estimator = 100, depth=10, rate = 0.1, c = 1.0, epsi = 0.2, not_opti = True, regressor = 'RF'):
    """
    Input:
        norm_merged_df (DataFrame): The normalized and merged DataFrame containing patient data.
        best_corr (list): A list of selected features for regression.
        show_process (bool): Whether to print intermediate results during the process. 
        estimator (int): Number of estimators (trees) for ensemble models. 
        depth (int): Maximum depth of the tree for ensemble models. 
        rate (float): Learning rate for Gradient Boosting. 
        c (float): Regularization parameter for Support Vector Regression. 
        epsi (float): Epsilon parameter for Support Vector Regression.
        not_opti (bool): Whether to print the best results.
        regressor (str): The type of regression model ('RF', 'GRADBOOST', 'SVR'). 

    Description:
        Perform regression using the specified features and regression model. Evaluate performance metrics such as
        Mean Squared Error (MSE), R2 score, and a custom accuracy metric for different feature combinations.

    Output:
        - r2_max: Maximum R2 score achieved.
        - pos_r: The feature combination that achieved the maximum R2 score.
        - error_min: Minimum Mean Squared Error (MSE) achieved.
        - pos_e: The feature combination that achieved the minimum MSE.
        - acc_max: Maximum Custom Accuracy achieved.
        - pos_acc: The feature combination that achieved the maximum accuracy.
    """
    y = np.array(norm_merged_df['test_VASS'])

    error_min = np.inf
    pos_e = 0
    r2_max = 0
    pos_r = 0
    acc_max = 0  # New variable to store accuracy
    pos_acc = 0  # New variable to store position of the best accuracy


    l = combination(best_corr, 1) + combination(best_corr, 2) + combination(best_corr, 3) + combination(best_corr, 4) + combination(best_corr, 5) + [best_corr]
    random.seed()
    for j in l:
        dt_std_cp = norm_merged_df.copy()

        for i in dt_std_cp:
            if i not in j:
                dt_std_cp = dt_std_cp.drop(i, axis=1)

        x = dt_std_cp.to_numpy()
        error = 0
        r2 = 0
        acc = 0

        x_train, y_train, x_val, y_val = split_data(x, y)

        if regressor == 'RF':
            regr = RandomForestRegressor(n_estimators=estimator, max_depth=depth, random_state=random.seed(42))
        if regressor == 'GRADBOOST':
            regr = GradientBoostingRegressor(n_estimators=estimator, learning_rate=rate, max_depth=depth, random_state=random.seed(42))
        if regressor == 'SVR':
            regr = SVR(kernel='linear', C=c, epsilon=epsi)

        regr.fit(x_train, y_train)


        error += mean_squared_error(y_val, regr.predict(x_val))
        r2 += r2_score(y_val, regr.predict(x_val))

        # Calculate the custom metric: percentage of predictions within a threshold
        acc = custom_accuracy(y_val, regr.predict(x_val))

        if show_process:
            print(j)
            print("Mean Squared Error: ", error )
            print("R2 Score: ", r2 )
            print("Custom Accuracy: ", acc)
            print("--------------------------------------")

        if r2 > r2_max:
            r2_max = r2
            pos_r = j
        if error < error_min:
            error_min = error 
            pos_e = j
        if acc > acc_max:
            acc_max = acc
            pos_acc = j

    if not_opti:
        print("Para el modelo de regresión:", regressor)
        print("Mejor R2 Score de todos: ", pos_r, r2_max)
        print("Mejor MSE de todos: ", pos_e, error_min)
        print("Mejor Custom Accuracy de todos: ", pos_acc, acc_max)


    return r2_max, pos_r, error_min, pos_e, acc_max, pos_acc



def best_params_rf(n_estimators, max_depth_values, norm_merged_df, best_corr, plot = True):
    """
    Input:
        n_estimators (list): List of values for the number of estimators (trees).
        max_depth_values (list): List of values for the maximum depth of the tree.
        norm_merged_df (DataFrame): The normalized and merged DataFrame containing patient data.
        best_corr (list): A list of selected features for regression.
        plot (bool): Whether to plot the evolution of MSE and R2 values. Default is True.

    Description:
        Find the best hyperparameters for the Random Forest regression model based on R2 score, MSE, and accuracy.

    Output:
        - r2_max: Maximum R2 score achieved.
        - error_min: Minimum Mean Squared Error (MSE) achieved.
        - acc_max: Maximum Custom Accuracy achieved.
    """

    r2_max = -100
    error_min = 100
    acc_max = 0
    mse_values = []
    r2_values = []
    opt_acc = []

    for estimator in n_estimators:
        for depth in max_depth_values:
            r_2, pos_r, mse, pos_e, acc_m, poss_acc = perform_regression_2(norm_merged_df, best_corr, False, estimator, depth)
            r2_values.append((estimator, depth, r_2))
            mse_values.append((estimator, depth, mse))
            if r_2 > r2_max:
                r2_max = r_2
                opt_r = pos_r
                best_n_r2 = estimator
                best_depth_r2 = depth
            if mse < error_min:
                error_min = mse
                opt_mse = pos_e
                best_n_mse = estimator
                best_depth_mse = depth
            if acc_m > acc_max:
                acc_max = acc_m
                opt_acc = poss_acc
                best_n_acc = estimator
                best_depth_acc = depth
    
    if plot:
        # Plot evolution mse and r2
        plot_evolution_rf(n_estimators, max_depth_values, mse_values, r2_values)

    print("--------------------------------------")
    print("RESULTADOS MEJORES PARÁMETROS")
    print("--------------------------------------")
    print("Mejor R2 Score de todos: ", opt_r, r2_max)
    print("Mejor número de n_estimators para R2: ", best_n_r2)
    print("Mejor número de max_depth para R2: ", best_depth_r2)
    print("Mejor MSE de todos: ", opt_mse, error_min)
    print("Mejor número de n_estimators para MSE: ", best_n_mse)
    print("Mejor número de max_depth para MSE: ", best_depth_mse)
    print("Mejor accuracy de todos: ", opt_acc, acc_max)
    print("Mejor número de n_estimators para accuracy: ", best_n_acc)
    print("Mejor número de max_depth para accuracy: ", best_depth_acc)
    print("--------------------------------------")
    print("--------------------------------------")

    return r2_max, error_min, acc_max


    
def best_params_gb(n_estimators, learning_rate_values, max_depth_values, norm_merged_df, best_corr, plot=True):
    """
    Input:
        n_estimators (list): List of values for the number of estimators.
        learning_rate_values (list): List of values for the learning rate.
        max_depth_values (list): List of values for the maximum depth of the individual trees.
        norm_merged_df (DataFrame): The normalized and merged DataFrame containing patient data.
        best_corr (list): A list of selected features for regression.
        plot (bool): Whether to plot the evolution of MSE and R2 values. Default is True.

    Description:
        Find the best hyperparameters for the Gradient Boosting regression model based on R2 score, MSE, and accuracy.

    Output:
        - r2_max: Maximum R2 score achieved.
        - error_min: Minimum Mean Squared Error (MSE) achieved.
        - acc_max: Maximum Custom Accuracy achieved.
    """
    r2_max = -100
    error_min = 100
    acc_max = 0
    mse_values = []
    r2_values = []
    opt_acc = []

    for estimator in n_estimators:
        for rate in learning_rate_values:
            for depth in max_depth_values:
                r_2, pos_r, mse, pos_e, acc_m, poss_acc = perform_regression_2(norm_merged_df, best_corr, False, estimator, depth, rate, True, regressor='GRADBOOST')
                r2_values.append((estimator, depth, rate, r_2))
                mse_values.append((estimator, depth, rate, mse))
                if r_2 > r2_max:
                    r2_max = r_2
                    opt_r = pos_r
                    best_n_r2 = estimator
                    best_depth_r2 = depth
                    best_rate_r2 = rate
                if mse < error_min:
                    error_min = mse
                    opt_mse = pos_e
                    best_n_mse = estimator
                    best_depth_mse = depth
                    best_rate_mse = rate
                if acc_m > acc_max:
                    acc_max = acc_m
                    opt_acc = poss_acc
                    best_n_acc = estimator
                    best_depth_acc = depth
    
    if plot:
        # Plot evolution mse and r2
        plot_evolution_gb(n_estimators, max_depth_values, learning_rate_values, mse_values, r2_values)

    print("--------------------------------------")
    print("RESULTADOS MEJORES PARÁMETROS")
    print("--------------------------------------")
    print("Mejor R2 Score de todos: ", opt_r, r2_max)
    print("Mejor número de n_estimators para R2: ", best_n_r2)
    print("Mejor número de max_depth para R2: ", best_depth_r2)
    print("Mejor learning_rate para R2: ", best_rate_r2)
    print("Mejor MSE de todos: ", opt_mse, error_min)
    print("Mejor número de n_estimators para MSE: ", best_n_mse)
    print("Mejor número de max_depth para MSE: ", best_depth_mse)
    print("Mejor learning_rate para MSE: ", best_rate_mse)
    print("Mejor accuracy de todos: ", opt_acc, acc_max)
    print("Mejor número de n_estimators para accuracy: ", best_n_acc)
    print("Mejor número de max_depth para accuracy: ", best_depth_acc)
    print("--------------------------------------")
    print("--------------------------------------")

    return r2_max, error_min, acc_max



def best_params_svr(c_list, epsilon_list, norm_merged_df, best_corr, plot = True):
    """
    Input:
        c_list (list): List of values for the regularization parameter C in SVR.
        epsilon_list (list): List of values for the epsilon parameter in SVR.
        norm_merged_df (DataFrame): The normalized and merged DataFrame containing patient data.
        best_corr (list): A list of selected features for regression.
        plot (bool): Whether to plot the evolution of MSE and R2 values. Default is True.

    Description:
        Find the best hyperparameters for the Support Vector Regression (SVR) model based on R2 score, MSE, and accuracy.

    Output:
        - r2_max: Maximum R2 score achieved.
        - error_min: Minimum Mean Squared Error (MSE) achieved.
        - acc_max: Maximum Custom Accuracy achieved.
    """
   
    r2_max = -100
    error_min = 100
    acc_max = 0
    mse_values = []
    r2_values = []
    opt_acc = []

    for c in c_list:
        for epsi in epsilon_list:
            r_2, pos_r, mse, pos_e, acc_m, poss_acc = perform_regression_2(norm_merged_df, best_corr, False, c, epsi, True, regressor='SVR')
            r2_values.append((c, epsi, r_2))
            mse_values.append((c, epsi, mse))
            if r_2 > r2_max:
                r2_max = r_2
                opt_r = pos_r
                best_c_r2 = c
                best_epsi_r2 = epsi
            if mse < error_min:
                error_min = mse
                opt_mse = pos_e
                best_c_mse = c
                best_epsi_mse = epsi
            if acc_m > acc_max:
                acc_max = acc_m
                opt_acc = poss_acc
                best_c_acc = c
                best_epsi_acc = epsi
    
    if plot:
        # Plot evolution MSE and R²
        plot_evolution_svr(c_list, epsilon_list, mse_values, r2_values)

    print("--------------------------------------")
    print("RESULTADOS MEJORES PARÁMETROS")
    print("--------------------------------------")
    print("Mejor R2 Score de todos: ", opt_r, r2_max)
    print("Mejor número de c para R2: ", best_c_r2)
    print("Mejor número de epsilon para R2: ", best_epsi_r2)
    print("Mejor MSE de todos: ", opt_mse, error_min)
    print("Mejor número de c para MSE: ", best_c_mse)
    print("Mejor número de epsilon para MSE: ", best_epsi_mse)
    print("Mejor accuracy de todos: ", opt_acc, acc_max)
    print("Mejor número de c para accuracy: ", best_c_acc)
    print("Mejor número de epsi para accuracy: ", best_epsi_acc)
    print("--------------------------------------")
    print("--------------------------------------")

    return r2_max, error_min, acc_max


############################# FUNCTIONS PLOTTING ##################################


def plot_heatmap(correlations, best_corr, reduced_corr, all = True, reduced = True, best = True):
    """
    Input:
        correlations (DataFrame): The correlation matrix.
        best_corr (list): A list of selected features for regression.
        reduced_corr (list): A list of reduced features based on correlation.
        all (bool, optional): Whether to plot the heatmap of all correlations. Default is True.
        reduced (bool): Whether to plot the heatmap of reduced correlations. Default is True.
        best (bool): Whether to plot the heatmap of best correlations. Default is True.

    Description:
        Plot heatmaps of correlations based on the specified criteria.

    Output: None
    """

    if all:
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlations, annot=False, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title('Heatmap of All Correlations')
        plt.tight_layout()  # Adjust layout to prevent cutoff
        plt.show()

    if reduced:
        red_corr_df = correlations[reduced_corr].loc[reduced_corr]
        plt.figure(figsize=(8, 6))
        sns.heatmap(red_corr_df, annot=False, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title('Heatmap of Reduced Correlations')
        plt.xticks(rotation=45, ha='right')  # Adjust the rotation and alignment as needed
        plt.yticks(rotation=0)  # Adjust the rotation as needed
        plt.tight_layout()  # Adjust layout to prevent cutoff
        plt.show()

    if best:
        best_corr_df = correlations[best_corr].loc[best_corr]
        plt.figure(figsize=(8, 6))
        sns.heatmap(best_corr_df, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title('Heatmap of Best Correlations')
        plt.xticks(rotation=45, ha='right')  # Adjust the rotation and alignment as needed
        plt.yticks(rotation=0)  # Adjust the rotation as needed
        plt.tight_layout()  # Adjust layout to prevent cutoff
        plt.show()



def plot_pairplot(data_corr, show=True):
    """
    Input:
        data_corr (DataFrame): The DataFrame containing columns for pair plotting.
        show (bool): Whether to display the pair plot. Default is True.

    Description:
        Plot a pair plot for visualizing relationships between variables.

    Output: None
    """
    if show:
        sns.set(style="ticks", color_codes=True)
        sns.pairplot(data_corr)
        plt.show()



def plot_3d_rels(data_corr, best_corr, show = True):
    """
    Input:
        data_corr (DataFrame): The DataFrame containing columns for 3D relationships.
        best_corr (list): List of column names to be considered in the plot.
        show (bool): Whether to display the 3D relationship plots. Default is True.

    Description:
        Plot 3D relationships between variables in the specified DataFrame.

    Output:
        None
    """

    if show: # Show the plot
        M = data_corr.to_numpy()
        M = M.transpose()

        l = [i for i in range(np.shape(M)[0]-1)]
        n = best_corr[1:]

        z = np.array(data_corr['test_VASS'])


        for i in l:
            for j in l:
                if j > i:
                    print("test_VASS: ", n[i], "--", n[j])

                    # Create the figure
                    fig = plt.figure()

                    # Create the 3D plot
                    ax = fig.add_subplot(111, projection='3d')

                    # Define the data
                    x = M[i]
                    y = M[j]

                    # Define the color based on 'test_VASS' values
                    color = cm.viridis(z)

                    # Add the points to the 3D plot with color mapping
                    scatter = ax.scatter(x, y, z, c=color, marker='o', cmap='viridis')

                    # Add color bar
                    cbar = plt.colorbar(scatter)
                    cbar.set_label('test_VASS')

                    # Add labels to axes
                    ax.set_xlabel(n[i])
                    ax.set_ylabel(n[j])
                    ax.set_zlabel('test_VASS')

                    
                    plt.show()
                    plt.close()



def plot_predictions(y_true, y_pred, model_line=None, title="Plot de Predicciones"):
    """
    Input:
        y_true (array-like): The true values.
        y_pred (array-like): The predicted values.
        model_line (array-like): The model line to be plotted. Default is None.
        title (str): The title of the plot. Default is "Plot de Predicciones".

    Description: Plot predictions against true values with an optional model line.

    Output: None
    """

    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, y_true, alpha=0.5, label="Predicciones vs. Datos reales")
    
    if model_line is not None:
        sorted_indices = np.argsort(y_pred)
        plt.plot(y_pred[sorted_indices], model_line[sorted_indices], color='red', linewidth=2, label="Línea del Modelo")

    plt.title(title)
    plt.xlabel("Predicciones")
    plt.ylabel("Valores Reales")
    plt.legend()
    nombre_archivo = str(uuid.uuid4()) + ".png"
    ruta_archivo = os.path.join("C:/Users/onasa/Documents/3r2n_semestre/BDnR/Pycharm/TFG_plots_auto/", nombre_archivo)
    plt.savefig(ruta_archivo)



def plot_evolution_rf(n_estimators_values, max_depth_values, mse_values, r2_values):
    """
    Input:
        n_estimators_values (array-like): List of values for the number of estimators.
        max_depth_values (array-like): List of values for the maximum depth.
        mse_values (list): List of MSE values.
        r2_values (list): List of R2 values.

    Description:
        Plot the evolution of MSE and R2 with varying hyperparameters (number of estimators and max depth) for Random Forest.

    Output None
    """

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    for i, metric_values in enumerate([(mse_values, 'MSE'), (r2_values, 'R2')]):
        metric_data, metric_label = metric_values

        for depth in max_depth_values:
            metric_vals = [metric for (n_estimators, max_depth, metric) in metric_data if max_depth == depth]

            axes[i].plot(n_estimators_values, metric_vals, label=f'Max Depth={depth}')

        axes[i].set_xlabel('Number of Estimators')
        axes[i].set_ylabel(metric_label)
        axes[i].set_title(f'Evolution of {metric_label} with Hyperparameters')
        axes[i].legend()
    nombre_archivo = str(uuid.uuid4()) + ".png"
    ruta_archivo = os.path.join("C:/Users/onasa/Documents/3r2n_semestre/BDnR/Pycharm/TFG_plots_auto/", nombre_archivo)
    plt.savefig(ruta_archivo)



def plot_evolution_gb(n_estimators_values, max_depth_values, learning_rate_values, mse_values, r2_values):
    """
    Input:
        n_estimators_values (array-like): List of values for the number of estimators.
        max_depth_values (array-like): List of values for the maximum depth.
        learning_rate_values (array-like): List of values for the learning rate.
        mse_values (list): List of MSE values.
        r2_values (list): List of R2 values.

    Description:
        Plot the evolution of MSE and R2 with varying hyperparameters (number of estimators, max depth, and learning rate) for Gradient Boosting.

    Output: None
    """
    fig = plt.figure(figsize=(15, 10))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_xlabel('Number of Estimators')
    ax1.set_ylabel('Max Depth')
    ax1.set_zlabel('Learning Rate')
    ax1.set_title('Evolution of MSE with Hyperparameters')

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_xlabel('Number of Estimators')
    ax2.set_ylabel('Max Depth')
    ax2.set_zlabel('Learning Rate')
    ax2.set_title('Evolution of R2 with Hyperparameters')

    n_estimators_values, max_depth_values, learning_rate_values, mse_vals = zip(*mse_values)
    ax1.scatter(n_estimators_values, max_depth_values, learning_rate_values, c=mse_vals, marker='o', label='MSE')

    n_estimators_values, max_depth_values, learning_rate_values, r2_vals = zip(*r2_values)
    ax2.scatter(n_estimators_values, max_depth_values, learning_rate_values, c=r2_vals, marker='^', label='R2')

    ax1.legend()
    ax2.legend()
    nombre_archivo = str(uuid.uuid4()) + ".png"
    ruta_archivo = os.path.join("C:/Users/onasa/Documents/3r2n_semestre/BDnR/Pycharm/TFG_plots_auto/", nombre_archivo)
    plt.savefig(ruta_archivo)



def plot_evolution_svr(c_list, epsilon_list, mse_values, r2_values):
    """
    Input:
        c_list (array-like): List of values for the regularization parameter C.
        epsilon_list (array-like): List of values for the epsilon parameter.
        mse_values (list): List of tuples containing (C, epsilon, MSE).
        r2_values (list): List of tuples containing (C, epsilon, R2).

    Description:
        Plot the evolution of MSE and R2 with varying hyperparameters (C and epsilon) for Support Vector Regression (SVR).

    Output: None
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Iterate over metrics (MSE and R2)
    for i, metric_values in enumerate([(mse_values, 'MSE'), (r2_values, 'R2')]):
        metric_data, metric_label = metric_values

        # Generate a unique color for each data point
        colors = plt.cm.viridis(np.linspace(0, 1, len(metric_data)))

        # Iterate over data points
        for j, (c, epsilon, metric) in enumerate(metric_data):
            # Scatter plot with a unique color for each data point
            axes[i].scatter(c, epsilon, color=colors[j], label=f'Metric Value={metric}', alpha=0.7)

        axes[i].set_xlabel('C (Regularization Parameter)')
        axes[i].set_ylabel('Epsilon')
        axes[i].set_title(f'Evolution of {metric_label}')
        axes[i].legend()

    plt.tight_layout()  # Adjust layout for better spacing
    nombre_archivo = str(uuid.uuid4()) + ".png"
    ruta_archivo = os.path.join("C:/Users/onasa/Documents/3r2n_semestre/BDnR/Pycharm/TFG_plots_auto/", nombre_archivo)
    plt.savefig(ruta_archivo)



def plot_metrics(result_dicts):
    """
    Input:
        result_dicts (dict): Dictionary containing performance metrics (mse, r2, accuracy) for different models.

    Description:
        Plot a grouped bar chart to visualize the normalized R2 score, MSE, and Accuracy for each model.

    Output: None
    """
    models = list(result_dicts['r2_list'].keys())

    r2_values = np.array(list(result_dicts['r2_list'].values()))
    mse_values = np.array(list(result_dicts['mse_list'].values()))
    accuracy_values = np.array(list(result_dicts['acc_list'].values()))

    accuracy_values_norm = accuracy_values / 100

    bar_width = 0.2
    index = np.arange(len(models))

    plt.figure(figsize=(12, 8))

    plt.bar(index - bar_width, r2_values, bar_width, label='R2 Score', color='skyblue')
    plt.bar(index, mse_values, bar_width, label='MSE', color='lightcoral')
    plt.bar(index + bar_width, accuracy_values_norm, bar_width, label='Accuracy', color='lightgreen')

    plt.xlabel('Model')
    plt.ylabel('Metric Value')
    plt.title('Performance Metrics for Each Model')
    plt.xticks(index, models)
    plt.legend()
    nombre_archivo = str(uuid.uuid4()) + ".png"
    ruta_archivo = os.path.join("C:/Users/onasa/Documents/3r2n_semestre/BDnR/Pycharm/TFG_plots_auto/", nombre_archivo)
    plt.savefig(ruta_archivo)
