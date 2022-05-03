# python IFT3710/logisticsRegression.py EEG/Med_eeg_raw/
'''
Authors : Equipe EEG
Fichier pour training le model Logistic Regression pour dataset med_eeg
Last updated : 15-04-2022

The code is inspire from : https://www.youtube.com/watch?v=cuEV-eB3Dyo&list=PLtGXgNsNHqPTgP9wyR8pmy2EuM2ZGHU5Z&index=2
'''

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, GridSearchCV
import numpy as np

from eeg_meditation import med_preprocessing


# Function for model Loggistic Regression
def logRegression(Y, features, group, params_grid):
    print('-- Traing.....')
    clf = LogisticRegression(max_iter=300)
    gkf = GroupKFold(5)
    pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
    param_grid = params_grid
    grid_search_cv = GridSearchCV(pipe, param_grid, cv=gkf, n_jobs=12)
    grid_search_cv.fit(features, Y, groups=group)

    accuracy = np.array(grid_search_cv.cv_results_['mean_test_score']).reshape(
        len(param_grid['clf__C']))

    print('-- Done --')

    return accuracy, grid_search_cv.best_params_, grid_search_cv.best_score_
