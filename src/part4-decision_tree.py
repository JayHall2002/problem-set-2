'''
PART 4: Decision Trees
'''

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

def run_decision_tree(df_arrests_train, df_arrests_test):
    X_train = df_arrests_train[['current_charge_felony', 'num_fel_arrests_last_year']]
    y_train = df_arrests_train['y']
    X_test = df_arrests_test[['current_charge_felony', 'num_fel_arrests_last_year']]
    y_test = df_arrests_test['y']

    param_grid_dt = {'max_depth': [3, 5, 7]}
    dt_model = DecisionTreeClassifier()
    gs_cv_dt = GridSearchCV(dt_model, param_grid_dt, cv=5)
    gs_cv_dt.fit(X_train, y_train)

    print("Optimal value for max_depth:", gs_cv_dt.best_params_['max_depth'])
    print("Did it have the most or least regularization? Or in the middle?")
    # Print interpretation based on max_depth value

    pred_dt = gs_cv_dt.predict(X_test)
    df_arrests_test['pred_dt'] = pred_dt

    return df_arrests_test, pred_dt
