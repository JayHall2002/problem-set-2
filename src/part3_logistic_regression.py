'''
PART 3: Logistic Regression
'''

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression

def run_logistic_regression(df_arrests):
    X = df_arrests[['current_charge_felony', 'num_fel_arrests_last_year']]
    y = df_arrests['y']

    df_arrests_train, df_arrests_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, stratify=y)

    param_grid = {'C': [0.01, 0.1, 1]}
    lr_model = LogisticRegression()
    gs_cv = GridSearchCV(lr_model, param_grid, cv=5)
    gs_cv.fit(df_arrests_train, y_train)

    print("Optimal value for C:", gs_cv.best_params_['C'])
    print("Did it have the most or least regularization? Or in the middle?")
    # Print interpretation based on C value

    pred_lr = gs_cv.predict(df_arrests_test)
    df_arrests_test['pred_lr'] = pred_lr

    return df_arrests_train, df_arrests_test, pred_lr
