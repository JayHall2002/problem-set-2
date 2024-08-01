'''
PART 3: Logistic Regression
'''

import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

def run(data_dir):
    # Paths to the preprocessed data files
    preprocessed_pred_universe_file = os.path.join(data_dir, 'preprocessed_pred_universe.csv')
    preprocessed_arrest_events_file = os.path.join(data_dir, 'preprocessed_arrest_events.csv')

    # Read the data
    pred_universe = pd.read_csv(preprocessed_pred_universe_file)
    arrest_events = pd.read_csv(preprocessed_arrest_events_file)

    # Assume 'target' is the column to be predicted
    X = pred_universe.drop(columns=['target'])
    y = pred_universe['target']

    # Check the distribution of the target variable
    unique_classes = y.unique()
    if len(unique_classes) < 2:
        raise ValueError(f"The target variable needs to have at least two classes, but it has only one class: {unique_classes}")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the model and parameters
    model = LogisticRegression()
    param_grid = {'C': [0.1, 1, 10, 100, 1000], 'solver': ['liblinear', 'saga']}
    gs_cv = GridSearchCV(model, param_grid, cv=5)

    # Fit the model
    gs_cv.fit(X_train, y_train)

    # Make predictions
    y_pred = gs_cv.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Logistic Regression Accuracy: {accuracy}")

    # Save the model if needed
    # import joblib
    # joblib.dump(gs_cv.best_estimator(), os.path.join(data_dir, 'logistic_regression_model.pkl'))

    # Optionally return data for further analysis
    return X_train, X_test, y_train, y_test, gs_cv.best_estimator_

# If this script is run standalone, execute the run function
if __name__ == "__main__":
    data_directory = os.path.join(os.path.dirname(__file__), '../data')
    run(data_directory)
