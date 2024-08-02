'''
You will run this problem set from main.py, so set things up accordingly
'''

import pandas as pd
import part1_etl as etl
import part2_preprocessing as preprocessing
import part3_logistic_regression as logistic_regression
import part4_decision_tree as decision_tree
import part5_calibration_plot as calibration_plot


# Call functions / instantiate objects from the .py files
def main():
    # PART 1: Instantiate etl, saving the two datasets in `./data/`
    etl.extract_and_save_data()

    # PART 2: Call functions/instantiate objects from preprocessing
    df_arrests = preprocessing.preprocess_data()
    print("Columns in df_arrests:", df_arrests.columns)
    # Rename or create the target column 'y'
    if 'y' not in df_arrests.columns:
        df_arrests['y'] = df_arrests['charge_degree']  # Adjust 'charge_degree' to your target column


    # PART 3: Call functions/instantiate objects from logistic_regression
    df_arrests_train, df_arrests_test, pred_lr = logistic_regression.run_logistic_regression(df_arrests)

    # PART 4: Call functions/instantiate objects from decision_tree
    df_arrests_test, pred_dt = decision_tree.run_decision_tree(df_arrests_train, df_arrests_test)

    # PART 5: Call functions/instantiate objects from calibration_plot
    y_test = df_arrests_test['y']
    calibration_plot.calibration_plot(y_test, df_arrests_test['pred_lr'], n_bins=5)
    calibration_plot.calibration_plot(y_test, df_arrests_test['pred_dt'], n_bins=5)

    # Print and analyze calibration results
    print("Which model is more calibrated?")
    # More code for PPV and AUC calculations for extra credit can go here

if __name__ == "__main__":
    main()
