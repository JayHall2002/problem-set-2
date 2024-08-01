'''
PART 2: Pre-processing
'''

import pandas as pd

def preprocess_data():
    pred_universe_raw = pd.read_csv('../data/pred_universe_raw.csv')
    arrest_events_raw = pd.read_csv('../data/arrest_events_raw.csv')

    df_arrests = pd.merge(pred_universe_raw, arrest_events_raw, on='person_id', how='outer')

    # Create target variable y
    df_arrests['y'] = df_arrests.apply(lambda row: check_felony_rearrest(row), axis=1)
    print("What share of arrestees were rearrested for a felony crime in the next year?")
    print(df_arrests['y'].mean())

    # Create predictive features
    df_arrests['current_charge_felony'] = df_arrests['charge_type'].apply(lambda x: 1 if x == 'Felony' else 0)
    print("What share of current charges are felonies?")
    print(df_arrests['current_charge_felony'].mean())

    df_arrests['num_fel_arrests_last_year'] = df_arrests.apply(lambda row: count_felony_arrests_last_year(row), axis=1)
    print("What is the average number of felony arrests in the last year?")
    print(df_arrests['num_fel_arrests_last_year'].mean())

    return df_arrests

# Helper functions (example implementation)
def check_felony_rearrest(row):
    # Logic to check if the person was rearrested for a felony crime in the next year
    pass

def count_felony_arrests_last_year(row):
    # Logic to count felony arrests in the last year
    pass
