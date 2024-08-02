'''
PART 2: Pre-processing
'''

import pandas as pd
import os

def preprocess_data():
    data_dir = os.path.join(os.path.dirname(__file__), '../data')

    pred_universe_raw = pd.read_csv(os.path.join(data_dir, 'pred_universe_raw.csv'))
    arrest_events_raw = pd.read_csv(os.path.join(data_dir, 'arrest_events_raw.csv'))

    # Print column names for verification
    print("pred_universe_raw columns:", pred_universe_raw.columns)
    print("arrest_events_raw columns:", arrest_events_raw.columns)

    df_arrests = pd.merge(pred_universe_raw, arrest_events_raw, on='person_id', how='outer')

    # Check for column names after merging
    print("df_arrests columns:", df_arrests.columns)

    # Ensure date columns are in datetime format
    df_arrests['arrest_date_event'] = pd.to_datetime(df_arrests['arrest_date_event'])
    arrest_events_raw['arrest_date_event'] = pd.to_datetime(arrest_events_raw['arrest_date_event'])

    # Create target variable y
    df_arrests['y'] = df_arrests.apply(lambda row: check_felony_rearrest(row, arrest_events_raw), axis=1)
    print("What share of arrestees were rearrested for a felony crime in the next year?")
    print(df_arrests['y'].mean())

    # Check the distribution of 'y'
    print("Distribution of 'y':")
    print(df_arrests['y'].value_counts())

    # Create predictive features
    if 'offense_category' in df_arrests.columns:
        df_arrests['current_charge_felony'] = df_arrests['offense_category'].apply(lambda x: 1 if 'Felony' in x else 0)
        print("What share of current charges are felonies?")
        print(df_arrests['current_charge_felony'].mean())
    else:
        print("'offense_category' column not found in df_arrests")

    df_arrests['num_fel_arrests_last_year'] = df_arrests.apply(lambda row: count_felony_arrests_last_year(row, arrest_events_raw), axis=1)
    print("What is the average number of felony arrests in the last year?")
    print(df_arrests['num_fel_arrests_last_year'].mean())

    return df_arrests

# Helper functions (example implementation)
def check_felony_rearrest(row, arrest_events_raw):
    # Check if the person was rearrested for a felony crime in the next year
    arrest_date = row['arrest_date_event']
    person_id = row['person_id']
    if pd.isnull(arrest_date) or pd.isnull(person_id):
        return 0
    rearrests = arrest_events_raw[(arrest_events_raw['person_id'] == person_id) &
                                  (arrest_events_raw['arrest_date_event'] > arrest_date) &
                                  (arrest_events_raw['arrest_date_event'] <= arrest_date + pd.Timedelta(days=365)) &
                                  (arrest_events_raw['offense_category'].str.contains('Felony', na=False))]
    # Debugging statement
    print(f"Person ID: {person_id}, Arrest Date: {arrest_date}, Rearrests: {len(rearrests)}")
    return 1 if not rearrests.empty else 0

def count_felony_arrests_last_year(row, arrest_events_raw):
    # Count felony arrests in the last year
    arrest_date = row['arrest_date_event']
    person_id = row['person_id']
    if pd.isnull(arrest_date) or pd.isnull(person_id):
        return 0
    past_arrests = arrest_events_raw[(arrest_events_raw['person_id'] == person_id) &
                                     (arrest_events_raw['arrest_date_event'] < arrest_date) &
                                     (arrest_events_raw['arrest_date_event'] >= arrest_date - pd.Timedelta(days=365)) &
                                     (arrest_events_raw['offense_category'].str.contains('Felony', na=False))]
    return len(past_arrests)

# If this script is run standalone, execute the function
if __name__ == "__main__":
    preprocess_data()
