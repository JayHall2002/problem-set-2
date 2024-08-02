import pandas as pd

def preprocess_data():
    # Load the datasets
    pred_universe_raw = pd.read_csv('../data/pred_universe_raw.csv')
    arrest_events_raw = pd.read_csv('../data/arrest_events_raw.csv')

    # Drop rows with NaN values
    pred_universe_raw.dropna(inplace=True)
    arrest_events_raw.dropna(inplace=True)

    # Convert the 'arrest_date_univ' and 'arrest_date_event' columns to datetime format
    pred_universe_raw['arrest_date_univ'] = pd.to_datetime(pred_universe_raw['arrest_date_univ'])
    arrest_events_raw['arrest_date_event'] = pd.to_datetime(arrest_events_raw['arrest_date_event'])

    # Merge the datasets on 'person_id'
    merged_data = pd.merge(pred_universe_raw, arrest_events_raw, left_on='person_id', right_on='person_id')

    # Calculate the number of rearrests for each person
    rearrests = merged_data.groupby('person_id')['arrest_date_event'].count().reset_index()
    rearrests.columns = ['person_id', 'rearrests']
    rearrests['rearrests'] -= 1

    # Calculate the number of people with rearrests
    num_people_with_rearrests = rearrests[rearrests['rearrests'] > 0].shape[0]

    print(f"Number of people with rearrests: {num_people_with_rearrests}")

    # Extract relevant columns
    data = merged_data[['person_id', 'arrest_date_event']]

    # Merge the rearrests count back to the original dataframe
    data = data.merge(rearrests, on='person_id')

    # Keep only the records with rearrests
    people_with_rearrests = data[data['rearrests'] > 0]

    # Sort by person_id and arrest_date_event
    people_with_rearrests = people_with_rearrests.sort_values(by=['person_id', 'arrest_date_event'])

    # Find the earliest arrest date for each person
    first_arrests = people_with_rearrests.groupby('person_id').first().reset_index()

    # Function to calculate time difference in months
    def diff_month(d1, d2):
        return (d1.year - d2.year) * 12 + d1.month - d2.month

    # Calculate time to rearrest in months
    people_with_rearrests['time_to_rearrest_months'] = people_with_rearrests.apply(
        lambda row: diff_month(row['arrest_date_event'], first_arrests[first_arrests['person_id'] == row['person_id']]['arrest_date_event'].values[0]), axis=1)

    # Filter out the first arrest date rows as they are not rearrests
    people_with_rearrests = people_with_rearrests[people_with_rearrests['time_to_rearrest_months'] != 0]

    # Print the results
    for index, row in people_with_rearrests.iterrows():
        print(f"Person ID: {row['person_id']}, Arrest Date: {row['arrest_date_event']}, Time to Rearrest (months): {row['time_to_rearrest_months']}")

    # Part 2: Find the number of people with no rearrests
    num_people_no_rearrests = rearrests[rearrests['rearrests'] == 0].shape[0]

    print(f"Number of people with no rearrests: {num_people_no_rearrests}")

    # Extract people with no rearrests
    people_no_rearrests = data[data['rearrests'] == 0]

    # Sort by person_id and arrest_date_event
    people_no_rearrests = people_no_rearrests.sort_values(by=['person_id', 'arrest_date_event'])

    # Print the results
    for index, row in people_no_rearrests.iterrows():
        print(f"Person ID: {row['person_id']}, Arrest Date: {row['arrest_date_event']}, Rearrests: {row['rearrests']}")

    return merged_data

# If this script is run standalone, execute the preprocess_data function
if __name__ == "__main__":
    preprocess_data()
