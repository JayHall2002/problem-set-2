import pandas as pd

def preprocess_data():
    # Load the datasets saved in Part 1
    pred_universe_raw = pd.read_csv('../data/pred_universe_raw.csv')
    arrest_events_raw = pd.read_csv('../data/arrest_events_raw.csv')

    # Process the data as required
    pred_universe_raw['arrest_date_univ'] = pd.to_datetime(pred_universe_raw['arrest_date_univ'])
    arrest_events_raw['arrest_date_event'] = pd.to_datetime(arrest_events_raw['arrest_date_event'])

    # Merge the datasets
    merged_data = pd.merge(pred_universe_raw, arrest_events_raw, left_on='Person ID', right_on='Person ID')

    # Drop rows with NaN values
    merged_data.dropna(inplace=True)

    # Calculate the number of rearrests for each person
    rearrests = merged_data.groupby('Person ID')['arrest_date_event'].count().reset_index()
    rearrests.columns = ['Person ID', 'Rearrests']
    rearrests['Rearrests'] -= 1

    # Calculate the number of people with rearrests
    num_people_with_rearrests = rearrests[rearrests['Rearrests'] > 0].shape[0]
    print(f"Number of people with rearrests: {num_people_with_rearrests}")

    # Extract relevant columns
    merged_data = merged_data[['Person ID', 'arrest_date_event']]

    # Merge the rearrests count back to the original dataframe
    merged_data = merged_data.merge(rearrests, on='Person ID')

    # Keep only the records with rearrests
    people_with_rearrests = merged_data[merged_data['Rearrests'] > 0]

    # Sort by Person ID and Arrest Date
    people_with_rearrests = people_with_rearrests.sort_values(by=['Person ID', 'arrest_date_event'])

    # Find the earliest arrest date for each person
    first_arrests = people_with_rearrests.groupby('Person ID').first().reset_index()

    # Function to calculate time difference in months
    def diff_month(d1, d2):
        return (d1.year - d2.year) * 12 + d1.month - d2.month

    # Calculate time to rearrest in months
    people_with_rearrests['Time to Rearrest (months)'] = people_with_rearrests.apply(
        lambda row: diff_month(row['arrest_date_event'], first_arrests[first_arrests['Person ID'] == row['Person ID']]['arrest_date_event'].values[0]), axis=1)

    # Filter out the first arrest date rows as they are not rearrests
    people_with_rearrests = people_with_rearrests[people_with_rearrests['Time to Rearrest (months)'] != 0]

    # Print the results
    for index, row in people_with_rearrests.iterrows():
        print(f"Person ID: {row['Person ID']}, Arrest Date: {row['arrest_date_event']}, Time to Rearrest (months): {row['Time to Rearrest (months)']}")

    # Part 2: Find the number of people with no rearrests
    num_people_no_rearrests = rearrests[rearrests['Rearrests'] == 0].shape[0]
    print(f"Number of people with no rearrests: {num_people_no_rearrests}")

    # Extract people with no rearrests
    people_no_rearrests = merged_data[merged_data['Rearrests'] == 0]

    # Sort by Person ID and Arrest Date
    people_no_rearrests = people_no_rearrests.sort_values(by=['Person ID', 'arrest_date_event'])

    # Print the results
    for index, row in people_no_rearrests.iterrows():
        print(f"Person ID: {row['Person ID']}, Arrest Date: {row['arrest_date_event']}, Rearrests: {row['Rearrests']}")

    return merged_data

# If this script is run standalone, execute the preprocess_data function
if __name__ == "__main__":
    preprocess_data()
