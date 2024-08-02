import pandas as pd

# Load data
data = pd.read_csv('/mnt/data/recidivism.csv')

# Drop rows with NaN values
data.dropna(inplace=True)

# Convert the 'Arrest Date' column to datetime format
data['Arrest Date'] = pd.to_datetime(data['Arrest Date'])

# Calculate the number of rearrests for each person
rearrests = data.groupby('Person ID')['Arrest Date'].count().reset_index()
rearrests.columns = ['Person ID', 'Rearrests']
rearrests['Rearrests'] -= 1

# Calculate the number of people with rearrests
num_people_with_rearrests = rearrests[rearrests['Rearrests'] > 0].shape[0]

print(f"Number of people with rearrests: {num_people_with_rearrests}")

# Extract relevant columns
data = data[['Person ID', 'Arrest Date']]

# Merge the rearrests count back to the original dataframe
data = data.merge(rearrests, on='Person ID')

# Keep only the records with rearrests
people_with_rearrests = data[data['Rearrests'] > 0]

# Sort by Person ID and Arrest Date
people_with_rearrests = people_with_rearrests.sort_values(by=['Person ID', 'Arrest Date'])

# Find the earliest arrest date for each person
first_arrests = people_with_rearrests.groupby('Person ID').first().reset_index()

# Function to calculate time difference in months
def diff_month(d1, d2):
    return (d1.year - d2.year) * 12 + d1.month - d2.month

# Calculate time to rearrest in months
people_with_rearrests['Time to Rearrest (months)'] = people_with_rearrests.apply(
    lambda row: diff_month(row['Arrest Date'], first_arrests[first_arrests['Person ID'] == row['Person ID']]['Arrest Date'].values[0]), axis=1)

# Filter out the first arrest date rows as they are not rearrests
people_with_rearrests = people_with_rearrests[people_with_rearrests['Time to Rearrest (months)'] != 0]

# Print the results
for index, row in people_with_rearrests.iterrows():
    print(f"Person ID: {row['Person ID']}, Arrest Date: {row['Arrest Date']}, Time to Rearrest (months): {row['Time to Rearrest (months)']}")

# Part 2: Find the number of people with no rearrests
num_people_no_rearrests = rearrests[rearrests['Rearrests'] == 0].shape[0]

print(f"Number of people with no rearrests: {num_people_no_rearrests}")

# Extract people with no rearrests
people_no_rearrests = data[data['Rearrests'] == 0]

# Sort by Person ID and Arrest Date
people_no_rearrests = people_no_rearrests.sort_values(by=['Person ID', 'Arrest Date'])

# Print the results
for index, row in people_no_rearrests.iterrows():
    print(f"Person ID: {row['Person ID']}, Arrest Date: {row['Arrest Date']}, Rearrests: {row['Rearrests']}")
