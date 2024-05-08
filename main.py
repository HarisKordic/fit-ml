import pandas as pd

# Define the file paths
input_file_path = r'data\raw_data.csv'
model_subjects_path = r'data\model_subjects.csv'
output_file_path = r'data\filtered_model_data.csv'

# Read the CSV file with a specific encoding
df = pd.read_csv(input_file_path, delimiter=';', encoding='latin1')

# Read the filter values as a single row and split them into individual values
with open(model_subjects_path, 'r') as file:
    filter_values = file.read().strip().split(',')

# Filter the 'Predmet' column
filtered_df = df[df['Predmet'].isin(filter_values)]

# Save the filtered data to a new CSV file
filtered_df.to_csv(output_file_path, index=False)
