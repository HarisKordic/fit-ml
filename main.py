import pandas as pd

# Define the file paths
input_file_path = r'data\raw_data.csv'
output_file_path = r'data\filtered_model_data.csv'

# Read the CSV file with a specific encoding
df = pd.read_csv(input_file_path, delimiter=';', encoding='latin1')

# Define the filter values
filter_values = ['P-175', 'P-176', 'P-177', 'P-150', 'P-157']

# Filter the 'Predmet' column
filtered_df = df[df['Predmet'].isin(filter_values)]

# Save the filtered data to a new CSV file
filtered_df.to_csv(output_file_path, index=False)
