import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

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

# Select the features and target column
X = df[['Ocjena', 'BrojKredita']]
y = df['BrojKredita']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the decision tree regressor
reg = DecisionTreeRegressor()
reg.fit(X_train, y_train)

# Make predictions on the test set
y_pred = reg.predict(X_test)

# Evaluate the model
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-squared:", r2_score(y_test, y_pred))

# Predict the number of ECTS credits for a new subject
new_subject = [[8, 9]]  # Ocjena = 7, BrojKredita = 9
predicted_credits = reg.predict(new_subject)[0]
print("\nPredicted Number of ECTS Credits for the new subject:", predicted_credits)
