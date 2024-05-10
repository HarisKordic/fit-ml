import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

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



# Define the classes based on ECTS credits
def classify_ects(ects):
    if ects >= 8:
        return 'High'
    elif ects >= 6:
        return 'Middle'
    else:
        return 'Low'

# Apply the classification to create the target column
df['ECTS_class'] = df['BrojKredita'].apply(classify_ects)

# Create a new feature that combines Ocjena and BrojKredita
df['Ocjena_BrojKredita_interaction'] = df['Ocjena'] * df['BrojKredita']

# Select the features and target column
X = df[['Ocjena', 'BrojKredita', 'Ocjena_BrojKredita_interaction']]
y = df['ECTS_class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Predict the ECTS class for a new subject
new_subject = [[6, 6, 6*6]]  # Ocjena = 6, BrojKredita = 10
predicted_class = clf.predict(new_subject)[0]
print("\nPredicted ECTS Class for the new subject:", predicted_class)
