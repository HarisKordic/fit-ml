import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
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

# Define the hyperparameters to tune
param_grid = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create the grid search object
grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,  # 5-fold cross-validation
    scoring='accuracy',
    n_jobs=-1,  # Use all available cores
    verbose=1
)

# Perform the grid search to find the best hyperparameters
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Train the decision tree classifier with the best hyperparameters
clf = DecisionTreeClassifier(**best_params)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Predict the ECTS class for a new subject
#new_subject = [[6, 9, 6*9]]
new_subject = [[9, 9, 9*9]]
predicted_class = clf.predict(new_subject)[0]
print("\nPredicted ECTS Class for the new subject:", predicted_class)
