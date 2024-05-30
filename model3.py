import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMClassifier  # Use LightGBM for faster training
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import joblib
import matplotlib.pyplot as plt

# Database connection parameters
db_name = 'TrafficData'
db_user = 'postgres'
db_password = 'pavlov3'
db_host = 'localhost'
db_port = '5432'

# Create a connection string
connection_string = f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'

# Create a database engine
engine = create_engine(connection_string)

# Query to load data
query = "SELECT * FROM accidents2 LIMIT 5000000"

# Load data into a Pandas DataFrame
data = pd.read_sql(query, engine)

# Ensure all categorical columns are converted to strings
categorical_features = [
    'weather_condition', 'junction', 'traffic_signal', 
    'roundabout', 'traffic_calming', 'stop', 'railway', 'no_exit', 'bump'
]

for col in categorical_features:
    if col in data.columns:
        data[col] = data[col].astype('category')  # Use category dtype for efficiency

# Feature Engineering
data['start_hour'] = pd.to_datetime(data['start_time']).dt.hour

# Drop irrelevant columns
columns_to_keep = [
    'start_hour', 'weather_condition', 'junction', 'traffic_signal', 
    'roundabout', 'traffic_calming', 'stop', 'railway', 'no_exit', 'bump', 'severity'
]

data = data[columns_to_keep]

# Handle remaining missing values
data = data.fillna(method='ffill').fillna(method='bfill')

# Split the data into training and holdout test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=123, stratify=data['severity'])

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', ['start_hour']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Define a classifier
classifier = LGBMClassifier(random_state=123)

# Build a pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', classifier)
])

# Define hyperparameters for tuning
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__learning_rate': [0.01, 0.1],
    'classifier__max_depth': [3, 5]
}

# Perform Grid Search with cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1)

# Fit the pipeline
grid_search.fit(train_data.drop(columns=['severity']), train_data['severity'])

# Save the best model
joblib.dump(grid_search.best_estimator_, 'best_tuned_model.pkl')

# Load the model (for testing purposes)
model = joblib.load('best_tuned_model.pkl')

# Make predictions on the test set
predictions = model.predict(test_data.drop(columns=['severity']))

# Evaluate the model
accuracy = accuracy_score(test_data['severity'], predictions)
print(f"Accuracy: {accuracy}")

# Display the confusion matrix
conf_matrix = confusion_matrix(test_data['severity'], predictions)
print("Confusion Matrix:")
print(conf_matrix)

# Plot the confusion matrix
ConfusionMatrixDisplay(conf_matrix).plot()
plt.show()
