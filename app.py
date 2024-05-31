# app.py
from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('best_tuned_model.pkl')

@app.route('/')
def form():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect user input
    input_data = {
        'start_hour': [int(request.form['start_hour'])],
        'weather_condition': [request.form['weather_condition']],
        'junction': [request.form['junction']],
        'traffic_signal': [request.form['traffic_signal']],
        'roundabout': [request.form['roundabout']],
        'traffic_calming': [request.form['traffic_calming']],
        'stop': [request.form['stop']],
        'railway': [request.form['railway']],
        'no_exit': [request.form['no_exit']],
        'bump': [request.form['bump']]
    }

    # Convert to DataFrame
    input_df = pd.DataFrame.from_dict(input_data)
    
    # Make predictions using the loaded model
    predictions = model.predict(input_df)
    
    # Get the prediction probabilities
    prediction_probabilities = model.predict_proba(input_df)
    
    # Get the predicted severity and its probability
    severity = predictions[0]
    probability = max(prediction_probabilities[0]) * 100  # Convert to percentage
    
    return render_template('result.html', severity=severity, probability=probability)

if __name__ == '__main__':
    app.run(debug=True)
