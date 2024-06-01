mma867project
Traffic Accident Severity Prediction
This project uses a trained machine learning model to predict the severity of traffic accidents based on user input. The model, built using PyCaret and trained on 7 million data points with LightGBM, creates the best_tuned_model.pkl for predictions. The application is hosted using Flask.

Requirements (Tested on Windows with different computers) MacOS has problems with
Anaconda
Python 3.8+
Git https://git-scm.com/download/win
PostgreSQL (For Development Only)
Setup Instructions
1. Clone the Repository
Clone the repository to your local machine:

Go to terminal

git clone "https://oztarim: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@github.com/oztarim/mma867project.git"


cd mma867project
Create and Activate Virtual Environment

Run the following commands

conda create --name pycaret-env python=3.8

conda activate pycaret-env
Install Dependencies

pip install -r requirements.txt
Run the Application

python app.py
Access the Application
Open your web browser and navigate to http://localhost:5000 to access the application.

Usage

1.	Fill in the form with the necessary details.
2.	Click the “Predict” button to get the severity prediction and confidence score.
Compatibility Tested on operating system Windows with different computers and had no issues on deployment. MacOS had LightGBM package problems during tests on Apple computers. Problems can be solved by following brew instructions to meet certain package requirements.

Notes

•	If you encounter any dependency issues, make sure the versions of the packages are compatible as specified in requirements.txt.
Troubleshooting

•	If you encounter an error related to werkzeug.urls during the Flask app run, ensure you have the compatible versions of Flask and Werkzeug installed.
Example Command for Compatibility Issues

pip install Flask==2.0.2 Werkzeug==2.0.2
