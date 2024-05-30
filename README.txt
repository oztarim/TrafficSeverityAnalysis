# Traffic Accident Severity Prediction

This project uses a trained machine learning model to predict the severity of traffic accidents based on user input. The model is built using PyCaret and various other Python libraries. The application is hosted using Flask and data is stored in a PostgreSQL database.

## Requirements

- Anaconda
- Python 3.8+
- PostgreSQL

## Setup Instructions

### 1. Clone the Repository

Clone the repository to your local machine:

```sh
git clone <repository-url>
cd <repository-directory>


2. Create and Activate Virtual Environment

conda create --name pycaret-env python=3.8
conda activate pycaret-env

3. Install Dependencies

pip install -r requirements.txt

4. Set Up PostgreSQL Database (For Development Only - No need for DEMO)

5. Run the Application

python app.py

6. Access the Application

Open your web browser and navigate to http://localhost:5000 to access the application.

Usage

	1.	Fill in the form with the necessary details.
	2.	Click the “Predict” button to get the severity prediction and confidence score.

Notes

	•	Ensure you have the necessary permissions and configurations for the PostgreSQL database.
	•	If you encounter any dependency issues, make sure the versions of the packages are compatible as specified in requirements.txt.

Troubleshooting

	•	If you encounter an error related to werkzeug.urls during the Flask app run, ensure you have the compatible versions of Flask and Werkzeug installed.

Example Command for Compatibility Issues

pip install Flask==2.0.2 Werkzeug==2.0.2