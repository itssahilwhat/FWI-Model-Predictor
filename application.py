import pickle
from flask import Flask, request, render_template
import numpy as np

application = Flask(__name__)
app = application

# Import the Ridge regression model and scaler
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

# Home page route
@app.route('/')
def index():
    return render_template('index.html')

# Prediction page route
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        # Retrieve input values
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        # Scale the input data
        new_data_scaled = standard_scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        # Make prediction
        result = ridge_model.predict(new_data_scaled)

        # Render the home page with the result
        return render_template('home.html', result=result[0])

    # Render the prediction form if GET request
    return render_template('home.html', result=None)

if __name__ == "__main__":
    app.run(host="0.0.0.0")