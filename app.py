from flask import Flask, render_template, request
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

# Load model and data
model = pickle.load(open('car_price_model.pkl', 'rb'))
car = pd.read_csv('car_cleaned.csv')

@app.route('/', methods=['GET', 'POST'])
def index():
    companies = sorted(car['company'].unique().tolist())
    car_models = sorted(car['name'].unique().tolist())
    year = sorted(car['year'].unique().tolist(), reverse=True)
    fuel_type = sorted(car['fuel_type'].unique().tolist())

    companies.insert(0, 'Select Company')
    return render_template('index.html', companies=companies, car_models=car_models, years=year, fuel_types=fuel_type)


@app.route('/predict', methods=['POST'])
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_models')
    year = int(request.form.get('year'))  # Convert to int
    fuel_type = request.form.get('fuel_type')
    driven = float(request.form.get('kilo_driven'))  # Convert to float

    prediction = model.predict(pd.DataFrame({
        'name': [car_model],
        'company': [company],
        'year': [year],
        'kms_driven': [driven],
        'fuel_type': [fuel_type]
    }))

    return str(np.round(prediction[0], 2))


if __name__ == '__main__':
    app.run()