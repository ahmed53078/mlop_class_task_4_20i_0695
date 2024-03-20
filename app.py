from flask import Flask, request, jsonify
import pickle
import os

app = Flask(__name__)
current_dir = os.path.dirname(__file__)
model_path = os.path.join(os.getcwd(), 'model.pkl')
with open(model_path, 'rb') as file:
    model = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    housing_median_age = data['housing_median_age']
    total_rooms = data['total_rooms']
    total_bedrooms = data['total_bedrooms']
    population = data['population']
    households = data['households']
    median_income = data['median_income']
    ocean_proximity = data['ocean_proximity']
    prediction_data = [[housing_median_age, total_rooms, total_bedrooms, population, households, median_income, ocean_proximity]]
    prediction = model.predict(prediction_data)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)