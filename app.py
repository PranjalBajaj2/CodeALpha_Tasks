from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load model
model, columns = joblib.load('credit_model.joblib')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array([[data.get(col, 0) for col in columns]])
    prob = model.predict_proba(features)[0][1]
    pred = model.predict(features)[0]
    return jsonify({'prediction': int(pred), 'probability': round(float(prob), 2)})

if __name__ == '__main__':
    app.run(debug=True)
