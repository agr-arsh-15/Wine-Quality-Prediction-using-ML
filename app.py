from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

# HTML template for displaying result
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Prediction Result</title>
</head>
<body>
    <h1>Wine Quality Prediction</h1>
    <form action="/predict" method="post">
        <label>Enter 11 Features (comma separated):</label>
        <input type="text" name="features" required>
        <button type="submit">Predict</button>
    </form>
    {% if prediction is not none %}
    <h2>Prediction: {{ prediction }}</h2>
    {% endif %}
</body>
</html>
"""

@app.route('/', methods=['GET'])
def home():
    return render_template_string(html_template, prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input from form
        features = request.form['features']
        features_list = np.array([float(x) for x in features.split(',')]).reshape(1, -1)

        # Validate input
        if features_list.shape[1] != 11:
            return "Error: Please enter exactly 11 features."

        # Predict
        prediction = model.predict(features_list)[0]
        return render_template_string(html_template, prediction=prediction)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)