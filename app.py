from flask import Flask, render_template, request # type: ignore
import pickle
import numpy as np # type: ignore

# Initialize the Flask app
app = Flask(__name__)

# Load the trained insurance model
try:
    with open('insurance_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Define the route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Define the route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from form
        age = int(request.form['age'])

        # Prepare the input for the model
        features = np.array([[age]])

        # Predict whether insurance was bought
        prediction = model.predict(features)[0]

        # Render result page with the prediction
        result = "eligible for insurance" if prediction == 1 else "not eligible for insurance"
        return render_template('result.html', age=age, result=result)

    except Exception as e:
        print(f"Error occurred during prediction: {str(e)}")
        return render_template('result.html', age=None, result="Error")

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
