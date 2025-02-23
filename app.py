from flask import Flask, request, render_template
import pickle
import pandas as pd

# Load the trained pipeline
with open('model_pipeline.pkl', 'rb') as f:
    model_pipeline = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        input_data = {
            'Age': [float(request.form['Age'])],
            'Sleep Duration': [float(request.form['Sleep Duration'])],
            'Stress Level': [float(request.form['Stress Level'])],
            'Gender': [request.form['Gender']],
            'Sleep Disorder': [request.form['Sleep Disorder']],
            'BMI Category': [request.form['BMI Category']]
        }

        # Create a DataFrame from the input data
        input_df = pd.DataFrame(input_data)

        # Make prediction using the loaded pipeline
        prediction = model_pipeline.predict(input_df)

        # Return the prediction result
        return render_template('index.html', prediction_text=f'Predicted Work Performance: {prediction[0]}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
