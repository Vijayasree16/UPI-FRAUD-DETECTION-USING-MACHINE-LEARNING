from flask import Flask, request, render_template
import pickle
import pandas as pd

# Load the trained model and column names
with open('fraud_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('model_columns.pkl', 'rb') as file:
    model_columns = pickle.load(file)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user inputs
    input_data = {
        'amount': float(request.form['amount']),
        'oldbalanceOrg': float(request.form['oldbalanceOrg']),
        'newbalanceOrig': float(request.form['newbalanceOrig']),
        'oldbalanceDest': float(request.form['oldbalanceDest']),
        'newbalanceDest': float(request.form['newbalanceDest']),
        'type_TRANSFER': 1 if request.form['type'] == 'TRANSFER' else 0,
        'type_CASH_OUT': 1 if request.form['type'] == 'CASH_OUT' else 0,
    }

    # Convert to DataFrame and reindex to match the model's expected input
    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    # Make prediction and get probabilities for both classes
    prediction_proba = model.predict_proba(input_df)[0][1]  # Probability of fraud (class 1)
    
    # Adjust threshold to be more sensitive to fraud detection
    threshold = 0.05 # You can adjust this threshold (e.g., to 0.4 or 0.6 based on performance)
    result = "Fraudulent" if prediction_proba > threshold else "Legitimate"

    return render_template('index.html', prediction_text=f"Transaction is: {result}")

if __name__ == '__main__':
    app.run(debug=True)
