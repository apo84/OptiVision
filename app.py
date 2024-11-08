from flask import Flask, request, render_template
import numpy as np

app = Flask(__name__)

# Route to serve the homepage with the form
@app.route('/')
def homepage():
    return render_template('OptiVision.html')

# Flatten each array with a check
def flatten_array(arr):
    # If the element is a numpy array, flatten and convert it to a list of floats
    if isinstance(arr[0], np.ndarray):
        return [float(val) for val in arr[0]]
    # If it's already a float, return it as a list with one element
    elif isinstance(arr[0], float):
        return [arr[0]]
    # Handle any unexpected cases
    else:
        raise TypeError("Unexpected data type in array.")

# Route to handle the form submission
@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the form
    percent_increase = request.form['percentIncrease']
    time_period = request.form['timePeriod']
    ticker = request.form['ticker']

    # Process the input with your Python logic (dummy example here)
    # In a real scenario, you could have complex logic here
    from optionAnalyzer import combination_analysis
    num_buys, knn_pred_count, knn_prec_count, nb_pred_count, nb_prec_count, svm_pred_count, svm_prec_count = combination_analysis(ticker, int(time_period), int(percent_increase))
    
    # Apply the flattening function to each variable
    knn_pred_count = flatten_array(knn_pred_count)
    knn_prec_count = flatten_array(knn_prec_count)
    nb_pred_count = flatten_array(nb_pred_count)
    nb_prec_count = flatten_array(nb_prec_count)
    svm_pred_count = flatten_array(svm_pred_count)
    svm_prec_count = flatten_array(svm_prec_count)
    

    # Return the result back to the user
    return render_template('results.html',
                           num_buys=num_buys, 
                           knn_pred_count=knn_pred_count, 
                           knn_prec_count=knn_prec_count, 
                           nb_pred_count=nb_pred_count, 
                           nb_prec_count=nb_prec_count, 
                           svm_pred_count=svm_pred_count, 
                           svm_prec_count=svm_prec_count)

if __name__ == '__main__':
    app.run(debug=True)
