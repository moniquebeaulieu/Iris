from flask import Flask
from flask import request, render_template

from joblib import load
import numpy as np

app = Flask(__name__)

# load iris classfier from file
#      use try-except, print message and exit if there is a problem
try:
    clf = load('iris-classifier.joblib')
except:
    print("Error loading classifier") 
    exit()


@app.route('/')
def index():
    return render_template('prediction_input.html')

#GET REQUEST
@app.route('/iris_prediction')
def get_iris_prediction():
    
    values_ok = True
    pred_str = 'None'
    pred_proba = 0.0
    
    # Get feature values as float from request.values dictionary
    #      Set values_ok to False if any conversion produces an error.
    feature_names = ['sepal_length','sepal_width','petal_length','petal_width']
    target_names = ['setosa', 'versicolor', 'virginica']
    
    feature_values= []
    try:
        for name in feature_names:
            feature_values.append(float(request.values[name]))
    except:
        values_ok = False
    else:
        values_ok = True
        
    feature_values = np.reshape(feature_values, (1, -1))


    # print(feature_values) 

    
    # call predict() on the loaded classifier using the feature values
    #      and retrieve the predicted iris flower string
    #      assign string to pred_str
    index_arr_pred = clf.predict(feature_values)
    index_pred = index_arr_pred[0]
    # print(index_pred)
    

    pred_str =  target_names[index_pred]
    print(pred_str)

    # call predict_proba() on the loaded classifier
    #      assign probablity to pred_proba
    array_pred_proba = clf.predict_proba(feature_values)
    pred_proba = array_pred_proba[0][index_pred]
    
    
    return render_template('prediction_response.html',
                           request_dict=request.values,
                          pred_str=pred_str,
                           pred_proba='{:.3f}'.format(pred_proba),
                          values_ok=values_ok) 


if __name__ == '__main__':
    app.debug = True
    app.run()
