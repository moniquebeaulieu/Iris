# Lab5 - Iris classification web server

## Goal

The goal of this lab is to train a classifier on the Iris dataset, save it to file, and allow users to enter Iris flower measurements on a website to receive the predicted Iris flower type.

You will train the classifier in a Jupyter notbook and save it to file.

In a Python script, a Flask server template is prepared that allows obtaining Iris flower measurements. You will modify the flask server to load the classifier, parse the Iris measurements and return the predicted Iris type and probability.

## What to do

### Dataset
scikit-learn Iris dataset:  
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html


### Classifier 
- `LogisticRegression()`


### Steps

A. Train a classifier and save to file in `train_iris_classifier.ipynb`:
  1. Load iris dataset from sklearn.
  2. use `train_test_split()` with `random_state=34` to create train and test sets.
  3. Train a logistic regression model on the training data.
  4. Print accuracy on the test dataset.
  5. Save the model to file using joblib.

B. Adapt to code of the flask server in `flask-ml-server.py`
  - Load the classifier.
  - Parse the request values to obtain the feature values.
  - Generate prediction and predicted probability.


### Specifications
Add code to `train_iris_classifier.ipynb` as indicated.

In `flask-ml-server.py`:
- Add code to load the classifier as indicated.
- Modify `get_iris_prediction()` function to parse and produce predicted Iris type.



## What to hand in
- Python script `flask-ml-server.py`.
- Jupyter notebook `train_iris_classifier.ipynb`.
- Keep code clean and remove any unnecessary cells. 
- Complete the *Conclusion* section in the notebook.
- Complete the *Reflection* section in the notebook, include a sentence or two about what you liked/disliked, found interesting/confusing/challangeing/motivating while working on this assignment.

During development, checkin progress with git and use descriptive commit messages.
