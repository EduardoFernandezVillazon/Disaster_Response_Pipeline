from models.train_classifier import evaluate_model, load_data, tokenize
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from pandas import DataFrame
import joblib

model = joblib.load("../models/classifier.pkl")
database_filepath = '../data/DisasterResponse.db'
X, Y, category_names = load_data(database_filepath)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

Y_pred= DataFrame(data=model.predict(X_test.head(10)), columns=category_names)
evaluate_model(model, X_test.head(10), Y_test.head(10), category_names)