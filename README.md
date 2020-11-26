# Disaster Response Pipeline Project

###Summary:
This project is a guided project for the Udacity Data Scientist Nanodegree. The purpose of this project is to create a machine learning pipeline to process and classify emergency messages. It should then help emergency services prioritize actions when in an extreme situation when they are saturated with emergency messages. The trained model should be able to classify a message according to 36 categories. The training data set consists of labelled messages from tweets in emergency situations.

### Instructions:
How to use the repository:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

###Description of the repository:

The repository is divided in three directories:

- app: contains the HTML templates and the python script to run the webapp backend: run.py.
- data: it contains the raw csv data files, the final .db file with the clean dataset in a SQLite database (once the data has been processed), and the process_data.py to process the raw data and create the database.
- models: it contains the train_classifier.py file which creates and trains the classifier model, which is then stored in a .pkl file in this same directory.

###File description:

- run.py: this file creates the webapp backend using flask. It will create visuals that will be displayed on the front end, including a message classification functionality.
- go.html: this file contains the HTML template for the message classification functionality.
- master.html: this file contains the HTML template for the graphs and visuals.
- process_data.py: this file implements the ETL pipeline that will feed data to our machine learning model. It extracts and processes the data from the raw csv files and outputs a clean dataset stored in SQLite database.
- train_classifier.py: this file trains a classifier model using the clean data and labels from the SQLite. It implements a GridSearchCV model with a ML pipeline to find the optimal parameters for the model. 
