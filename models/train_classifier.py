import sys
from sqlite3 import connect
from pandas import read_sql_query, DataFrame, Series
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
from nltk import download
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from pickle import dump

download('wordnet')
download('punkt')
download('stopwords')
download('averaged_perceptron_tagger')


def load_data(database_filepath):
    """This function loads the data from a SQL database to a dataframe"""
    connection = connect(database_filepath)
    df = read_sql_query('SELECT * FROM messages', connection)
    Y = df.drop(columns=['message', 'original', 'id', 'genre'])
    X = df['message']
    return X, Y, Y.columns


def tokenize(text):
    """This function is used to tokenize the text input from the messages column in the dataframe"""
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = map(lambda tok: lemmatizer.lemmatize(tok), tokens)
    return list(tokens)


def build_model():
    """This function builds the ML model"""
    pipeline = Pipeline([
        ('vectorize', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('classifier', MultiOutputClassifier(KNeighborsClassifier()))
    ])

    parameters = {
        # UNCOMMENT ACCORDING TO NEEDS
        # 'classifier__estimator__algorithm': ('auto', 'kd_tree'),
        # 'classifier__estimator__leaf_size': (20, 25,),
        # 'classifier__estimator__n_jobs': (1),
        'classifier__estimator__n_neighbors': (3, 4),
        # 'classifier__estimator__p': (1, 2),
        # 'classifier__estimator__weights': ('uniform', 'distance'),
        # 'classifier__n_jobs': (1)
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """This function evaluates the accuracy of the model"""
    Y_pred = DataFrame(data=model.predict(X_test), columns=category_names)
    for category in category_names:
        print(classification_report(Y_test[category], Y_pred[category]))
    print("\nBest Parameters:", model.best_params_)


def save_model(model, model_filepath):
    """This function saves the trained ML model as a pickle file"""
    dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        try:
            evaluate_model(model, X_test, Y_test, category_names)
        except Exception as e:
            print(e)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
