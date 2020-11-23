import sys

from pandas.compat import numpy
from sqlalchemy import create_engine
from pandas import read_sql_table
from pandas import DataFrame
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
from nltk import download
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from pickle import dump

download('wordnet')
download('punkt')
download('stopwords')
download('averaged_perceptron_tagger')

def load_data(database_filepath):
    engine = create_engine(database_filepath)
    df = read_sql_table('messages', engine)
    Y = df.drop(columns=['message', 'original', 'id'])
    X = df['message']
    return X, Y


def tokenize(text):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = map(lambda tok: lemmatizer.lemmatize(tok), tokens)
    return list(tokens)


def build_model():
    pipeline = Pipeline([
        ('vectorize', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('classifier', MultiOutputClassifier(KNeighborsClassifier()))
    ])

    parameters = {
        'vectorize__analyzer': 'word',
        'vectorize__binary': False,
        'vectorize__decode_error': 'strict',
        # 'vectorize__dtype': numpy.int64,
        'vectorize__encoding': 'utf-8',
        'vectorize__input': 'content',
        'vectorize__lowercase': True,
        'vectorize__max_df': 1.0,
        'vectorize__max_features': None,
        'vectorize__min_df': 1,
        'vectorize__ngram_range': (1, 1),
        'vectorize__preprocessor': None,
        'vectorize__stop_words': None,
        'vectorize__strip_accents': None,
        'vectorize__token_pattern': '(?u)\\b\\w\\w+\\b',
        # 'vectorize__tokenizer': tokenize(),
        'vectorize__vocabulary': None,
        'tfidf__norm': 'l2',
        'tfidf__smooth_idf': True,
        'tfidf__sublinear_tf': False,
        'tfidf__use_idf': True,
        'classifier__estimator__algorithm': ('auto', 'kd_tree'),
        'classifier__estimator__leaf_size': (20, 25, 30, 35, 40),
        'classifier__estimator__n_jobs': (1, 2),
        'classifier__estimator__n_neighbors': (3, 4, 5, 6),
        'classifier__estimator__p': (1, 2, 3),
        'classifier__estimator__weights': ('uniform', 'distance'),
        'classifier__n_jobs': (1, 2)}

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):

    Y_pred = model.predict(X_test)
    confusion_mat = confusion_matrix(Y_test, Y_pred, labels=category_names)
    accuracy = (Y_pred == Y_test).mean()

    print("Labels:", category_names)
    print("Confusion Matrix:\n", confusion_mat)
    print("Accuracy:", accuracy)
    print("\nBest Parameters:", model.best_params_)
    print(classification_report(Y_test, Y_pred, labels=category_names))


def save_model(model, model_filepath):
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
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()