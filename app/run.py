import json
import plotly
from pandas import DataFrame, Series
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from pandas import read_sql_table
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine
from sqlite3 import connect
from pandas import read_sql_query


app = Flask(__name__)

def tokenize(text):
    """Function used to tokenize"""

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def get_common_topics(df, number_of_topics):
    """This function returns"""
    df_to_sort = df.drop(columns=['message', 'original', 'id', 'genre'])
    df_to_sort = df_to_sort.astype('float').aggregate('mean', axis=0)
    df_to_sort = df_to_sort.to_frame().reset_index().rename(columns={'index': 'Topic', 0: 'Mean'}).sort_values(
        by='Mean', ascending=False)
    return df_to_sort.head(number_of_topics).Mean, df_to_sort.head(number_of_topics).Topic

# load data
connection = connect('../data/DisasterResponse.db')
df = read_sql_query('SELECT * FROM messages', connection)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    topic_frequencies, topic_names = get_common_topics(df, 10)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=topic_names,
                    y=topic_frequencies
                )
            ],

            'layout': {
                'title': 'Frequencies of the Most Common Topics',
                'yaxis': {
                    'title': "Relative Frequency"
                },
                'xaxis': {
                    'title': "Topic"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()