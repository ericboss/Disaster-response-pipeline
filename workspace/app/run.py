import json
import plotly
import pandas as pd
from collections import Counter
import numpy as np 
import re
import nltk 
nltk.download(['punkt', 'stopwords', 'wordnet'])
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from plotly.graph_objs import Pie
from sklearn.externals import joblib
from sqlalchemy import create_engine



app = Flask(__name__)

def tokenize(text):
    """Prepare text for modeling.
    Args:
        text: Text string.
    Returns:
        clean_tokens: Cleaned tokens ready for modeling.
    """
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text
    tokens = word_tokenize(text)

    # lemmatize and remove stop words
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words("english")
    clean_tokens = [lemmatizer.lemmatize(word).strip() for word in tokens if word not in stop_words]

    return clean_tokens
# load model
model = joblib.load("models/classifier.pkl")
# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table("disaster", engine)




# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    columns = list(df.columns[4:])
    mean_ = []
    for i in columns:
        mean_.append(df[i].mean())
    def compute_word_counts(messages, load=True, filepath='data/counts.npz'):
        '''
        input: (
        messages: list or numpy array
        load: Boolean value if load or run model 
        filepath: filepath to save or load data
            )
        Function computes the top 20 words in the dataset with counts of each term
        output: (
        top_words: list
        top_counts: list 
            )
        '''
        if load:
        # load arrays
            data = np.load(filepath)
            return list(data['top_words']), list(data['top_counts'])
        else:
        # get top words 
            counter = Counter()
            for message in messages:
                tokens = tokenize(message)
                for token in tokens:
                    counter[token] += 1
        # top 20 words 
            top = counter.most_common(20)
            top_words = [word[0] for word in top]
            top_counts = [count[1] for count in top]
        # save arrays
            np.savez(filepath, top_words=top_words, top_counts=top_counts)
        return list(top_words), list(top_counts)
    
    message = df.message.values.tolist()
    word, count = compute_word_counts(message, load=False)
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
                Pie(
                    labels=word[:10],
                    values=count[:10]
                )
            ],

           "layout": {
        "title":"Top Ten Word Messages"}
             
            
        },
        
        {
            'data': [
                Pie(
                    labels=columns,
                    values=mean_
                )
            ],

           "layout": {
        "title":"Distribution of Category Messages"}
             
            
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
