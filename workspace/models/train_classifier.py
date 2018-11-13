# import libraries
import sys
import nltk
nltk.download(['punkt', 'stopwords', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

import pandas as pd
import re
from sqlalchemy import create_engine
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, classification_report
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import pickle
import sys


def load_data(database_filepath):
    """Load data from SQL table.
    Args:
        database_filepath: Path to database.
    Returns:
        X: Features for modeling.
        Y: Labels for model.
        category_names: Labels column names.
    """
    engine = create_engine('sqlite:///' + database_filepath)
    # Must use table name from process_data.py. 'cleanData' in this case
    df = pd.read_sql_table('disaster', engine)
    
    
    X = df['message'].values
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = Y.columns
    return X, Y, category_names


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



def build_model():
    """
    Builds classification model 
     Args:
        None
    Returns:
        cv: model pipeline.  
    
    """

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'clf__estimator__min_samples_split': [2, 4],
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=2, n_jobs=4)
    return cv




def evaluate_model(model, X_test, Y_test, category_names):
     """
    Evaluates classification model performance on data 
     Args:
        model: classification model
        X_test: test data
        Y_test: test categories
        caregory_names: column names of the categories
    Returns:
        None
    
    """

    
    y_preds = model.predict(X_test)
    print(classification_report(Y_test.values, y_preds, target_names=category_names))
        
            



def save_model(model, model_filepath):
     """
    Saves classification model 
     Args:
        model: classification model
        model_filepath: path to save the model
    Returns:
        None  
    
    """

    # Dump the trained  classifier with Pickle
    model_pkl_filename = model_filepath
    # Open the file to save as pkl file
    model_pkl = open(model_pkl_filename, 'wb')
    pickle.dump(model, model_pkl)
    # Close the pickle instances
    model_pkl.close()


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
