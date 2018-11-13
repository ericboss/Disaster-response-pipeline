# Disaster Response Pipeline Project

## Project Motivation
This project helped me build my data engineering skills to expand my opportunities and potential as a data scientist. In this project, I applied these skills to analyze disaster data from [Figure Eight](https://www.figure-eight.com/) to build a model for an API that classifies disaster messages.

### Install

This project requires **Python 3** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [regex](https://regexr.com/)
- [nltk](https://www.nltk.org/)
- [Sqlalchemy](https://www.sqlalchemy.org/)
- [Plotly](https://plot.ly/)
- [json](https://www.json.org/)
- [Flask](http://flask.pocoo.org/)

## File Description

    workspace.
    ├── app     
    │   ├── run.py                           # Flask file that runs app
    │   └── templates   
    │       ├── go.html                      # Classification result page of web app
    │       └── master.html                  # Main page of web app    
    ├── data                   
    │   ├── disaster_categories.csv          # Dataset including all the categories  
    │   ├── disaster_messages.csv            # Dataset including all the messages
    │   └── process_data.py                  # Data cleaning
    ├── models
    │   └── train_classifier.py              # Train ML model           
    └── README.md


### Instructions:
1. Run the following commands in the project's root directory(workspace) to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

![alt text](https://github.com/ericboss/Disaster-response-pipeline/blob/master/workspace/img.png)


![alt text](https://github.com/ericboss/Disaster-response-pipeline/blob/master/workspace/img2.png)
