import pandas as pnd
import numpy as np
import random as rnd
from sklearn import preprocessing

def scale_column(data_frame,column):
    temp = data_frame[column]
    temp = (temp - temp.mean()) / (temp.max() - temp.min())
    data_frame[column] = temp
    return data_frame
    
def extract_cabin(data_frame):
    data_frame.Cabin = data_frame.Embarked.fillna('N')
    return data_frame
    
def clean_age(data_frame):
    data_frame.Age = data_frame.Embarked.fillna(-1)
    return data_frame
    
def clean_port(data_frame):
    data_frame.Embarked = data_frame.Embarked.fillna('N')
    return data_frame
    
def family(data_frame):
    data_frame['Family'] = data_frame.SibSp + data_frame.Parch
    return data_frame
    
def parse_title(Name):
    name_parts = Name.split(" ")
    title = "Common"
    for _,part in enumerate(name_parts):
        if part.contains("."):
            title = part
            break
    return title
    
def parse_names(data_frame):
    data_frame['Surname'] = data_frame['Name'].apply(lambda x: x.split(',')[0])
    data_frame['Title'] = data_frame['Name'].apply(parse_title)
    return data_frame
    
def pre_wrangle_data_frame(data_frame):
    data_frame = extract_cabin(data_frame)
    data_frame = clean_age(data_frame)
    data_frame = clean_port(data_frame)
    data_frame = family(data_frame)
    data_frame = scale_column(data_frame,"Fare")
    data_frame = scale_column(data_frame,"Age")
    data_frame = parse_names(data_frame)
    data_frame = data_frame.drop(['PassengerId','Ticket', 'Name', 'Embarked','SibSp','Parch'], axis=1)
    return data_frame

def prepare_data(train_set, test_set):
    train_set = pre_wrangle_data_frame(train_set)
    test_set = pre_wrangle_data_frame(test_set)
    
    combined_set = [train_set,test_set]    
    features = ['Fare', 'Cabin', 'Age', 'Sex', 'Surname', 'Title']
    
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(combined_set[feature])
        train_set[feature] = le.transform(train_set[feature])
        test_set[feature] = le.transform(test_set[feature])
    return train_set, test_set

def split_data(full_set):
    
    seed = 12345
    
    rnd.seed(seed)
    train_set = full_set.select(lambda x: rnd.uniform(0,1) >= 0.6)
    rnd.seed(seed)
    cv_set = full_set.select(lambda x: [0.2 <= r < 0.6 for r in [rnd.uniform(0,1)]][0])
    rnd.seed(seed)
    test_set = full_set.select(lambda x: rnd.uniform(0,1) < 0.2)
    
    return train_set, cv_set, test_set