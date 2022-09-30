
import pandas as pd
import numpy as np
import re

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score, train_test_split


# Function that reads the given .txt file and converts it into a pandas dataframe

def import_data(filename):

    # Store all text in a variable (string)
    with open(filename, 'r') as f:
        data = f.read()
    
    # Convert string to a single line and remove noise
    data = data.replace('\n', '')
    data = data.replace('[1]', '')
    data = re.sub(' "Record [0-9]+"', '', data)
    data = data[1:]
    
    # Split the data string into data points
    # Array split the result into 9868 rows
    data = np.array(data.split(' '))
    data = np.array_split(data, 9868)

    # Create pandas dataframe from rows
    df = pd.DataFrame(data, columns=['id', 'duration', 'sex', 'age', 'num_children', 'country', 'acc_type'])
    df = df.replace('<NA>', np.nan)

    return(df)

def clean_data(df):

    # Convert numerical variables
    df.duration = pd.to_numeric(df.duration)
    df.age = pd.to_numeric(df.age)
    df.num_children = pd.to_numeric(df.num_children)

    # Convert binary variables
    df.sex = [1 if sex == 'M' else 0 for sex in df.sex]
    df.acc_type = [1 if acc == 'Apt' else 0 for acc in df.acc_type]

    # Add dummy variables for country
    country_dummies = pd.get_dummies(df.country)
    df = pd.concat([df, country_dummies], axis=1)

    # Drop unnecessary variables
    df = df.drop(['id', 'country'], axis=1)

    return(df)

# Function that prepares the cleaned data for model training
def preprocess_data(df):
    
    #Split data into train and target variables
    y = df['acc_type']
    X = df.drop('acc_type', axis=1)

    #Impute missing data using mean strategy
    imp = SimpleImputer(strategy='mean')
    X = imp.fit_transform(X)

    #Scale data using min-max approach
    mms = MinMaxScaler()
    X = mms.fit_transform(X)

    return(X, y)

if __name__ == '__main__':
    
    df = import_data('train_data.txt')
    df = clean_data(df)
    print(df.head())

    X, y = preprocess_data(df)
    print(X.shape, y.shape)


 