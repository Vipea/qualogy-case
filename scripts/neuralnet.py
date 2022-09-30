
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential


from preprocess import import_data, clean_data, preprocess_data
from baseline import evaluate_model

# Function that builds and compiles a simple ann
def build_model():

    model = Sequential()
    model.add(Dense(256, activation='relu', input_dim=12))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return(model)

if __name__ == '__main__':

    sns.set_theme()

    # Store preprocessed input and target variables
    df = import_data('train_data.txt')
    df = clean_data(df)
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
    
    # Build and fit model
    ann = build_model()    
    hist = ann.fit(X_train, y_train, epochs=1, batch_size=40, validation_split=0.2)

    # Evaluation methods
    scores = ann.evaluate(X_test, y_test)
    print("loss: {}, accuracy: {}".format(scores[0], scores[1]))

    pred_proba = ann.predict(X_test)
    y_hat = pred_proba > 0.5


    evaluate_model(y_hat, y_test, pred_proba, name="Neural network")

    # Learning rate visualization
    hist = hist.history

    plt.figure()
    plt.plot(hist['accuracy'], '--', label='train_accuracy')
    plt.plot(hist['val_accuracy'], label='val_accuracy')
    plt.ylim(0.5, 0.9)
    plt.legend()

    plt.show()

    # Train ANN on entire training set
    ann_full = build_model()
    ann_full.fit(X, y, epochs=40, batch_size =10)

    # Prepare test data for making predictions
    df = pd.read_csv('TestDataAccomodation.csv')
    df.columns = ['id', 'duration', 'sex', 'age', 'num_children', 'country', 'acc_type']

    df = clean_data(df)
    X_test, _ = preprocess_data(df)

    #Make final prediction and store the predictions in predictions.csv
    y_hat = ann_full.predict(X_test) > 0.5

    df = pd.read_csv('TestDataAccomodation.csv')
    df['acc_type_pred'] = ['Apt' if i else 'Hotel' for i in y_hat]

    df.to_csv('predictions.csv')



