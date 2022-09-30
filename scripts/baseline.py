from preprocess import import_data, clean_data, preprocess_data

import numpy as np

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report, roc_curve, RocCurveDisplay, precision_recall_curve, PrecisionRecallDisplay

from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import seaborn as sns



def evaluate_model(y_hat, y_test, pred_proba, name=""):

    print('MODEL: {}'.format(name))

    # sklearn classification
    print(classification_report(y_hat, y_test))

    # Confusion matrix
    fig = plt.figure()
    cf = confusion_matrix(y_hat, y_test)
    sns.heatmap(cf, annot=True, fmt="d")
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion matrix ({})'.format(name))

    # Visualize ROC and PR curves
    fpr, tpr, _ = roc_curve(y_test, pred_proba, pos_label=1)
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr)

    prec, rec, _ = precision_recall_curve(y_test, pred_proba, pos_label=1)
    pr_display = PrecisionRecallDisplay(precision=prec, recall=rec)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

    roc_display.plot(ax=ax1)
    pr_display.plot(ax=ax2)

    plt.suptitle('ROC and PR curves ({})'.format(name))
    plt.show()
    


if __name__ == '__main__':

    sns.set_theme()

    # Store preprocessed input and target variables
    df = import_data('train_data.txt')
    df = clean_data(df)
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
    
    # Random forest cross validation and overall model evaluation
    rf = RandomForestClassifier(random_state=42)
    rf_scores = cross_val_score(rf, X, y, cv=5)

    # Random forest for evaluation function
    rf = RandomForestClassifier(random_state = 2)
    rf.fit(X_train, y_train)

    y_hat = rf.predict(X_test)
    pred_proba = [i[1] for i in rf.predict_proba(X_test)]

    evaluate_model(y_hat, y_test, pred_proba, name="Random Forest")
    print("Random Forest accuracy: {}".format(np.mean(rf_scores)))



    



