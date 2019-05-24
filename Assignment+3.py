#Import the data from fraud_data.csv. What percentage of the observations in the 
#dataset are instances of fraud?
#This function should return a float between 0 and 1.
def answer_one():
    
    import numpy as np
    import pandas as pd

    Fraud = pd.read_csv("fraud_data.csv")
    Counts = pd.value_counts(Fraud["Class"].values, sort=False)
    NotFraud = Counts[0]
    IsFraud = Counts[1]
    TotalCounts = NotFraud + IsFraud
    Perc_Fraud = (IsFraud / TotalCounts)

    
    return Perc_Fraud
answer_one()

#Using X_train, X_test, y_train, and y_test (as defined above), train a dummy 
#classifier that classifies everything as the majority class of the training data. 
#What is the accuracy of this classifier? What is the recall?
#This function should a return a tuple with two floats, i.e. (accuracy score, recall score).
def answer_two():
    # Use X_train, X_test, y_train, y_test for all of the following questions
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    df =  pd.read_csv("fraud_data.csv")

    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)   

    dummy_majority = DummyClassifier(strategy = 'most_frequent').fit(X_train, y_train)
    
    y_dummy_predictions = dummy_majority.predict(X_test)    
    
    acurracy = accuracy_score(y_test, y_dummy_predictions)
    
    recall = recall_score(y_test, y_dummy_predictions)
    
    Answer = (acurracy, recall)
    
    return Answer
answer_two()

#Using X_train, X_test, y_train, y_test (as defined above), train a SVC classifer 
#using the default parameters. What is the accuracy, recall, and precision of this classifier?
#This function should a return a tuple with three floats, i.e. (accuracy score, 
#recall score, precision score).
def answer_three():
    
    # Use X_train, X_test, y_train, y_test for all of the following questions
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    df = pd.read_csv('fraud_data.csv')

    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    svm = SVC().fit(X_train, y_train)
    svm_predictions = svm.predict(X_test)

    acurracy = (accuracy_score(y_test, svm_predictions))
    recall = (recall_score(y_test, svm_predictions))
    precision = precision_score(y_test, svm_predictions)
    
    Answer = (acurracy, recall, precision)
    
    return Answer
answer_three()

#Using the SVC classifier with parameters {'C': 1e9, 'gamma': 1e-07}, what is 
#the confusion matrix when using a threshold of -220 on the decision function. 
#Use X_test and y_test.
#This function should return a confusion matrix, a 2x2 numpy array with 4 integers.
def answer_four():
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import confusion_matrix
    
    df = pd.read_csv('fraud_data.csv')

    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    svm = SVC(C = 1e9, gamma = 1e-07).fit(X_train, y_train)
    svm_predictions = svm.decision_function(X_test) > -220
    confusion2 = confusion_matrix(y_test, svm_predictions)
    
    
    return confusion2
answer_four()

#Train a logisitic regression classifier with default parameters using X_train and y_train.
#For the logisitic regression classifier, create a precision recall curve and a 
#roc curve using y_test and the probability estimates for X_test (probability it is fraud).
#Looking at the precision recall curve, what is the recall when the precision is 0.75?
#Looking at the roc curve, what is the true positive rate when the false positive rate is 0.16?
#This function should return a tuple with two floats, i.e. (recall, true positive rate).
def answer_five():
        
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import roc_curve, auc
    
    df = pd.read_csv("fraud_data.csv")

    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    lr = LogisticRegression()
    y_proba_lr = lr.fit(X_train, y_train).predict_proba(X_test)
    y_scores_lr = lr.fit(X_train, y_train).decision_function(X_test)
    
    precision, recall, thresholds = precision_recall_curve(y_test, y_scores_lr)
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_scores_lr)
    
    recall = float(recall[np.where(precision==0.75)])
    true_positive_rate = float(tpr_lr[np.where((fpr_lr >= 0.159) & (fpr_lr <= 0.161))][0])
    Answer = (recall, true_positive_rate)
    return Answer
answer_five()

#Perform a grid search over the parameters listed below for a Logisitic Regression 
#classifier, using recall for scoring and the default 3-fold cross validation.
#'penalty': ['l1', 'l2']
#'C':[0.01, 0.1, 1, 10, 100]
#From .cv_results_, create an array of the mean test scores of each parameter combination. i.e.
#| | l1 | l2 | |:----: |---- |---- | | 0.01 | ? | ? | | 0.1 | ? | ? | | 1 | ? | ? | | 10 | ? | ? | | 100 | ? | ? |
#This function should return a 5 by 2 numpy array with 10 floats.
#Note: do not return a DataFrame, just the values denoted by '?' above in a numpy 
#array. You might need to reshape your raw result to meet the format we are looking for.
def answer_six():    
    
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    
    df = pd.read_csv("fraud_data.csv")

    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    clf = LogisticRegression()
    
    grid_values = {'penalty': ['l1', 'l2'], 'C':[0.01, 0.1, 1, 10, 100]}
    
    
    grid_clf_recall = GridSearchCV(clf, param_grid = grid_values, scoring = 'recall')
    grid_clf_recall.fit(X_train, y_train)
    output = grid_clf_recall.cv_results_["mean_test_score"].reshape(5,2)  

    return output
answer_six()

# Use the following function to help visualize results from the grid search
def GridSearch_Heatmap(scores):
    %matplotlib notebook
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.figure()
    sns.heatmap(scores.reshape(5,2), xticklabels=['l1','l2'], yticklabels=[0.01, 0.1, 1, 10, 100])
    plt.yticks(rotation=0);

GridSearch_Heatmap(answer_six())