#For this assignment, you will be using the Breast Cancer Wisconsin (Diagnostic) 
#Database to create a classifier that can help diagnose patients. 

#Convert the sklearn.dataset cancer to a DataFrame.
def answer_one():
    import numpy as np
    import pandas as pd
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    cancer = load_breast_cancer()
    cancer.keys()
    data = np.c_[cancer.data, cancer.target]
    columns = np.append(cancer.feature_names, ["target"])
    DF = pd.DataFrame(data, columns=columns)
    DF["target"] = DF["target"].astype(int)
    return DF
answer_one()

#What is the class distribution? (i.e. how many instances of malignant (encoded 0) 
#and how many benign (encoded 1)?)
#This function should return a Series named target of length 2 with integer values 
#and index = ['malignant', 'benign']
def answer_two():
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    DF = answer_one()
    Targetdict = {0 : "malignant", 1: "benign"}
    Target = pd.Series(DF["target"].astype(int))
    Target = Target.map(Targetdict, na_action=None)
    Answer = Target.value_counts()
    return Answer
answer_two()

#Split the DataFrame into X (the data) and y (the labels).
#This function should return a tuple of length 2: (X, y), where
#X, a pandas DataFrame, has shape (569, 30)
#y, a pandas Series, has shape (569,).
def answer_three():
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    DF = answer_one()
    X = DF.loc[:, "mean radius":"worst fractal dimension"]
    y = DF["target"]
    return X, y
answer_three()

#Using train_test_split, split X and y into training and test sets (X_train, X_test, 
#y_train, and y_test).
#Set the random number generator state to 0 using random_state=0 to make sure 
#your results match the autograder!
#This function should return a tuple of length 4: (X_train, X_test, y_train, y_test), 
#where
#X_train has shape (426, 30)
#X_test has shape (143, 30)
#y_train has shape (426,)
#y_test has shape (143,)
def answer_four():
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    X, y = answer_three()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    return X_train, X_test, y_train, y_test
answer_four()

#Using KNeighborsClassifier, fit a k-nearest neighbors (knn) classifier with X_train, 
#y_train and using one nearest neighbor (n_neighbors = 1).
#*This function should return a * sklearn.neighbors.classification.KNeighborsClassifier.
def answer_five():
    import numpy as np
    import pandas as pd
    from sklearn.neighbors import KNeighborsClassifier
    X_train, X_test, y_train, y_test = answer_four()
    knn = KNeighborsClassifier(n_neighbors = 1)
    Answer = knn.fit(X_train, y_train)
    return Answer
answer_five()

#Using your knn classifier, predict the class label using the mean value for each feature.
#Hint: You can use cancerdf.mean()[:-1].values.reshape(1, -1) which gets the mean 
#value for each feature, ignores the target column, and reshapes the data from 1 
#dimension to 2 (necessary for the precict method of KNeighborsClassifier).
#This function should return a numpy array either array([ 0.]) or array([ 1.])
def answer_six():
    import numpy as np
    import pandas as pd
    from sklearn.neighbors import KNeighborsClassifier
    DF = answer_one()
    knn = answer_five()
    means = DF.mean()[:-1].values.reshape(1, -1)
    cancer_prediction = knn.predict(means) 
    return cancer_prediction
answer_six()

#Using your knn classifier, predict the class labels for the test set X_test.
#This function should return a numpy array with shape (143,) and values either 0.0 or 1.0.
def answer_seven():
    import numpy as np
    import pandas as pd
    from sklearn.neighbors import KNeighborsClassifier
    X_train, X_test, y_train, y_test = answer_four()
    knn = answer_five()
    y_pred = knn.predict(X_test)    
    return y_pred
answer_seven()

#Find the score (mean accuracy) of your knn classifier using X_test and y_test.
#This function should return a float between 0 and 1
def answer_eight():
    import numpy as np
    import pandas as pd
    from sklearn.neighbors import KNeighborsClassifier
    X_train, X_test, y_train, y_test = answer_four()
    knn = answer_five()
    Answer = knn.score(X_test, y_test)    
    return Answer
answer_eight()

