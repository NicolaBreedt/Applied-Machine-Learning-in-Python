#Write a function that fits a polynomial LinearRegression model on the training 
#data X_train for degrees 1, 3, 6, and 9. (Use PolynomialFeatures in sklearn.preprocessing 
#to create the polynomial features and then fit a linear regression model) For each 
#model, find 100 predicted values over the interval x = 0 to 10 (e.g. np.linspace(0,10,100))
# and store this in a numpy array. The first row of this array should correspond to the output 
#from the model trained on degree 1, the second row degree 3, the third row degree 6, and 
#the fourth row degree 9.
#*This function should return a numpy array with shape `(4, 100)`*
def answer_one():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures


    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split


    np.random.seed(0)
    n = 15
    x = np.linspace(0,10,n) + np.random.randn(n)/5
    y = np.sin(x)+x/6 + np.random.randn(n)/10


    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)
    X_train = X_train.reshape((11,1))
    X_predict_input = np.linspace(0,10,100).reshape(100,1)

    #POLY 1
    poly1 = PolynomialFeatures(degree=1)
    X_train_poly1 = poly1.fit_transform(X_train)

    linreg = LinearRegression().fit(X_train_poly1, y_train)

    X_predict_input_poly1 = poly1.fit_transform(X_predict_input)

    y_predict_output_poly1 = linreg.predict(X_predict_input_poly1)
    y_predict_output_poly1 = y_predict_output_poly1.flatten()

    #POLY 3
    poly3 = PolynomialFeatures(degree=3)
    X_train_poly3 = poly3.fit_transform(X_train)

    linreg = LinearRegression().fit(X_train_poly3, y_train)

    X_predict_input_poly3 = poly3.fit_transform(X_predict_input)

    y_predict_output_poly3 = linreg.predict(X_predict_input_poly3)
    y_predict_output_poly3 = y_predict_output_poly3.flatten()

    #POLY 6
    poly6 = PolynomialFeatures(degree=6)
    X_train_poly6 = poly6.fit_transform(X_train)

    linreg = LinearRegression().fit(X_train_poly6, y_train)

    X_predict_input_poly6 = poly6.fit_transform(X_predict_input)

    y_predict_output_poly6 = linreg.predict(X_predict_input_poly6)
    y_predict_output_poly6 = y_predict_output_poly6.flatten()

    #POLY 9
    poly9 = PolynomialFeatures(degree=9)
    X_train_poly9 = poly9.fit_transform(X_train)

    linreg = LinearRegression().fit(X_train_poly9, y_train)

    X_predict_input_poly9 = poly9.fit_transform(X_predict_input)

    y_predict_output_poly9 = linreg.predict(X_predict_input_poly9)
    y_predict_output_poly9 = y_predict_output_poly9.flatten()

    Answer = np.vstack([y_predict_output_poly1, y_predict_output_poly3, y_predict_output_poly6, y_predict_output_poly9])
    
    return Answer
answer_one()

#Write a function that fits a polynomial LinearRegression model on the training 
#data X_train for degrees 0 through 9. For each model compute the 𝑅2 (coefficient 
#of determination) regression score on the training data as well as the the test
# data, and return both of these arrays in a tuple.
#This function should return one tuple of numpy arrays (r2_train, r2_test). 
#Both arrays should have shape (10,)
def answer_two():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics.regression import r2_score

    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split


    np.random.seed(0)
    n = 15
    x = np.linspace(0,10,n) + np.random.randn(n)/5
    y = np.sin(x)+x/6 + np.random.randn(n)/10


    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)

    X_train = X_train.reshape((11,1))
    X_test = X_test.reshape((4,1))

    R_train = []
    R_test = []

    for i in range(0,10):
        poly = PolynomialFeatures(degree=i)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.fit_transform(X_test)
    
        linreg = LinearRegression().fit(X_train_poly, y_train)
    
        a = linreg.score(X_train_poly, y_train)
        R_train.append(a)
    
        b = linreg.score(X_test_poly, y_test)
        R_test.append(b)

    Answer = (R_train, R_test)
    return Answer
answer_two()

#Based on the 𝑅2 scores from question 2 (degree levels 0 through 9), what degree 
#level corresponds to a model that is underfitting? What degree level corresponds 
#to a model that is overfitting? What choice of degree level would provide a model 
#with good generalization performance on this dataset?
#Hint: Try plotting the 𝑅2 scores from question 2 to visualize the relationship 
#between degree level and 𝑅2. Remember to comment out the import matplotlib line 
#before submission.
#This function should return one tuple with the degree values in this order:
#(Underfitting, Overfitting, Good_Generalization). There might be multiple correct 
#solutions, however, you only need to return one possible solution, for example, (1,2,3).
def answer_three():
    
    x1 = answer_two()[0]
    x2 = answer_two()[1]
    underfitted = []
    overfitted = []
    good_generalization = []
    for i,j,k in zip(x1,x2,(0,1,2,3,4,5,6,7,8,9)):
        if (i < 0.5) and (j < 0.5):
            a = k
            underfitted.append(a)
        elif (i > 0.9) and (j < 0.5):
            b = k
            overfitted.append(b)
        elif (i > 0.9) and (j > 0.9):
            c = k
            good_generalization.append(c)
        else:
            d = k

    answer = (underfitted[0], overfitted[0], good_generalization[0])

    
    return answer
answer_three()

#Training models on high degree polynomial features can result in overly complex 
#models that overfit, so we often use regularized versions of the model to constrain 
#model complexity, as we saw with Ridge and Lasso linear regression.
#For this question, train two models: a non-regularized LinearRegression model 
#(default parameters) and a regularized Lasso Regression model (with parameters 
#alpha=0.01, max_iter=10000) both on polynomial features of degree 12. Return 
#the 𝑅2 score for both the LinearRegression and Lasso model's test sets.
#This function should return one tuple (LinearRegression_R2_test_score, Lasso_R2_test_score)
def answer_four():
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Lasso, LinearRegression
    from sklearn.metrics.regression import r2_score

    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split


    np.random.seed(0)
    n = 15
    x = np.linspace(0,10,n) + np.random.randn(n)/5
    y = np.sin(x)+x/6 + np.random.randn(n)/10


    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)

    X_train = X_train.reshape((11,1))
    X_test = X_test.reshape((4,1))

    poly = PolynomialFeatures(degree=12)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.fit_transform(X_test)

# a non-regularized LinearRegression model (default parameters) on polynomial features of degree 12  
    linreg = LinearRegression().fit(X_train_poly, y_train)    
    LinearRegression_R2_test_score = linreg.score(X_test_poly, y_test)

# a regularized Lasso Regression model (with parameters alpha=0.01, max_iter=10000) on polynomial features of degree 12
    linlasso = Lasso(alpha=0.01, max_iter=10000).fit(X_train_poly, y_train)
    Lasso_R2_test_score = linlasso.score(X_test_poly, y_test)

    answer = (LinearRegression_R2_test_score, Lasso_R2_test_score)


    return answer
answer_four()

#For this section of the assignment we will be working with the UCI Mushroom Data 
#Set stored in readonly/mushrooms.csv. The data will be used to train a model to 
#predict whether or not a mushroom is poisonous.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


mush_df = pd.read_csv('mushrooms.csv')
mush_df2 = pd.get_dummies(mush_df)

X_mush = mush_df2.iloc[:,2:]
y_mush = mush_df2.iloc[:,1]

# use the variables X_train2, y_train2 for Question 5
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_mush, y_mush, random_state=0)

# For performance reasons in Questions 6 and 7, we will create a smaller version of the
# entire mushroom dataset for use in those questions.  For simplicity we'll just re-use
# the 25% test split created above as the representative subset.
#
# Use the variables X_subset, y_subset for Questions 6 and 7.
X_subset = X_test2
y_subset = y_test2

#Using X_train2 and y_train2 from the preceeding cell, train a DecisionTreeClassifier 
#with default parameters and random_state=0. What are the 5 most important features 
#found by the decision tree?
#This function should return a list of length 5 containing the feature names in descending order of importance.
#Note: remember that you also need to set random_state in the DecisionTreeClassifier.
def answer_five():
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split


    mush_df = pd.read_csv('mushrooms.csv')
    mush_df2 = pd.get_dummies(mush_df)
    mush_df2 = mush_df2[:].astype(int)

    X_mush = mush_df2.iloc[:,2:]
    y_mush = mush_df2.iloc[:,1]

# use the variables X_train2, y_train2 for Question 5
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X_mush, y_mush, random_state=0)

# For performance reasons in Questions 6 and 7, we will create a smaller version of the
# entire mushroom dataset for use in those questions.  For simplicity we'll just re-use
# the 25% test split created above as the representative subset.
#
# Use the variables X_subset, y_subset for Questions 6 and 7.
    X_subset = X_test2
    y_subset = y_test2

    from sklearn.tree import DecisionTreeClassifier

    clf = DecisionTreeClassifier(random_state = 0).fit(X_train2, y_train2)
    
    Importance = clf.feature_importances_
    Feature = mush_df2.columns[2:]
    
    NewDF = pd.DataFrame({"Feature": Feature, "Importance": Importance})
    NewDF = NewDF.sort_values(by = "Importance", ascending = False)
    answer = list(NewDF["Feature"].iloc[0:5])

    return answer
answer_five()

#For this question, we're going to use the validation_curve function in sklearn.model_selection 
#to determine training and test scores for a Support Vector Classifier (SVC) with 
#varying parameter values. 
#With this classifier, and the dataset in X_subset, y_subset, explore the effect 
#of gamma on classifier accuracy by using the validation_curve function to find 
#the training and test scores for 6 values of gamma from 0.0001 to 10 (i.e. np.logspace(-4,1,6)).
#For each level of gamma, validation_curve will fit 3 models on different subsets 
#of the data, returning two 6x3 (6 levels of gamma x 3 fits per level) arrays of 
#the scores for the training and test sets.
#Find the mean score across the three models for each level of gamma for both arrays, 
#creating two arrays of length 6, and return a tuple with the two arrays.
#This function should return one tuple of numpy arrays (training_scores, test_scores) 
#where each array in the tuple has shape (6,).
def answer_six():

    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split

    mush_df = pd.read_csv('mushrooms.csv')
    mush_df2 = pd.get_dummies(mush_df)

    X_mush = mush_df2.iloc[:,2:]
    y_mush = mush_df2.iloc[:,1]


    X_train2, X_test2, y_train2, y_test2 = train_test_split(X_mush, y_mush, random_state=0)

# For performance reasons in Questions 6 and 7, we will create a smaller version of the
# entire mushroom dataset for use in those questions.  For simplicity we'll just re-use
# the 25% test split created above as the representative subset.
#
# Use the variables X_subset, y_subset for Questions 6 and 7.
    X_subset = X_test2
    y_subset = y_test2

    from sklearn.svm import SVC
    from sklearn.model_selection import validation_curve

#create an SVC object with default parameters (i.e. kernel='rbf', C=1) and random_state=0.
    clf = SVC(random_state=0, kernel = 'rbf', C=1).fit(X_subset, y_subset)

#explore the effect of gamma on classifier accuracy by using the validation_curve function 
#to find the training and test scores for 6 values of gamma from 0.0001 to 10 (i.e. np.logspace(-4,1,6))
    Param_Range = np.logspace(-4,1,6)
    train_scores, test_scores = validation_curve(SVC(random_state=0, kernel = 'rbf', C=1), X_subset, y_subset, param_name='gamma', param_range=Param_Range, cv=3, scoring="accuracy")
    
    training_scores = []
    for i in (0,1,2,3,4,5):
        x = train_scores[i].mean()
        training_scores.append(x)

    testing_scores = []
    for i in (0,1,2,3,4,5):
        x = test_scores[i].mean()
        testing_scores.append(x)

    answer = (training_scores, testing_scores)

    return answer
answer_six()

#Based on the scores from question 6, what gamma value corresponds to a model 
#that is underfitting (and has the worst test set accuracy)? What gamma value 
#corresponds to a model that is overfitting (and has the worst test set accuracy)?
#What choice of gamma would be the best choice for a model with good generalization 
#performance on this dataset (high accuracy on both training and test set)?
#This function should return one tuple with the degree values in this order: 
#(Underfitting, Overfitting, Good_Generalization) Please note there is only one correct solution.
def answer_seven():
    import numpy as np 
    
    x1 = answer_six()[0]
    x2 = answer_six()[1]
    Param_Range = np.logspace(-4,1,6)
    
    underfitted = []
    overfitted = []
    good_generalization = []
    
    for i,j,k in zip(x1,x2,Param_Range):
        if (i < 0.6) and (j < 0.6):
            a = k
            underfitted.append(a)
        elif (i > 0.9) and (j < 0.6):
            b = k
            overfitted.append(b)
        elif (i == j == 1):
            c = k
            good_generalization.append(c)
        else:
            d = k
    answer = (underfitted[0], overfitted[0], good_generalization[0])
    return answer
answer_seven()
