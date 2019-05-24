#For this assignment, create a function that trains a model to predict blight 
#ticket compliance in Detroit using readonly/train.csv. Using this model, return 
#a series of length 61001 with the data being the probability that each corresponding 
#ticket from readonly/test.csv will be paid, and the index being the ticket_id.

def blight_model():
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import LabelEncoder
    #import warnings
    #warnings.filterwarnings("ignore")
    
    TrainSet = pd.read_csv("train.csv", encoding = "ISO-8859-1") #File to large to upload onto GitHub
    TestSet = pd.read_csv("test.csv")
    Addresses = pd.read_csv("addresses.csv")
    LatLon = pd.read_csv("latlons.csv") 
    
    #mapping from ticket id to addresses, and from addresses to lat/lon coordinates. 
    Train01 = pd.merge(left = TrainSet, right = Addresses, how = "left", left_on = "ticket_id", right_on = "ticket_id")
    Test01 = pd.merge(left = TestSet, right = Addresses, how = "left", left_on = "ticket_id", right_on = "ticket_id")
    Train02 = pd.merge(left = Train01, right = LatLon, how = "left", left_on = "address", right_on = "address")
    Test02 = pd.merge(left = Test01, right = LatLon, how = "left", left_on = "address", right_on = "address")
    
    Train02 = Train02[['ticket_id', 'agency_name', 'inspector_name', 'violator_name',
       'violation_street_number', 'violation_street_name',
       'violation_zip_code', 'mailing_address_str_number',
       'mailing_address_str_name', 'city', 'state', 'zip_code',
       'non_us_str_code', 'country', 'ticket_issued_date', 'hearing_date',
       'violation_code', 'violation_description', 'disposition', 'fine_amount',
       'admin_fee', 'state_fee', 'late_fee', 'discount_amount',
       'clean_up_cost', 'judgment_amount', 'grafitti_status', 'address', 'lat',
       'lon', 'compliance']]
    
    #Removing nulls
    Train02 = Train02.dropna(axis = 0, subset=Train02.columns[[-1]], how='any')
    
    #Data for USA only
    Train02 = Train02.where(Train02["country"]=="USA")
    Test02 = Test02.where(Test02["country"]=="USA")
    
    Train02 = Train02[['ticket_id',
       'violation_code', 'disposition', 'fine_amount',
       'admin_fee', 'state_fee', 'late_fee', 'discount_amount',
       'clean_up_cost', 'judgment_amount', 'lat',
       'lon', 'compliance']]
    Test02 = Test02[['ticket_id',
       'violation_code', 'disposition', 'fine_amount',
       'admin_fee', 'state_fee', 'late_fee', 'discount_amount',
       'clean_up_cost', 'judgment_amount', 'lat',
       'lon']]
    
    Train02["disposition"] = Train02["disposition"].astype(str)
    Test02["disposition"] = Test02["disposition"].astype(str)
    Train02['violation_code'] = Train02['violation_code'].astype(str)
    Test02['violation_code'] = Test02['violation_code'].astype(str)
    
    label_encoder = LabelEncoder()
    label_encoder.fit(Train02['disposition'].append(Test02["disposition"], ignore_index=True))
    Train02['disposition'] = label_encoder.transform(Train02['disposition'])
    Test02['disposition'] = label_encoder.transform(Test02['disposition'])
    
    label_encoder = LabelEncoder()
    label_encoder.fit(Train02['violation_code'].append(Test02['violation_code'], ignore_index=True))
    Train02['violation_code'] = label_encoder.transform(Train02['violation_code'])
    Test02['violation_code'] = label_encoder.transform(Test02['violation_code'])
    
    #Filling nulls
    TrainLatMean = Train02["lat"].mean()
    TrainLonMean = Train02["lon"].mean()
    TestLatMean = Test02["lat"].mean()
    TestLonMean = Test02["lon"].mean()
    
    Train02["lat"].fillna(TrainLatMean, inplace = True)
    Train02["lon"].fillna(TrainLonMean, inplace = True)
    Test02["lat"].fillna(TestLatMean, inplace = True)
    Test02["lon"].fillna(TestLonMean, inplace = True)
    
    #Removing remaining nulls
    Train02 = Train02.dropna(how = "any")
    
    #Running model
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(Train02.iloc[:, Train02.columns != 'compliance'], Train02['compliance'])
    
    X_test = Test02[X_train.columns]
    
    from sklearn.ensemble import RandomForestRegressor
    classifier = RandomForestRegressor(n_estimators = 100, max_depth = 30)

    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    DF = pd.DataFrame(y_pred, X_test["ticket_id"])

    return DF

blight_model()