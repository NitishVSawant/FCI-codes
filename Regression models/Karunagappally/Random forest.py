# Importing the libraries
import numpy as np
import pandas as pd
import statsmodels.formula.api as sm

# Importing the dataset
datasetinitial = pd.read_csv('Karunagappally Regression.csv')

#Delete missing values in dataframe
dataset = datasetinitial[np.isfinite(datasetinitial['Available Warehouse capacity'])]

#Take numerical values
X = dataset.loc[:, ['Sun','Mon','Tue','Wed','Thu','Fri','Working',
                    'Available Warehouse capacity','Number of wagons placed',
                    'Arrival time of wagons']].values
y = dataset.loc[:, 'Demurrage Hours'].values

"""
X column name after fitting regression
0:Ones, 1:Sun,2:Mon,3:Tue,4:Wed,5:Thu,6:Fri,7:Working,
8:Available Warehouse capacity,9:Number of wagons placed,
10:Arrival time of wagons
"""

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, 
                                                    random_state = 3)

# Fitting Random Forest Regression to the Training set
from sklearn.ensemble import RandomForestRegressor

# Entering the parameters of algorithm
estimators=7

regressor = RandomForestRegressor(n_estimators = estimators, max_depth=3, random_state = 0)
regressor.fit(X_train, y_train)

#No of folds for cross-validation used in Cross-validation and Grid Search
CV=10

# Find the best parameters using Grid Search
from sklearn.model_selection import GridSearchCV
parameters = [{'n_estimators': [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}]
grid_search = GridSearchCV(estimator=regressor, param_grid=parameters, scoring='r2',cv=CV)
grid_search = grid_search.fit(X_train, y_train)
means = grid_search.cv_results_['mean_test_score']
stds = grid_search.cv_results_['std_test_score']
print("\nMean & Standard deviation of R-squared value for different parameters using "+str(CV)+"-fold cross-validation:")
for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
    print("\tMean=%0.3f & Standard deviation=%0.3f for %r" % (mean, std, params))
print("\nBest parameters set found on development set:")
best_accuracy=grid_search.best_score_
print("Best R-squared value obtained using "+str(CV)+"-fold cross-validation: %0.3f "%(best_accuracy))
print("\t"+str(grid_search.best_params_))

# Using best model obtained after tuning
regressor = regressor.set_params(**grid_search.best_params_)
regressor.fit(X_train, y_train)

# Predicting the Training set results
y_pred_train = regressor.predict(X_train)

# Predicting the Test set results
y_pred_test = regressor.predict(X_test)

#Mean square error for training and test set
from sklearn.metrics import mean_squared_error
print('Mean square error for training' 
      ' set is {}'.format(mean_squared_error(y_train, y_pred_train)))
print('Mean square error for test' 
      ' set is {}'.format(mean_squared_error(y_test, y_pred_test)))

# Program for input parameters
"""
X column name
X1:Sun,X2:Mon,X3:Tue,X4:Wed,X5:Thu,X6:Fri,X7:Working,
X8:Available Warehouse capacity,X9:Number of wagons placed,
X10:Arrival time of wagons
"""
print("\nEnter the following data: ")
X8=float(input("Enter the available warehouse capacity: "))
X9=float(input("Enter the number of wagons placed: "))
X10=float(input("Enter the arrival time of wagons in 24 hour clock format: "))
print("Consider the format for day of arrival: ")
print("\tSunday-SUN,\tMonday-MON,\tTuesday-TUE,\tWednesday-WED,"
      "\tThursday-THU,\tFriday-FRI,\tSaturday-SAT")
Day=input("Enter the day of arrival: ").upper()
if(Day=="SUN"):
    X1=1
    X2=X3=X4=X5=X6=0
    X7=0
if(Day=="MON"):
    X2=1
    X1=X3=X4=X5=X6=0
    X7=1
if(Day=="TUE"):
    X3=1
    X1=X2=X4=X5=X6=0
    X7=1
if(Day=="WED"):
    X4=1
    X1=X2=X3=X5=X6=0
    X7=1
if(Day=="THU"):
    X5=1
    X1=X2=X3=X4=X6=0
    X7=1
if(Day=="FRI"):
    X6=1
    X1=X2=X3=X4=X5=0
    X7=1
if(Day=="SAT"):
    X1=X2=X3=X4=X5=X6=0
    X7=0
    
# Prediction for demurrage cost
if(regressor.predict([[X1,X2,X3,X4,X5,X6,X7,X8,X9,X10]]))<=0:
    print('\n\tDemurrage Cost will not occur')
else:
    k=regressor.predict([[X1,X2,X3,X4,X5,X6,X7,X8,X9,X10]])
    print("\n\tDemurrage Cost will occur and the demurrage "
          "hours would be {} hours".format(np.round(k,1)))