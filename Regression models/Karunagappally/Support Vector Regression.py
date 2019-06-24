# Support Vector Machine (SVM)

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
datasetinitial = pd.read_csv('Karunagappally Regression.csv')

#Delete missing values in dataframe
dataset = datasetinitial[np.isfinite(datasetinitial['Available Warehouse capacity'])]

#Take numerical values
X = dataset.loc[:, ['Available Warehouse capacity','Number of wagons placed',
                    'Arrival time of wagons']].values
y = dataset.loc[:, 'Demurrage Hours'].values

#Take categorical variables
X_binary=dataset.loc[:,['Sun','Mon','Tue','Wed','Thu','Fri','Working']].values

#Finding mean & SD for numerical values of X
df=pd.DataFrame(X)
WCM=df.loc[:,0].mean()
WCSD=df.loc[:,0].std()
WPM=df.loc[:,1].mean()
WPSD=df.loc[:,1].std()
ATM=df.loc[:,2].mean()
ATSD=df.loc[:,2].std()

# Feature Scaling (standardise)
from sklearn import preprocessing
X = preprocessing.scale(X)
y = preprocessing.scale(y)

#Input matrix containing standarised numerical and categorical values
X=np.concatenate((X_binary,X),axis=1)

"""
X column name after fitting regression
0:Ones, 1:Sun,2:Mon,3:Tue,4:Wed,5:Thu,6:Fri,7:Working,
8:Available Warehouse capacity,9:Number of wagons placed,
10:Arrival time of wagons
"""

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, 
                                                    random_state = 0)

# Fitting SVM to the Training set
from sklearn.svm import SVR

# Entering the parameters of algorithm
C=0.01
gam=0.01
epsi=0

regressor = SVR(kernel = 'rbf', C=C, gamma=gam, epsilon=epsi)
regressor.fit(X_train, y_train)

#No of folds for cross-validation used in Cross-validation and Grid Search
CV=10

# Find the best parameters using Grid Search
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [0.01,0.05,0.1,0.25,0.5,0.75,1,1.5,2.5],'epsilon': [0.1,0.5,1,2,3], 'kernel': ['rbf'],
    'gamma': [0.01,0.02,0.025,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.25,0.3,0.4,0.5,0.75,0.9]}]
grid_search = GridSearchCV(estimator=regressor, param_grid=parameters, scoring='neg_mean_squared_error',cv=CV)
grid_search = grid_search.fit(X_train, y_train)
means = grid_search.cv_results_['mean_test_score']
stds = grid_search.cv_results_['std_test_score']
print("\nMean & Standard deviation of MSE value for different parameters using "+str(CV)+"-fold cross-validation:")
for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
    print("\tMean=%0.3f & Standard deviation=%0.3f for %r" % (mean, std, params))
print("\nBest parameters set found on development set:")
best_accuracy=grid_search.best_score_
print("Best MSE value obtained using "+str(CV)+"-fold cross-validation: %0.3f "%(best_accuracy))
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