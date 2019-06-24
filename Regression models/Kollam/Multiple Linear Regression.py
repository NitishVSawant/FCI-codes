# Multiple linear regression

# Importing the libraries
import numpy as np
import pandas as pd
import statsmodels.formula.api as sm

# Importing the dataset
dataset = pd.read_csv('Kollam Regression.csv')

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
                                                    random_state =1)

#Adding column with ones needed for backward elimination
X_train = np.append(arr = np.ones((X_train.shape[0], 1)).astype(int), 
                    values = X_train, axis = 1)

#Backward Elimination with p-values
def backwardElimination(X, y, SL):
    features = np.arange(0, X.shape[1])
    do = 1
    while (do == 1):
        X_opt = X[:, features]
        regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
        pvalues = regressor_OLS.pvalues
        max_pvalue_value = pvalues.max()
        max_pvalue_index = pvalues.argmax()
        
        if max_pvalue_value > SL:
            prev_features = features
            features = np.delete(features, max_pvalue_index)
        else:
            do = 0
    return features

#Significance level
SL = 0.05
a=backwardElimination(X_train, y_train, SL)
print("The indices of optimal team of independent variables is {}".format(a))
regressor_OLS = sm.OLS(endog = y_train, exog = X_train[:,a]).fit()
print(regressor_OLS.summary())

#Removing the ones from X_train for fitting multiple regression model
if a[0]==0:
    a=np.delete(a,0)
    
# Fitting Multiple Linear Regression to the Training set after finding 
#optimal team of independent variables
X_mod=X_train[:, a]
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_mod, y_train)

# Normal Q-Q plot
from statsmodels.graphics.gofplots import ProbPlot
model_fit = sm.OLS(y_train, X_mod).fit()
model_norm_residuals = model_fit.get_influence().resid_studentized_internal
QQ = ProbPlot(model_norm_residuals)
plot_lm_2 = QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1)
plot_lm_2.set_figheight(8)
plot_lm_2.set_figwidth(12)
plot_lm_2.axes[0].set_title('Normal Q-Q Plot')
plot_lm_2.axes[0].set_xlabel('Theoretical Quantiles')
plot_lm_2.axes[0].set_ylabel('Standardized Residuals');

# Predicting the Training set results
y_pred_train = regressor.predict(X_mod)

# Predicting the Test set results
b = [x - 1 for x in a]
y_pred_test = regressor.predict(X_test[:,b])

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
if(regressor.predict([[X7,X8,X9]]))<=0:
    print('\n\tDemurrage Cost will not occur')
else:
    k=regressor.predict([[X7,X8,X9]])
    print("\n\tDemurrage Cost will occur and the demurrage "
          "hours would be {} hours".format(np.round(k,1)))