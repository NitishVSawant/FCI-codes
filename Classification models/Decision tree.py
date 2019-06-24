# Decision Tree

# Importing the libraries
import numpy as np
import pandas as pd

# Warehouse selection
print("Consider the format warehouse name: ")
print("\tKazhakuttam-KAZ,\tChingavanam-CHI")
ware=input("Enter the name of warehouse: ").upper()

# Importing the dataset
if(ware=="KAZ"):
    datasetinitial = pd.read_csv('Kazhakuttam Classification.csv')

if(ware=="CHI"):
    datasetinitial = pd.read_csv('Chingavanam Classification.csv')

#Delete missing values in dataframe
dataset = datasetinitial[np.isfinite(datasetinitial['Available Warehouse capacity'])]

#Take numerical values
X = dataset.loc[:, ['Available Warehouse capacity','Number of wagons placed',
                    'Arrival time of wagons']].values
y = dataset.loc[:, 'Demurrage occurs or not'].values

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

#Input matrix containing standarised numerical and categorical values
X=np.concatenate((X_binary,X),axis=1)

"""
X column name
1:Sun,2:Mon,3:Tue,4:Wed,5:Thu,6:Fri,7:Working,
8:Available Warehouse capacity,9:Number of wagons placed,
10:Arrival time of wagons
"""

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.35, random_state = 0)

# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(max_depth=4,criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

#No of folds for cross-validation used in Cross-validation and Grid Search
CV=10

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = CV)
print("Mean of accuracy obtained by "+str(CV)+"-fold cross-validation for training model parameters: %0.3f "%(accuracies.mean()))
print("Standard deviation of accuracy obtained by "+str(CV)+"-fold cross-validation for for training model parameters: %0.3f "%(accuracies.std()))

# Predicting the Test & Training set results
y_pred_test = classifier.predict(X_test)
y_pred_train = classifier.predict(X_train)

# Making the confusion matrix
print("\nConfusion matrix:")
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_train, y_pred_train)
print("Confusion matrix for training dataset:")
print(str(cm1))
print("Confusion matrix for test dataset:")
cm2 = confusion_matrix(y_test, y_pred_test)
print(str(cm2))

#TP, TN, FP, FN
#Training
TP1=cm1[0,0]
TN1=cm1[1,1]
FP1=cm1[1,0]
FN1=cm1[0,1]

#Test
TP2=cm2[0,0]
TN2=cm2[1,1]
FP2=cm2[1,0]
FN2=cm2[0,1]

#Accuracy,
print("\nAccuracy scores:")
ac1=(TP1+TN1)/(TP1+TN1+FP1+FN1)
print("\tAccuracy score for training dataset: %0.3f "%(ac1))
ac2=(TP2+TN2)/(TP2+TN2+FP2+FN2)
print("\tAccuracy score for test dataset: %0.3f "%(ac2))

#Sensitivity or Recall
print("\nSensitivity or Recall scores:")
sen1=(TP1)/(TP1+FN1)
print("\tSensitivity score for training dataset: %0.3f "%(sen1))
sen2=(TP2)/(TP2+FN2)
print("\tSensitivity score for test dataset: %0.3f "%(sen2))

#Specificity
print("\nSpecificity scores:")
spe1=(TN1)/(TN1+FP1)
print("\tSpecificity score for training dataset: %0.3f "%(spe1))
spe2=(TN2)/(TN2+FP2)
print("\tSpecificity score for test dataset: %0.3f "%(spe2))

#Precision
print("\nPrecision scores:")
pre1=(TP1)/(TP1+FP1)
print("\tPrecision score for training dataset: %0.3f "%(pre1))
pre2=(TP2)/(TP2+FP2)
print("\tPrecision score for test dataset: %0.3f "%(pre2))

#F-measure
print("\nF-measure scores:")
f11=(2*pre1*sen1)/(pre1+sen1)
print("\tF-measure score for training dataset: %0.3f "%(f11))
f12=(2*pre2*sen2)/(pre2+sen2)
print("\tF-measure score for test dataset: %0.3f "%(f12))

#ROC curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
probs = classifier.predict_proba(X_test)
probs = probs[:, 1]
auc = roc_auc_score(y_test, probs)
fpr, tpr, thresholds = roc_curve(y_test, probs)
plt.figure()
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(fpr, tpr, marker='.',label='AUC = %0.2f' % auc)
print('\nArea under curve (AUC): %.3f' % auc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (ROC) curve')
plt.legend(loc="lower right")
plt.show()

# Program for input parameters
"""
X column name
1:Sun,2:Mon,3:Tue,4:Wed,5:Thu,6:Fri,7:Working,
8:Available Warehouse capacity,9:Number of wagons placed,
10:Arrival time of wagons
"""

print("\nEnter the following data: ")
X8=float(input("Enter the available warehouse capacity: "))
X8=(X8-WCM)/WCSD
X9=float(input("Enter the number of wagons placed: "))
X9=(X9-WPM)/WPSD
X10=float(input("Enter the arrival time of wagons in 24 hour clock format: "))
X10=(X10-ATM)/ATSD
print("Consider the format for day of arrival: ")
print("\tSunday-SUN,\tMonday-MON,\tTuesday-TUE,\tWednesday-WED,\tThursday-THU,\tFriday-FRI,\tSaturday-SAT")
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
if(classifier.predict([[X1,X2,X3,X4,X5,X6,X7,X8,X9,X10]]))==0:
    print('\n\tDemurrage Cost will not occur')
else:
    print("\n\tDemurrage Cost will occur")

# Decision tree visualisation
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
import graphviz
dot_data = StringIO()
export_graphviz(classifier, out_file=dot_data,filled=True, rounded=True,
                special_characters=True,
                feature_names=["Sun","Mon","Tue","Wed","Thu","Fri",
    "Working","Available Warehouse capacity","Number of wagons placed",
                "Arrival time of wagons"],class_names=["DC does not occur","DC occurs"])
print("\n In 'value' row first number stands for number of observations" 
      " for which 'Demurrage Cost does not occur' and the second number" 
      " stands for number of observations for which 'Demurrage Cost occurs'")
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())

"""
# Decision tree visualisation (another approach)
from sklearn.tree import export_graphviz
export_graphviz(classifier, out_file="Decision tree.dot",filled=True, rounded=True,
                special_characters=True,
                feature_names=["Sun","Mon","Tue","Wed","Thu","Fri",
    "Working","Available Warehouse capacity","Number of wagons placed",
                "Arrival time of wagons"],class_names=["DC does not occur","DC occurs"])
import graphviz
with open("Decision tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)
"""