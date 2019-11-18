import numpy as np
import pandas as pd
import sklearn as sp
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from sklearn.model_selection import cross_val_score

# Import data from csv file
data = pd.read_csv('C:/Users/Mandeep Jandoria/Desktop/LoanPrediction/LoanApplicantData.csv', header=0);

# Feature preprocessing

# Covert category values to numeric values
data.Gender = pd.Categorical(data.Gender)
data['Gender_Mod'] = data.Gender.cat.codes

data.Married = pd.Categorical(data.Married)
data['Married_Mod'] = data.Married.cat.codes

data.Loan_ID = pd.Categorical(data.Loan_ID)
data['Loan_ID_Mod'] = data.Loan_ID.cat.codes

data.Education = pd.Categorical(data.Education)
data['Education_Mod'] = data.Education.cat.codes

data.Self_Employed = pd.Categorical(data.Self_Employed)
data['Self_Employed_Mod'] = data.Self_Employed.cat.codes

data.Property_Area = pd.Categorical(data.Property_Area)
data['Property_Area_Mod'] = data.Property_Area.cat.codes

data.Loan_Status = pd.Categorical(data.Loan_Status)
data['Loan_Status_Mod'] = data.Loan_Status.cat.codes

# Feature generation
for index, row in data.iterrows():
   if row["Dependents"] == 0:
       data.loc[index, 'PerIncome']  = float(row["ApplicantIncome"]) + float(row["CoapplicantIncome"])

   else:
       data.loc[index, 'PerIncome'] = (float(row["ApplicantIncome"]) + float(row["CoapplicantIncome"]))/float(row["Dependents"])+1
       
for index, row in data.iterrows():
    data.loc[index, 'EMI']  = float(row["LoanAmount "])/float(row["Loan_Amount_Term"])
	   
data_mod = data[['Loan_ID_Mod','Gender_Mod','Dependents','Education_Mod','Self_Employed_Mod','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area_Mod','Loan_Status_Mod','PerIncome','EMI']]

x_mod = data[['Loan_ID_Mod','Gender_Mod','Dependents','Education_Mod','Self_Employed_Mod','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area_Mod','PerIncome','EMI']]
y_mod = data[['Loan_Status_Mod']]

x_mod = np.array(x_mod).reshape((len(x_mod)), x_mod.shape[1])
y_mod = np.array(y_mod).reshape((len(y_mod)), y_mod.shape[1])

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x_mod, y_mod, test_size=0.05, random_state=101)

# Min max normalization of the data
from sklearn.preprocessing import MinMaxScaler
scaler1 = MinMaxScaler()
scaler1.fit(X_train)
MinMaxScaler(copy=True, feature_range=(0, 1))
X_train = scaler1.transform(X_train)
X_test = scaler1.transform(X_test)

# Tuning the model
from sklearn.neural_network import MLPClassifier # MLPClassifier uses cross-entropy loss function by default
mlp = MLPClassifier(hidden_layer_sizes=(50, 50), activation="tanh", max_iter=500, learning_rate="constant", learning_rate_init=0.0001, verbose=False)
mlp.fit(X_train, Y_train)

# Accuracy using 8-fold cross validation
scores = cross_val_score(mlp, X_train, Y_train, cv=8, scoring='accuracy')
print ("Accuracy of the classifier is" , scores.mean())

# Recall using 8-fold cross validation
score = cross_val_score(mlp, X_train, Y_train, cv=8, scoring='recall')
print ("Recall of the classifier is" , score.mean())

# Plot for loss function versus number of iterations
loss_values= mlp.loss_curve_
plt.xlabel('N iteration')
plt.ylabel('loss values')
plt.plot(loss_values)
plt.show()
