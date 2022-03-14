#import libraries
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#load data to python environment
df=pd.read_csv('HRDataset_v14.csv')


#outlier removal
import numpy as np

# We have no Years of Experience column in the dataset, so add a column based on the DateofHire and DateofTermination (if available) 
# using below function. This may not show the full experience of the employee before the current employment, but useful 
from datetime import date
def experienceCalc(doj, dot):
    today = date.today()
    return np.where(pd.isna(dot), today.year - doj.dt.year, dot.dt.year - doj.dt.year)

df = pd.DataFrame(df)
 
df['YearsOfExperience'] = experienceCalc(df['DateofHire'], df['DateofTermination'])
df[['YearsOfExperience']]





#LAbel encoding
from sklearn import model_selection, preprocessing
for c in df.columns:
    if df[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(df[c].values)) 
        df[c] = lbl.transform(list(df[c].values))
        #x_train.drop(c,axis=1,inplace=True)
#Moved the salary column to another variable
y= df.pop('Salary')

df.head()
X = df.drop(['Employee_Name', 'EmpID','DOB','DateofHire','DateofTermination','LastPerformanceReview_Date','ManagerID'], axis=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2)


#standardizi data
from sklearn import preprocessing
stand=preprocessing.StandardScaler()
X=stand.fit_transform


from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
RF_regressor = RandomForestRegressor(n_estimators= 600,min_samples_split=5,min_samples_leaf= 1,max_depth=60,
bootstrap=False)
RF_regressor.fit(X_train, 
               y_train,
               validation_data=(X_train,y_train),
               validation_split=0.8,
               nb_epoch=250,
               shuffle=True,
               batch_size=402,
               verbose=1)
y_pred=RF_regressor.predict(X_test)

#Fitting the model
predict_test=lbl.inverse_transform(y_pred)
#Saving the model to disk
pickle.dump(RF_regressor,open('model.pkl','wb') )
