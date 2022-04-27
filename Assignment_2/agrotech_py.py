
import pandas as pd
from scipy import stats
from scipy.stats import norm, skew #for some statistics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import seaborn as sns
import xgboost as xgb
from lightgbm import LGBMRegressor
import numpy as np
from sklearn.datasets import load_linnerud
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import KFold,StratifiedKFold, TimeSeriesSplit
from sklearn.model_selection import KFold,StratifiedKFold, TimeSeriesSplit


#Reading the xlsx file via each sheet to create 4 sheets to loop through

data = pd.read_excel('Data.xlsx', sheet_name=None)

# loop through the dictionary and save csv
for sheet_name, df in data.items():
  df.to_csv(f'{sheet_name}.csv')

#Created 4 csv file from the sheet and loading in each of the sheet
flight  = pd.read_csv("flight dates.csv")
planting  = pd.read_csv("planting.csv")
plant  =  pd.read_csv("plants.csv")
weather = pd.read_csv("weather.csv")

#droping a empty row in the plant data
plant.drop(["Unnamed: 0"],axis  = 1,inplace  = True)
flight.drop(["Unnamed: 0"],axis  = 1,inplace  = True)
planting.drop(["Unnamed: 0"],axis  = 1,inplace  = True)
weather.drop(["Unnamed: 0"],axis  = 1,inplace  = True)

#Selecting speific coloumns
plant.columns = ['BatchNumber', 'PlantDate', 'Class', 'FreshWeight(g)',
       'HeadWeight(g)', 'RadialDiameter(mm)', 'PolarDiameter(mm)',
       'DiameterRatio', 'Leaves', 'Density(kg/L)', 'LeafArea(cm^2)',
       'SquareID', 'CheckDate', 'FlightDate', 'Remove']

flight.columns  = ['BatchNumber', 'FlightDate']

plant.FlightDate  = plant['BatchNumber'].map(flight.set_index('BatchNumber')['FlightDate'])

weather.columns = ['PlantDate', 'SolarRadiation[avg]', 'Precipitation[sum]',
       'Wind Speed[avg]', 'Wind Speed[max]', 'BatteryVoltage[last]',
       'LeafWetness[time]', 'AirTemperature[avg]', 'AirTemperature[max]',
       'AirTemperature[min]', 'RelativeHumidity[avg]', 'DewPoint[avg]',
       'DewPoint[min]', 'ET0[result]']

#Creating a copy to keep the original file
data  = plant.copy()

weather.PlantDate  = pd.to_datetime(weather.PlantDate)

data.PlantDate  = pd.to_datetime(data.PlantDate)

#Mapping and merging Plant data with the Plant date data
data = pd.merge(data,weather,on = "PlantDate", how  = "left"  )

#Converting Flight date data to a datetime formart and manipulating to create Month and Day
data.FlightDate = pd.to_datetime(data.FlightDate)

data["number_of_days"] = (data["FlightDate"] - data.PlantDate).dt.days

data  = data[data['HeadWeight(g)'].notna()]

data["Plantmonth"]= data.PlantDate.dt.month
data["PlantDay"] = data.PlantDate.dt.day

#Dropping irrelevants columns

data.drop(["PlantDate","CheckDate","FlightDate","Remove"],axis  = 1,inplace  = True)

data  = data.reset_index()
data.drop("index",axis  = 1 , inplace  = True)

data.columns = ['BatchNumber', 'Class', 'FreshWeight(g)', 'HeadWeight(g)',
       'RadialDiameter(mm)', 'PolarDiameter(mm)', 'DiameterRatio', 'Leaves',
       'Density(kg/L)', 'LeafArea(cm2)', 'SquareID', 'SolarRadiationavg',
       'Precipitationsum', 'Wind Speedavg', 'Wind Speedmax',
       'BatteryVoltagelast', 'LeafWetnesstime', 'AirTemperatureavg',
       'AirTemperaturemax', 'AirTemperaturemin', 'RelativeHumidityavg',
       'DewPointavg', 'DewPointmin', 'ET0result', 'number_of_days',
       'Plantmonth', 'PlantDay']

X  = data.drop(['HeadWeight(g)', 'RadialDiameter(mm)', 'PolarDiameter(mm)',"Leaves"], axis  = 1)
y = data[['HeadWeight(g)', 'RadialDiameter(mm)', 'PolarDiameter(mm)']]

#Creating the cross validation using KFold with 5 Folds

errcb=[]
y_pred_totcb=[]

splits = 5
fold=KFold(n_splits=splits)
i=1
for train_index, test_index in fold.split(X,y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)

    clf_xgb =  MultiOutputRegressor(model_xgb).fit(X_train, y_train)

    xgb_pred = clf_xgb.predict(X_test)
  
 #Printing the result using XGBoost
  
    print("folds")
    print("--------------------------------------------------------------------------------------------------")
    print("xgb mean squared error of HeadWeight {0}".format(mean_squared_error(y_test["HeadWeight(g)"], xgb_pred[:,:1])))
    print("xgb mean squared error of RadialDiameter {0}".format(mean_squared_error(y_test["RadialDiameter(mm)"], xgb_pred[:,1:2])))
    print("xgb mean squared error of PolarDiameter {0}".format(mean_squared_error(y_test["PolarDiameter(mm)"], xgb_pred[:,2:])))


#Second Model LightGBM model to be able to compare  

errcb=[]
y_pred_totcb=[]

splits = 5
fold=KFold(n_splits=splits)
i=1
for train_index, test_index in fold.split(X,y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    model_lgb = LGBMRegressor(learning_rate=0.05, max_depth=3 )


    clf_lgb =  MultiOutputRegressor(model_lgb).fit(X_train, y_train)

    lgb_pred  = clf_lgb.predict(X_test)
  
    print("folds")
    print("--------------------------------------------------------------------------------------------------")
    print("lgb mean squared error of HeadWeight {0}".format(mean_squared_error(y_test["HeadWeight(g)"], lgb_pred[:,:1])))
    print("lgb mean squared error of RadialDiameter {0}".format(mean_squared_error(y_test["RadialDiameter(mm)"], lgb_pred[:,1:2])))
    print("lgb mean squared error of PolarDiameter {0}".format(mean_squared_error(y_test["PolarDiameter(mm)"], lgb_pred[:,2:])))


#Printing the various metrics for model performance Mean Squared Error and r2_score for xgb and Lgbm
"""#xgb evaluation"""

from sklearn.metrics import mean_squared_error,r2_score
print("xgb mean squared error of HeadWeight {0}".format(mean_squared_error(y_test["HeadWeight(g)"], xgb_pred[:,:1])))
print("xgb mean squared error of RadialDiameter {0}".format(mean_squared_error(y_test["RadialDiameter(mm)"], xgb_pred[:,1:2])))
print("xgb mean squared error of PolarDiameter {0}".format(mean_squared_error(y_test["PolarDiameter(mm)"], xgb_pred[:,2:])))

print("xgb r2_score of HeadWeight {0}".format(r2_score(y_test["HeadWeight(g)"], xgb_pred[:,:1])))
print("xgb r2_score of RadialDiameter {0}".format(r2_score(y_test["RadialDiameter(mm)"], xgb_pred[:,1:2])))
print("xgb r2_score of PolarDiameter {0}".format(r2_score(y_test["PolarDiameter(mm)"], xgb_pred[:,2:])))

"""#Lgbm evaluation"""

from sklearn.metrics import mean_squared_error,r2_score
print("lgb mean squared error of HeadWeight {0}".format(mean_squared_error(y_test["HeadWeight(g)"], lgb_pred[:,:1])))
print("lgb mean squared error of RadialDiameter {0}".format(mean_squared_error(y_test["RadialDiameter(mm)"], lgb_pred[:,1:2])))
print("lgb mean squared error of PolarDiameter {0}".format(mean_squared_error(y_test["PolarDiameter(mm)"], lgb_pred[:,2:])))

print("lgb r2_score of HeadWeight {0}".format(r2_score(y_test["HeadWeight(g)"], lgb_pred[:,:1])))
print("lgb r2_score of RadialDiameter {0}".format(r2_score(y_test["RadialDiameter(mm)"], lgb_pred[:,1:2])))
print("lgb r2_score of PolarDiameter {0}".format(r2_score(y_test["PolarDiameter(mm)"], lgb_pred[:,2:])))


#Predicting the values of a test set (20% of the dataset)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgb_pred  = clf_xgb.predict(X_test)

Prediction = pd.DataFrame(xgb_pred, 
             columns=['HeadWeight(g)', 'RadialDiameter(mm)',	'PolarDiameter(mm)'])

pd.DataFrame(Prediction).to_csv('Yield_Prediction.csv', index=False)
