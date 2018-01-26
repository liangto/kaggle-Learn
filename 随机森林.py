import pandas as pd 

melbourne_file_path='melb_data.csv'
melbourne_data=pd.read_csv(melbourne_file_path)

filtered_melbourne_data=melbourne_data.dropna(axis=0)
y=filtered_melbourne_data.Price
melbourne_predictors= ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 
                        'YearBuilt', 'Lattitude', 'Longtitude']
        
x=filtered_melbourne_data[melbourne_predictors]

from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(x, y,random_state = 0)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
forest_model=RandomForestRegressor()
forest_model.fit(train_X,train_y)
melb_preds=forest_model.predict(val_X)
print(mean_absolute_error(val_y,melb_preds))