import pandas as pd 

melbourne_file_path='melb_data.csv'
melbourne_data=pd.read_csv(melbourne_file_path)

filtered_melbourne_data=melbourne_data.dropna(axis=0)
y=filtered_melbourne_data.Price
melbourne_predictors= ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 
                        'YearBuilt', 'Lattitude', 'Longtitude']
        
x=filtered_melbourne_data[melbourne_predictors]

from sklearn.tree import DecisionTreeRegressor
melbourne_model=DecisionTreeRegressor()
melbourne_model.fit(x,y)
from sklearn.metrics import mean_absolute_error
predited_home_prices=melbourne_model.predict(x)
print(mean_absolute_error(y,predited_home_prices))

from sklearn.model_selection import train_test_split
# split data into training and validation data, for both predictors and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(x, y,random_state = 0)
# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model
melbourne_model.fit(train_X, train_y)
# get predicted prices on validation data
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))


def get_mae(max_leaf_nodes,predictors_train,predictors_val,targ_train,tar_val):
    model=DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes,random_state=0)
    model.fit(predictors_train,targ_train)
    preds_val=model.predict(predictors_val)
    mae=mean_absolute_error(tar_val,preds_val)
    return mae

print ("#"*50)

for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae=get_mae(max_leaf_nodes,train_X,val_X,train_y,val_y)
    print("Max leaf nodes: %d \t\t Mean Absolute Erro : %d" %(max_leaf_nodes,my_mae))