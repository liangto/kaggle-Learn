import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer

data = pd.read_csv('train.csv')
data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = data.SalePrice
X = data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])
train_X, test_X, train_y, test_y = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.25)

my_imputer = Imputer()
train_X = my_imputer.fit_transform(train_X)
test_X = my_imputer.transform(test_X)

from xgboost import XGBRegressor
# my_model=XGBRegressor()
# my_model.fit(train_X,train_y,verbose=False)
# predictions=my_model.predict(test_X)
# from sklearn.metrics import mean_absolute_error
# print("Error:" + str(mean_absolute_error(predictions,test_y)))

my_model=XGBRegressor(n_estimators=1000,learning_rate=0.5)
my_model.fit(train_X,train_y,early_stopping_rounds=5
                ,eval_set=[(test_X,test_y)],verbose=False)
predictions=my_model.predict(test_X)
from sklearn.metrics import mean_absolute_error
print("Error:" + str(mean_absolute_error(predictions,test_y)))


from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer

my_pipeline = make_pipeline(Imputer(), RandomForestRegressor())
my_pipeline.fit(train_X, train_y)
predictions = my_pipeline.predict(test_X)