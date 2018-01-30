import pandas as pd 
melbourne_file_path='melb_data.csv'
melb_data=pd.read_csv(melbourne_file_path)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

melb_target=melb_data.Price
melb_predictors=melb_data.drop(['Price'],axis=1)

#select_dtypes 按数据类型选取datepandas 列 include, exclude : list-like(传入想要查找的类型)
melb_numeric_predictors=melb_predictors.select_dtypes(exclude=['object'])

X_train, X_test, y_train, y_test = train_test_split(melb_numeric_predictors, melb_target,
                                        train_size=0.7,test_size=0.3,random_state=0)

def score_dataset(X_train, X_test, y_train, y_test):
    model=RandomForestRegressor()
    model.fit(X_train,y_train)
    preds=model.predict(X_test)
    return mean_absolute_error(y_test,preds)

# 删除缺失列
cols_with_missing=[col for col in X_train.columns if X_train[col].isnull().any()]
reduced_X_train=X_train.drop(cols_with_missing,axis=1)
reduced_X_test=X_test.drop(cols_with_missing,axis=1)
print("Mean Absolute Error from dropping columns with Missing Values:")
print(score_dataset(reduced_X_train,reduced_X_test,y_train,y_test))

# 机器学习补充缺失列
from sklearn.preprocessing import Imputer
my_imputer=Imputer()
imputed_X_train=my_imputer.fit_transform(X_train)
imputed_X_test=my_imputer.transform(X_test)
print("Mean Absolute Error from dropping columns with Missing Values:")
print(score_dataset(imputed_X_train,imputed_X_test,y_train,y_test))

