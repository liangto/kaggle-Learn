import pandas as pd 

train_data=pd.read_csv("train.csv")
test_data=pd.read_csv("test.csv")

train_data.dropna(axis=0,subset=['SalePrice'],inplace=True)
target=train_data.SalePrice
col_with_missing=[col for col in train_data.columns if train_data[col].isnull().any()]     
candidate_train_predictors=train_data.drop(['Id','SalePrice']+col_with_missing,axis=1)
candidate_test_predictors=test_data.drop(['Id']+col_with_missing,axis=1)
low_cardinality_cols=[cname for cname in candidate_train_predictors.columns if
                        candidate_train_predictors[cname].nunique()<10 and 
                        candidate_train_predictors[cname].dtype=='object']
numeric_cols=[cname for cname in candidate_train_predictors.columns if 
                candidate_train_predictors[cname].dtype in ['int64','float64']]
my_cols=low_cardinality_cols+numeric_cols
train_predictors=candidate_train_predictors[my_cols]
test_predictors=candidate_test_predictors[my_cols]

# print(train_predictors.dtypes.sample(10))
one_hot_encoded_train_predictors=pd.get_dummies(train_predictors)
print(one_hot_encoded_train_predictors.head())


one_hot_encoded_training_predictors = pd.get_dummies(train_predictors)
one_hot_encoded_test_predictors = pd.get_dummies(test_predictors)
final_train, final_test = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors,join='left', axis=1)