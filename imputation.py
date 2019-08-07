from read_data import get_split_data
import pandas as pd
import numpy as np
from statsmodels.imputation import mice
import statsmodels.api as sm
from sklearn.preprocessing import Imputer
from fancyimpute import KNN, MICE

np.set_printoptions(threshold=10000)


X_train, Y_train, X_val, Y_val, X_test, Y_test, headers = get_split_data()
np.place(X_train, X_train==99.99, [np.nan])
np.place(X_val, X_val==99.99, [np.nan])
np.place(X_test, X_test==99.99, [np.nan])


# def statsmodel_mice_imputation():
# 	data = pd.DataFrame(X_train)
# 	data.columns = headers
# 	print("columns: ", data.columns)
# 	imp = mice.MICEData(data)


# 	print(imp.data)
# 	model=mice.MICE(model_formula='BPQ020', model_class=sm.OLS, data=imp)
# 	results=model.fit()
# 	print(results.summary())


def sklearn_imputation():
	imp = Imputer(strategy='mean', axis=1)
	X_train_new = imp.fit_transform(X_train)
	
	X_val_new = imp.fit_transform(X_val)
	X_test_new = imp.fit_transform(X_test)
	return X_train_new, X_val_new, X_test_new 


def knn_imputation(k):
	X_train_new = KNN(k=k).complete(X_train)
	X_val_new = KNN(k=k).complete(X_val)
	X_test_new = KNN(k=k).complete(X_test)
	return X_train_new, X_val_new, X_test_new


def fancyimpute_mice_imputation():
	X_train_new = MICE(init_fill_method='median').complete(X_train)
	X_val_new = MICE(init_fill_method='median').complete(X_val)
	X_test_new = MICE(init_fill_method='median').complete(X_test)
	return X_train_new, X_val_new, X_test_new

#Change method of imputation here
X_train_new, X_val_new, X_test_new = sklearn_imputation()


def get_imputed_traindata():
	return X_train_new, Y_train

def get_imputed_valdata():
	return X_val_new, Y_val

def get_imputed_testdata():
	return X_test_new, Y_test
