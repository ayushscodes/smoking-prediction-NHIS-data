import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import read_data
import pca
from imputation import *

USE_IMPUTED_DATA = True

def train_basicSVM():

	#basic SVM model

	basic_svm = LinearSVC()
	basic_svm.fit(X_train, y_train)
	print("Basic SVM Score: ", basic_svm.score(X_val, y_val))
	print("Basic SVM Score Test: ", basic_svm.score(X_test, y_test))


def train_L1SVM():
	# L1 regularized SVM 
	svm_lasso = LinearSVC(penalty= "l1", dual=False, C=0.1)
	svm_lasso.fit(X_train, y_train)
	print("SVM with L1 regularization Score: ", svm_lasso.score(X_val, y_val))
	print("Best L1 regularization Test Score: ", svm_lasso.score(X_test, y_test))


def tune_SVM():
	# tuning hyperparameters

	param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 
	              'penalty': ['l1', 'l2'],
	              'loss': ['hinge', 'squared_hinge'],
	             'dual':[False, True]}

	basic_svm_2 = LinearSVC()
	svm_tune = GridSearchCV(basic_svm_2, param_grid, error_score = 0.0)
	svm_tune.fit(X_train_and_val, y_train_and_val)
	print(svm_tune.best_estimator_)
	print("Best score: ", svm_tune.best_score_)
	print("Best C: ", svm_tune.best_estimator_.C)
	print("Best penalty: ", svm_tune.best_estimator_.penalty)
	print("Best loss: ", svm_tune.best_estimator_.loss)


def train_RBFSVM():

    rbf_svm = SVC()
    rbf_svm.fit(X_train, y_train)
    print("RBF Kernel SVM Score: ", rbf_svm.score(X_val, y_val)) 


def train_pca_SVM(num_components):

	#train SVM with PCA model 

	X_train_transform = pca.apply_pca(X_train, num_components)
	X_val_transform = pca.apply_pca(X_val, num_components)

	basic_svm = LinearSVC()
	basic_svm.fit(X_train_transform, y_train)
	print("SVM Score with PCA with ", num_components, "components: ", basic_svm.score(X_val_transform, y_val))



def train_pca_L1SVM(num_components):

	#train SVM with PCA model 

	X_train_transform = pca.apply_pca(X_train, num_components)
	X_val_transform = pca.apply_pca(X_val, num_components)

	svm = LinearSVC(penalty= "l1", dual=False)
	svm.fit(X_train_transform, y_train)
	print("L1 SVM Score with PCA with ", num_components, "components: ", svm.score(X_val_transform, y_val))


def train_SparsePCA_SVM(num_components):

	#train SVM with PCA model 

	X_transform = pca.apply_SparsePCA(X_train, num_components)

	basic_svm = LinearSVC()
	basic_svm.fit(X_transform, y_train)
	print("SVM Score with SparsePCA with ", num_components, "components: ", basic_svm.score(X_val, y_val))


if __name__ == "__main__":

	if USE_IMPUTED_DATA:
		X_train, y_train = get_imputed_traindata()
		X_val, y_val = get_imputed_valdata()
		X_test, y_test = get_imputed_testdata()
		X_train_and_val = np.concatenate((X_train, X_val))
		y_train_and_val = np.concatenate((y_train, y_val))


		train_basicSVM()
		train_L1SVM()
		train_RBFSVM()
		tune_SVM()

		train_pca_SVM(num_components = 120)
		train_pca_L1SVM(num_components = 120)

	# else:
	#     print ("--------------- LOADING DATA -------------------")
	#     X_train, y_train = read_data.get_traindata()
	#     X_val, y_val = read_data.get_valdata()
	#     X_train_and_val = np.concatenate((X_train, X_val))
	#     y_train_and_val = np.concatenate((y_train, y_val))

	#     print ("--------------- DATA IS LOADED -------------------")

	#     train_basicSVM()
	#     train_L1SVM()
	#     train_RBFSVM()
	#     tune_SVM()

	#     train_pca_SVM(num_components = 20)





'''
Basic SVM Score:  0.510101010101
SVM with L1 regularization Score:  0.765151515152

----tuning hyperparameters

Best score:  0.835265700483
Best C:  0.01
Best penalty:  l1
Best loss:  squared_hinge
Best dual:  False


With Imputation:
Best score:  0.662662662663
Best C:  1
Best penalty:  l1
Best loss:  squared_hinge

'''

