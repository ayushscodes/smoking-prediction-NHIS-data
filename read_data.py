import csv
import numpy as np
import copy
import matplotlib.pyplot as plt

np.set_printoptions(threshold=10000)


def read_dataset():

	DATASET = 'questionnaire'

	data = np.genfromtxt('data/%s.csv' %DATASET, delimiter=',', filling_values=99.99)

	with open('data/%s.csv' %DATASET) as f:
		reader = csv.reader(f)
		headers = next(reader)

	weird_col_index = headers.index('SMAQUEX.y')
	headers.remove('SMAQUEX.y')
	np.delete(data, weird_col_index, axis=1)

	# print(data.shape)

	return data, headers


def remove_smoking_columns(data, headers):

	#smoking_fieldnames = ['SMQ020', 'SMD030', 'SMQ040', 'SMQ050Q', 'SMQ050U', 'SMD055', 'SMD057', 'SMQ078', 'SMD641', 'SMD650', 'SMD093', 'SMDUPCA', 'SMD100BR', 'SMD100FL', 'SMD100MN', 'SMD100LN', 'SMD100TR', 'SMD100NI', 'SMD100CO', 'SMQ621', 'SMD630', 'SMQ661', 'SMQ665A', 'SMQ665B', 'SMQ665C', 'SMQ665D', 'SMQ670', 'SMQ848', 'SMQ852Q', 'SMQ852U', 'SMAQUEX2', 'SMD460', 'SMD470', 'SMD480', 'SMQ856', 'SMQ858', 'SMQ860', 'SMQ862', 'SMQ866', 'SMQ868', 'SMQ870', 'SMQ872', 'SMQ874', 'SMQ876', 'SMQ878', 'SMQ880', 'SMAQUEX.x', 'SMQ681', 'SMQ690A', 'SMQ710', 'SMQ720', 'SMQ725', 'SMQ690B', 'SMQ740', 'SMQ690C', 'SMQ770', 'SMQ690G', 'SMQ845', 'SMQ690H', 'SMQ849', 'SMQ851', 'SMQ690D', 'SMQ800', 'SMQ690E', 'SMQ817', 'SMQ690I', 'SMQ857', 'SMQ690J', 'SMQ861', 'SMQ863', 'SMQ690F', 'SMQ830', 'SMQ840']

	first_smoking_index = headers.index('SMQ020')
	last_smoking_index = headers.index('SMQ840')

	#SMQ040 = 'Do you now smoke cigarettes?'
	Y_index = headers.index('SMQ040')

	Y = data[:, Y_index]

	data_1, smoking, data_2 = np.split(data, [first_smoking_index, last_smoking_index+1], axis=1)
	X = np.concatenate((data_1, data_2), axis=1)

	headers_1, smoking_headers, headers_2 = np.split(headers, [first_smoking_index, last_smoking_index+1])
	nonsmoking_headers = np.concatenate((headers_1, headers_2))

	return X, Y, nonsmoking_headers


def remove_missing_data_rows(X, Y):

	Y = _norm_Y_data(Y)
	X, Y = _get_nonzero_rows(X, Y)

	return X, Y
	

def _split_data(X, Y):

	data_size, num_features = X.shape

	validation_set_size = int(round(data_size*.8))
	training_set_size = int(round(validation_set_size*.8))

	X_train = X[0:training_set_size]
	X_val = X[training_set_size:validation_set_size]
	X_test = X[validation_set_size:data_size]

	Y_train = Y[0:training_set_size]
	Y_val = Y[training_set_size:validation_set_size]
	Y_test = Y[validation_set_size:data_size]

	return X_train, Y_train, X_val, Y_val, X_test, Y_test


def _norm_Y_data(Y):
	new_Y = copy.deepcopy(Y)

	for i in range(len(Y)):
		if (Y[i] == 1) or (Y[i] == 2):
			new_Y[i] = 1
		elif Y[i] == 99.99:
			new_Y[i] = 99.99
		else:
			new_Y[i] = -1

	return new_Y


def _get_nonzero_rows(X, Y):

	new_X = copy.deepcopy(X)
	new_Y = copy.deepcopy(Y)

	indices = []
	for i in range(len(Y)):
		if Y[i] == 99.99:
			indices.append(i)

	new_Y = np.delete(new_Y, indices)
	new_X = np.delete(new_X, indices, 0)

	return new_X, new_Y


def clean_data(sparsity, X, Y, headers):

	# print("input data: ", X.shape)
	_, X1, Y1, headers = clean_column(sparsity, X, Y, headers)
	# print("column cleaned: ", X1.shape)
	_, X2, Y2 = clean_row(sparsity, X1, Y1)
	# print("final cleaned: ", X2.shape)

	return  X2, Y2, headers


def clean_column(sparsity, X, Y, headers): 
    remove = []
    count = 0
    full_headers = []
    i = 0
    L = len(X[0])
    header_index = 0
    while i < L:
        n = 1.0 - np.count_nonzero(X[:,i] == 99.99) * 1.0 / len(X)
        if n == 1.0:
        	full_headers.append(headers[header_index])
        if n <= sparsity:
            remove.append(header_index)
            X= np.delete(X,i,1)
            L -= 1
        else:
            count += 1
            i += 1
        header_index += 1
    headers = np.delete(headers, remove)
    print("col full: ", full_headers)
    return count, X, Y, headers

def clean_row(sparsity, X, Y):
    count = 0
    full = 0
    i = 0
    L = len(X)
    while i < L:
        n = 1.0 - np.count_nonzero(X[i] == 99.99) * 1.0 / len(X[i])
        if n == 1.0:
        	full += 1
        if n <= sparsity:
            X = np.delete(X, i, 0)
            Y = np.delete(Y, i)
            L -= 1
        else:
            count += 1
            i += 1
    print("full: ", full)      
    return count, X, Y
    

def spars_plot():
	
    X, Y = get_traindata()
    xaxis = np.arange(0.0, 1.0, 0.01)
    yaxis = [clean_column(i,X,Y)[0] for i in np.arange(0.0, 1.0, 0.01)]
    
    plt.plot(xaxis, yaxis)
    plt.ylabel('# of columns')
    plt.xlabel('sparsity')
    plt.show()


def get_data():

	data, headers = read_dataset()
	X, Y, headers = remove_smoking_columns(data, headers)
	X, Y = remove_missing_data_rows(X, Y)

	return X, Y, headers


def get_split_data():
	sparsity = 0.5
	X, Y, headers = get_data()
	X, Y, headers = clean_data(sparsity, X, Y, headers)
	X_train, Y_train, X_val, Y_val, X_test, Y_test = _split_data(X, Y)
	return X_train, Y_train, X_val, Y_val, X_test, Y_test, headers


def get_traindata():

	X, Y = get_split_data()[0], get_split_data()[1]
	return X, Y


if __name__ == "__main__":
	X_train, Y_train, X_val, Y_val, X_test, Y_test, headers = get_split_data()
	print("headers length: ", len(headers))
	# print("X shape: ", X.shape)
	# print("Y shape: ", Y.shape)
	print("X_train shape: ", X_train.shape)
	print("Y_train shape: ", Y_train.shape)
	print("X_val shape: ", X_val.shape)
	print("Y_val shape: ", Y_val.shape)
	print("X_test shape: ", X_test.shape)
	print("Y_test shape: ", Y_test.shape)
