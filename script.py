import numpy as np




#def costFunction(predicted, y):
#	return 1/2*m*sum((y - predicted)**2)

def h(thetas,featureMatrix,m):
	interceptVector  = np.repeat(thetas[0], m)
	if len(thetas[1:]) == 1:
		temp = thetas[1]
	else:
		temp = thetas[1:]
	return np.dot(featureMatrix,temp) + interceptVector

def fitLR(featureMatrix,y_train):
	m = len(featureMatrix)
	try:
		num_points = featureMatrix.shape[1]
	except IndexError:
		num_points = 1
		#featureMatrix = np.array(featureMatrix)[np.newaxis].T
	thetas = np.zeros(num_points+1)
	alpha = 0.001
	
	for i in range(1,100000):
		hVector = h(thetas, featureMatrix,m)
		differences = (hVector-y_train)
		#print(differences)
		tempMatrix = np.ones((m,num_points+1))
		if num_points != 1:
			tempMatrix[:,1:] = featureMatrix
		else:
			tempMatrix[:,1] = featureMatrix
		repeatedDifferences = np.tile(differences, (len(thetas),1)).T
		#print(tempMatrix,'\n',repeatedDifferences)
		summedTerm = np.sum(repeatedDifferences*tempMatrix,axis=0)
		derivatives = alpha*(1/m)*summedTerm
		thetas = thetas - derivatives
		print(h(thetas,featureMatrix,m))
	

X_train = np.array([[1,1],[2,2],[3,3]])
y_train = np.array([1,2,3])

size = np.array([1,2,3,4,5,6,7,8,9,10])
price = np.array([5,6,7,8,9,10,11,12,13,14])


fitLR(X_train,y_train)


	