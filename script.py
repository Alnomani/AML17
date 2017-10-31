import numpy as np

def costFunction(predicted, y):
	return 1/2*m *sum((y - predicted)**2)

def h(thetas,featureMatrix):
	repeatedThetas = np.tile(thetas,(len(featureMatrix),1))
	interceptVector  = repeatedThetas[:,0]
	return (repeatedThetas*featureMatrix) + interceptVector
#git test 2
def LRcls(k):
	thetas = np.zeros(k)
	alpha = 0.0001
	hVector = h(thetas, featureMatrix)
	
	for gradient > alpha:
		newThetas = thetas[0] - alpha*(1/m)*sum(hVector-y)



	