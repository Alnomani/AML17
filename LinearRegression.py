import numpy as np

def costFunction(predicted, y):
	return 1/2*m*sum((y - predicted)**2)


class LinearRegression:
	def predict(self,featureMatrix):
		m = len(featureMatrix)
		interceptVector  = np.repeat(self.thetas[0], m)
		if len(self.thetas[1:]) == 1:
			temp = self.thetas[1]
		else:
			temp = self.thetas[1:]
		#print(featureMatrix,"\n",temp)
		return np.dot(featureMatrix,temp) + interceptVector

	def fitLR(self,featureMatrix,y_train):		
		m = len(featureMatrix)
		
		try:
			num_points = featureMatrix.shape[1]
			tempMatrix = np.ones((m,num_points+1))
			tempMatrix[:,1:] = featureMatrix
		except IndexError:
			num_points = 1
			tempMatrix = np.ones((m,num_points+1))
			tempMatrix[:,1] = featureMatrix
		
		self.thetas = np.zeros(num_points+1)
		alpha = 0.001
			
		while True:
			hVector = self.predict(featureMatrix)
			differences = (hVector-y_train)
			summedTerm = np.dot(tempMatrix,differences.reshape(-1,1))
			derivatives = alpha*(1/m)*summedTerm.flatten()
			self.thetas = self.thetas - derivatives
			if max(abs(derivatives)) < 0.0000001:
				break

				
X_train = np.array([[1,1],[2,2],[3,3]])
y_train = np.array([1,2,3])
X_test = np.array([[4,4],[5,5]])
#size = np.array([1,2,3,4,5,6,7,8,9,10])
#price = np.array([5,6,7,8,9,10,11,12,13,14])

lm = LinearRegression()
lm.fitLR(X_train,y_train)
print(lm.predict(X_test))


	