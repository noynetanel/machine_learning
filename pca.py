import random
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
import numpy as np

def plot_vector_as_image(image, h, w):
	"""
	utility function to plot a vector as image.
	Args:
	image - vector of pixels
	h, w - dimesnions of original pi
	"""
	plt.imshow(image.reshape((h, w)), cmap=plt.cm.gray)
	plt.title('title', size=12)
	plt.show()

def get_pictures_by_name(name='Ariel Sharon'):
	"""
	Given a name returns all the pictures of the person with this specific name.
	YOU CAN CHANGE THIS FUNCTION!
	THIS IS JUST AN EXAMPLE, FEEL FREE TO CHANGE IT!
	"""
	lfw_people = load_data()
	selected_images = []
	n_samples, h, w = lfw_people.images.shape
	target_label = list(lfw_people.target_names).index(name)
	for image, target in zip(lfw_people.images, lfw_people.target):
		if (target == target_label):
			image_vector = image.reshape((h*w, 1))
			selected_images.append(image_vector)
	return selected_images, h, w

def load_data():
	# Don't change the resize factor!!!
	lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
	return lfw_people

######################################################################################
"""
Other then the PCA function below the rest of the functions are yours to change.
"""

def PCA(X, k):
	"""
	Compute PCA on the given matrix.

	Args:
		X - Matrix of dimesions (n,d). Where n is the number of sample points and d is the dimension of each sample.
		For example, if we have 10 pictures and each picture is a vector of 100 pixels then the dimesion of the matrix would be (10,100).
		k - number of eigenvectors to return

	Returns:
	  U - Matrix with dimension (k,d). The matrix should be composed out of k eigenvectors corresponding to the largest k eigenvectors
	  		of the covariance matrix.
	  S - k largest eigenvalues of the covariance matrix. vector of dimension (k, 1)
	"""
	n = X.shape[0]
	d = X.shape[1]
	X_centred = X -np.average(X, axis=0)
	covariance_matrix = 1 / n * (X_centred.T @ X_centred)
	P, D, Q = np.linalg.svd(covariance_matrix)
	U = Q[0:k, :]
	S = D[0:k]
	return U, S

def section_B():
	selected_images, h, w = get_pictures_by_name("Tony Blair")
	X = np.array(selected_images)[:, :, 0]
	U, S = PCA(X, 10)
	fig = plt.figure()
	for i in range(1, 11):
		fig.add_subplot(2, 5, i)
		plt.imshow(U[i - 1].reshape((h, w)), cmap=plt.cm.gray)
		plt.title('image ' + str(i), size=10)
	plt.tight_layout()
	plt.show()

def section_C():
	selected_images, h, w = get_pictures_by_name("Tony Blair")
	X = np.array(selected_images)[:, :, 0]
	n = X.shape[0]
	d = X.shape[1]
	K = [1, 5, 10, 30, 50, 100]
	l2 = []
	for k in K:
		cur_l2 = 0
		U, S = PCA(X, k)
		A = U @ X.T
		X_tag = (U.T @ A).T
		sample = random.sample(range(0, n), 5)
		fig = plt.figure(figsize=(4, 10))

		for i, image_index in enumerate(sample):
			fig.add_subplot(5, 2, 2*i+1)
			plt.imshow(X[image_index].reshape((h, w)), cmap=plt.cm.gray)
			plt.title('original', size=10)

			fig.add_subplot(5, 2, 2 * i + 2)
			plt.imshow(X_tag[image_index].reshape((h, w)), cmap=plt.cm.gray)
			plt.title('transformed', size=10)

			cur_l2 += np.linalg.norm(X[image_index]-X_tag[image_index])

		l2.append(cur_l2)
		fig.suptitle("k " + str(k), size=20)
		plt.tight_layout()
		plt.show()

	plt.plot(K, l2)
	plt.xlabel('k')
	plt.ylabel('sum of l2 distances')
	plt.title('l2 distances- k')
	plt.show()

if __name__ == "__main__":
	section_B()
	section_C()

