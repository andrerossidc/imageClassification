"""""
04/2019
Mateus Roder
----------------------------------------------------------------------------------------
Gaussian-Bernoulli - Restricted Boltzmann Machines, implementation from Hinton's paper:
> Geoffrey Hinton, A Practical Guide to Training Restricted Boltzmann Machines, 2010.
----------------------------------------------------------------------------------------
Contact:
Mateus Roder
mateusroder@hotmail.com
www.recogna.tech
"""""

import numpy as np
import pandas as pd

class GBRBM:
	def __init__(self, neurons_hidden, neurons_visible, epochs, batch, learning_rate, k, x_train, b_temp, seed):
		self.seed = seed
		np.random.seed(self.seed)
		self.neurons_hidden = neurons_hidden
		self.neurons_visible = neurons_visible 
		self.epochs = epochs 
		self.batch = batch
		self.learning_rate = learning_rate 
		self.k = k
		self.const = int(x_train.shape[0]//(self.batch))
		self.error = []
		self.weights = np.array(np.random.uniform(low = 0, high = 0.1,size = (neurons_visible, neurons_hidden)))
		self.a = np.zeros(shape=(self.batch, neurons_visible))
		self.b = np.zeros(shape=(self.batch, neurons_hidden))
		self.x_train = x_train
		self.b_temp = b_temp
		self.t1 = 1
		self.temp = []

	def sigmoid(self, x):
	    sig = (1/(1+np.exp(-x))).astype(float)
	    return np.round(sig,4)

	def energy (self, visibles, hidden, weights, a, b):
		dif = np.dot(visibles, self.a.T)
		dif2 = np.dot(hidden, self.b.T)
		dif3 = np.dot(np.dot(visibles, self.weights), hidden.T)
		soma = -dif - dif2 - dif3
		e = np.mean(soma)
		return  np.array(e)

	def runGibbsStep(self, data, weights, a, b, kstp, t1):
		V_0 = data
		for z in range(0,kstp):
			var1 = np.matmul(V_0, self.weights) + self.b
			pos_hidden_prob = (self.sigmoid(var1/self.t1)) 
			pos_hidden_state = pos_hidden_prob > np.random.ranf(size = pos_hidden_prob.shape)
			H_New = pos_hidden_state.astype(int)        
			
			x_1 = np.matmul(H_New, self.weights.T) + self.a
			V_0 = x_1
			
		return (x_1, H_New, pos_hidden_prob)


	def save_data (self, weights, a, b, error, neurons_hidden, epochs):
		w = pd.DataFrame(self.weights)
		w.to_csv('weights.csv')
		a1 = pd.DataFrame(self.a)
		a1.to_csv('a.csv')
		b1 = pd.DataFrame(self.b)
		b1.to_csv('b.csv')
		ene = pd.DataFrame(self.energ)
		ene.to_csv('energy.csv')
		erro = pd.DataFrame(self.error)
		erro.to_csv('error.csv')
#	    	save = {'Learning Rate': [learning_rate], 'Hidden Units':[neurons_hidden], 'Epochs':[epochs], 'K-Steps':[k]}
#		save = {'Hidden Units':[neurons_hidden], 'Epochs':[epochs], 'K-Steps':[k]}
#		save = pd.DataFrame(save)
#	    	save.to_csv('parameters100.csv')
		return ("Training Results Saved!")

	def get_error(self):
		return self.error
		
	def get_params(self):
		return (self.weights, self.a, self.b)

	def fit(self, x_train):
		np.random.seed(self.seed)
		q = 0
		cont = 0
		data = np.zeros((self.batch,self.neurons_visible))
		aux = 0
		for aux in range(0,self.batch): # first batch over all training samples
			data[aux,:] = self.x_train[aux,:]
			q += 1
		mi = np.zeros((self.neurons_visible))
		q = self.batch
		beta = 1
		self.temp.append(beta)
		energ = []
		temp_rate = 0.00001 # rate to fit temperature, similar to learning rate #
	    
		for epoch in range(0, self.epochs): # EPOCHS
	    	#while (np.mean(error) > 0.1):
			for i in range(0,self.const): # ITERATIONS OVER BATCHES
				var1 = np.matmul(data, self.weights) + self.b
				pos_hidden_prob = self.sigmoid(var1)
				pos_hidden_state = pos_hidden_prob > np.random.ranf(size = pos_hidden_prob.shape)
				H_0 = pos_hidden_state.astype(int)
				mi = np.matmul(H_0, self.weights.T) + self.a
				e1 = self.energy(data, H_0, self.weights, self.a, self.b)
				
				V_new, H_new, P_new = self.runGibbsStep(mi, self.weights, self.a, self.b, self.k, self.t1)
				e2 = self.energy(V_new, H_new, self.weights, self.a, self.b)
				energ.append(np.round(e2,2))
				et = np.round(e2-e1, 2)

				if (self.b_temp==True):
					beta = beta + temp_rate*et*(beta**2)
					self.t1 = 1/beta
					self.temp.append(self.t1)    
				else:
					self.t1 = 1

				
				pos_grad = np.matmul(data.T, pos_hidden_prob)
				neg_grad = np.matmul(V_new.T, P_new)
				    
				self.weights = self.weights + self.learning_rate*((pos_grad - neg_grad)/self.batch)
				    
				res = (data - V_new)
				b1 = (pos_hidden_prob - P_new)
				    
				self.a = self.a + self.learning_rate*res
				self.b = self.b + self.learning_rate*b1
						
				err = np.sum((data - V_new)**2, axis = 0)
				err = np.sum(err)/self.batch
				self.error.append(err)
				if(self.error[cont]==0.0000):	break

				cont += 1
				z=0
				for aux in range(q,q+self.batch):
					if (aux == self.x_train.shape[0]):
					    break
					data[z,:] = self.x_train[aux,:]
					q += 1
					z += 1
			if epoch%100==0:
				print("Epoch :%s, Reconstruction Error :%s" % (epoch, np.round(np.mean(self.error[:cont-1]),4)))

		 
		#epoch +=1
		#np.random.shuffle(self.x_train)

			if(self.error[cont-1]==0.0000):
				epoch = self.epochs

		return self.weights, self.a, self.b
