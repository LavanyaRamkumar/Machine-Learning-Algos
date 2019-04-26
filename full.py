######################################################### Decision trees #####################################3
"""
Make the imports of python packages needed
"""
#calculate entropy
#calculate ig
#draw decision tree
import pandas as pd
import numpy as np
from pprint import pprint
#Import the dataset and define the feature as well as the target datasets / columns#
dataset = pd.read_csv('zoo.csv',
					  names=['animal_name','hair','feathers','eggs','milk',
												   'airbone','aquatic','predator','toothed','backbone',
												  'breathes','venomous','fins','legs','tail','domestic','catsize','class',])#Import all columns omitting the fist which consists the names of the animals
#We drop the animal names since this is not a good feature to split the data on
dataset=dataset.drop('animal_name',axis=1)
###########################################################################################################
def entropy(target_col):
	"""
	Calculate the entropy of a dataset.
	The only parameter of this function is the target_col parameter which specifies the target column
	"""
	#print(target_col)
	#FILL UP CODE (easy)
	elements,counts = np.unique(target_col,return_counts = True)
	#print(elements,counts)
	entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
	#print(entropy)
	return entropy
########################################################################################################### 
	
###########################################################################################################
#print("information gain for each column")
def InfoGain(data,split_attribute_name,target_name="class"):
	"""
	Calculate the information gain of a dataset. This function takes three parameters:
	1. data = The dataset for whose feature the IG should be calculated
	2. split_attribute_name = the name of the feature for which the information gain should be calculated
	3. target_name = the name of the target feature. The default for this example is "class"
	"""	
	#Calculate the entropy of the total dataset
	#FILL UP CODE (This is easy)
	total_entropy = entropy(data[target_name])
	print("entropy for class=",total_entropy)
	print("split attribute is",split_attribute_name)
	
	
	
	vals,counts= np.unique(data[split_attribute_name],return_counts=True)
	#Calculate the weighted entropy (This is easy)
	#FILL UP CODE ( This is the |Sv|/|S| * E(Sv) part)
	Weighted_Entropy = np.sum([(counts[i]/np.sum(counts))*entropy(data.where(data[split_attribute_name]==vals[i]).dropna()[target_name]) for i in range(len(vals))])
	
	
	#Calculate the information gain  (This is easy)
	#FILL UP CODE 
	Information_Gain = total_entropy - Weighted_Entropy
	
	print(split_attribute_name,"information gain is",Information_Gain)
	print()
	return Information_Gain
	   
###########################################################################################################
###########################################################################################################
def ID3(data,originaldata,features,target_attribute_name="class",parent_node_class = None):
	#(training_data,training_data,training_data.columns[:-1])
	"""
	ID3 Algorithm: This function takes five paramters:
	1. data = the data for which the ID3 algorithm should be run --> In the first run this equals the total dataset
 
	2. originaldata = This is the original dataset needed to calculate the mode target feature value of the original dataset
	in the case the dataset delivered by the first parameter is empty
	3. features = the feature space of the dataset . This is needed for the recursive call since during the tree growing process
	we have to remove features from our dataset --> Splitting at each node
	4. target_attribute_name = the name of the target attribute
	5. parent_node_class = This is the value or class of the mode target feature value of the parent node for a specific node. This is 
	also needed for the recursive call since if the splitting leads to a situation that there are no more features left in the feature
	space, we want to return the mode target feature value of the direct parent node.
	"""   
	#Define the stopping criteria --> If one of this is satisfied, we want to return a leaf node#
	
	#If all target_values have the same value, return this value
	if len(np.unique(data[target_attribute_name])) <= 1:
		return np.unique(data[target_attribute_name])[0]
	
	#If the dataset is empty, return the mode target feature value in the original dataset
	elif len(data)==0:
		return np.unique(originaldata[target_attribute_name])[np.argmax(np.unique(originaldata[target_attribute_name],return_counts=True)[1])]
	
	#If the feature space is empty, return the mode target feature value of the direct parent node --> Note that
	#the direct parent node is that node which has called the current run of the ID3 algorithm and hence
	#the mode target feature value is stored in the parent_node_class variable.
	
	elif len(features) ==0:
		return parent_node_class
	
	#If none of the above holds true, grow the tree!
	
	else:
		#Set the default value for this node --> The mode target feature value of the current node
		parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name],return_counts=True)[1])]
		
		#Select the feature which best splits the dataset
		item_values = [InfoGain(data,feature,target_attribute_name) for feature in features] #Return the information gain values for the features in the dataset
		best_feature_index = np.argmax(item_values)
		best_feature = features[best_feature_index]
		
		#Create the tree structure. The root gets the name of the feature (best_feature) with the maximum information
		#gain in the first run
		tree = {best_feature:{}}
		
		
		#Remove the feature with the best inforamtion gain from the feature space
		features = [i for i in features if i != best_feature]
		
		#Grow a branch under the root node for each possible value of the root node feature
		
		for value in np.unique(data[best_feature]):
			value = value
			#Split the dataset along the value of the feature with the largest information gain and therwith create sub_datasets
			sub_data = data.where(data[best_feature] == value).dropna()
			
			#Call the ID3 algorithm for each of those sub_datasets with the new parameters --> Here the recursion comes in!
			subtree = ID3(sub_data,dataset,features,target_attribute_name,parent_node_class)
			
			#Add the sub tree, grown from the sub_dataset to the tree under the root node
			tree[best_feature][value] = subtree
			
		return(tree)	
				
###########################################################################################################
###########################################################################################################
	
	
def predict(query,tree,default = 1):
	"""
	Prediction of a new/unseen query instance. This takes two parameters:
	1. The query instance as a dictionary of the shape {"feature_name":feature_value,...}
	2. The tree 
	We do this also in a recursive manner. That is, we wander down the tree and check if we have reached a leaf or if we are still in a sub tree. 
	Since this is a important step to understand, the single steps are extensively commented below.
	1.Check for every feature in the query instance if this feature is existing in the tree.keys() for the first call, 
	tree.keys() only contains the value for the root node 
	--> if this value is not existing, we can not make a prediction and have to 
	return the default value which is the majority value of the target feature
	2. First of all we have to take care of a important fact: Since we train our model with a database A and then show our model
	a unseen query it may happen that the feature values of these query are not existing in our tree model because non of the
	training instances has had such a value for this specific feature. 
	For instance imagine the situation where your model has only seen animals with one to four
	legs - The "legs" node in your model will only have four outgoing branches (from one to four). If you now show your model
	a new instance (animal) which has for the legs feature the vale 5, you have to tell your model what to do in such a 
	situation because otherwise there is no classification possible because in the classification step you try to 
	run down the outgoing branch with the value 5 but there is no such a branch. Hence: Error and no Classification!
	We can address this issue with a classification value of for instance (999) which tells us that there is no classification
	possible or we assign the most frequent target feature value of our dataset used to train the model. Or, in for instance 
	medical application we can return the most worse case - just to make sure... 
	We can also return the most frequent value of the direct parent node. To make a long story short, we have to tell the model 
	what to do in this situation.
	In our example, since we are dealing with animal species where a false classification is not that critical, we will assign
	the value 1 which is the value for the mammal species (for convenience).
	3. Address the key in the tree which fits the value for key --> Note that key == the features in the query. 
	Because we want the tree to predict the value which is hidden under the key value (imagine you have a drawn tree model on 
	the table in front of you and you have a query instance for which you want to predict the target feature 
	- What are you doing? - Correct:
	You start at the root node and wander down the tree comparing your query to the node values. Hence you want to have the
	value which is hidden under the current node. If this is a leaf, perfect, otherwise you wander the tree deeper until you
	get to a leaf node. 
	Though, you want to have this "something" [either leaf or sub_tree] which is hidden under the current node
	and hence we must address the node in the tree which == the key value from our query instance. 
	This is done with tree[keys]. Next you want to run down the branch of this node which is equal to the value given "behind"
	the key value of your query instance e.g. if you find "legs" == to tree.keys() that is, for the first run == the root node.
	You want to run deeper and therefore you have to address the branch at your node whose value is == to the value behind key.
	This is done with query[key] e.g. query[key] == query['legs'] == 0 --> Therewith we run down the branch of the node with the
	value 0. Summarized, in this step we want to address the node which is hidden behind a specific branch of the root node (in the first run)
	this is done with: result = [key][query[key]]
	4. As said in the 2. step, we run down the tree along nodes and branches until we get to a leaf node.
	That is, if result = tree[key][query[key]] returns another tree object (we have represented this by a dict object --> 
	that is if result is a dict object) we know that we have not arrived at a root node and have to run deeper the tree. 
	Okay... Look at your drawn tree in front of you... what are you doing?...well, you run down the next branch... 
	exactly as we have done it above with the slight difference that we already have passed a node and therewith 
	have to run only a fraction of the tree --> You clever guy! That "fraction of the tree" is exactly what we have stored
	under 'result'.
	So we simply call our predict method using the same query instance (we do not have to drop any features from the query
	instance since for instance the feature for the root node will not be available in any of the deeper sub_trees and hence 
	we will simply not find that feature) as well as the "reduced / sub_tree" stored in result.
	SUMMARIZED: If we have a query instance consisting of values for features, we take this features and check if the 
	name of the root node is equal to one of the query features.
	If this is true, we run down the root node outgoing branch whose value equals the value of query feature == the root node.
	If we find at the end of this branch a leaf node (not a dict object) we return this value (this is our prediction).
	If we instead find another node (== sub_tree == dict objct) we search in our query for the feature which equals the value 
	of that node. Next we look up the value of our query feature and run down the branch whose value is equal to the 
	query[key] == query feature value. And as you can see this is exactly the recursion we talked about
	with the important fact that for each node we run down the tree, we check only the nodes and branches which are 
	below this node and do not run the whole tree beginning at the root node 
	--> This is why we re-call the classification function with 'result'
	"""
	result = ""
	#READ THE ABOVE COMMENTS AND FILL UP CODE. (Read comments carefully)
	#1.
	for key in list(query.keys()):
		if key in list(tree.keys()):
			#2.
			try:
				result = tree[key][query[key]] 
			except:
				return default
  
			#3.
			result = tree[key][query[key]]
			#4.
			if isinstance(result,dict):
				return predict(query,result)
			else:
				return result
	
	
	
	return result
		
		
"""
Check the accuracy of our prediction.
The train_test_split function takes the dataset as parameter which should be divided into
a training and a testing set. The test function takes two parameters, which are the testing data as well as the tree model.
"""
###########################################################################################################
###########################################################################################################
def train_test_split(dataset):
	training_data = dataset.iloc[:80].reset_index(drop=True)#We drop the index respectively relabel the index
	#starting form 0, because we do not want to run into errors regarding the row labels / indexes
	testing_data = dataset.iloc[80:].reset_index(drop=True)
	return training_data,testing_data
training_data = train_test_split(dataset)[0]
testing_data = train_test_split(dataset)[1] 
def test(data,tree):
	#Create new query instances by simply removing the target feature column from the original dataset and 
	#convert it to a dictionary
	queries = data.iloc[:,:-1].to_dict(orient = "records")
	
	#Create a empty DataFrame in whose columns the prediction of the tree are stored
	predicted = pd.DataFrame(columns=["predicted"]) 
	
	#Calculate the prediction accuracy
	for i in range(len(data)):
		predicted.loc[i,"predicted"] = predict(queries[i],tree,1.0) 
	print('The prediction accuracy is: ',(np.sum(predicted["predicted"] == data["class"])/len(data))*100,'%')
	
"""
Train the tree, Print the tree and predict the accuracy
"""
tree = ID3(training_data,training_data,training_data.columns[:-1])
pprint(tree)
test(testing_data,tree)

























########################################################## genetic #########################################################
import numpy
import random

#find the sum of 5 weights w1, w2, w3, w4 , w5 such that x1w1 + x2w2 + x3w3 + x4w4 + x5w5 = N, N = some number (Integer)

MUTATION_PROBABILITY = 0.04 # initial value

X = numpy.array([1, 2, 3, 4])

DECAY_RATE = 0.003 # reduce mutation probability , so that changes of mutation are less at endings

N = 30

NB_MOST_FIT = 4

class Member : 

    def __init__(self, genes, mutation_probability) :
        
        self.genes = genes
        self.nb_genes = len(self.genes)

        if mutation_probability > random.random() : 

            #select any 3 random genes for mutation :
            random_genes = numpy.random.randint(low = 0, high = self.nb_genes, size = (4))

            for i in random_genes :

                self.genes[i] = self.genes[i] + random.randint(a = 1, b = 50)


    def fitness_score(self, N) :
        LHS = numpy.matmul(X.T , self.genes)
        if(LHS==N):
        	print(X.T,self.genes,LHS,numpy.fabs(LHS - N))
        return numpy.fabs(LHS - N)
    
    def __str__(self):

        return "Member "+str(self.genes) + " LEAST ERROR  : "+str(self.fitness_score(N))
    

class Population : 

    #a population is a set of members, We limit population size to 5, so there are 5 members
    #select ration is 3 : 2 while crossover
    def __init__(self, nb_popsize = 10, nb_genesize = 4, select = 3, callback = None) :

        self.nb_popsize = nb_popsize
        self.nb_genesize = nb_genesize
        self.callback = callback

        self.population = self.create()
    
    def create(self):

        #creates a pouplation initially with random values for genes : 
        members = []
        for i in range(self.nb_popsize) :
            genes = numpy.random.randint(low = 1, high = 30, size = (self.nb_genesize))
            member = Member(genes, mutation_probability = MUTATION_PROBABILITY)
            members.append(member)
        
        return members

    
    def grade(self) :

        #obtain fitness scores and returns most fit members from population : 
        #and one unfit member

        fit_members = []
        temp = []
        for i in range(self.nb_popsize) :
            fitness, member = self.population[i].fitness_score(N), i
            temp.append((fitness, i))

        temp.sort()

        #get last 4 members who are most fit
        for p_index in range(NB_MOST_FIT) :

            i = temp[p_index][1]

            fit_members.append(self.population[i])

        #and add one unfit member : 
        rand_unfit = random.randint(a = NB_MOST_FIT, b = len(self.population) - 1)
        fit_members.append(self.population[temp[rand_unfit][1]])
        
        return fit_members
    
    def crossover_policy(self, dad, mom) :

        #Simple crossover scheme, use your own techniques for optimization: 
        child_genes = []
        for i in range(len(dad.genes)) :
            gene = None
            if i % 2 == 0 :
                gene = dad.genes[i]
            else : 
                gene = mom.genes[i]

            child_genes.append(gene)
        
        return Member(numpy.array(child_genes), mutation_probability = MUTATION_PROBABILITY)

    

    def crossover(self, old_fit_members) :

        #there is a requirement of some children, allow parents to reproduce
        requirement = self.nb_popsize - len(old_fit_members)

        #generate random pairs and mutate them :
        children = []
        while requirement > 0 :
            dad = random.randint(a = 0, b = len(old_fit_members) - 1)
            mom = random.randint(a = 0, b = len(old_fit_members) - 1)

            #print(mom, dad)

            if dad != mom : 
                child = self.crossover_policy(old_fit_members[dad], old_fit_members[mom])
                requirement = requirement - 1
                children.append(child)
        
        new_generation = old_fit_members + children
        return new_generation
    
    def next_generation(self) :

        global MUTATION_PROBABILITY

        fits = self.grade()
        population = self.crossover(fits)
        self.population = population
        

#test

population = Population()

NB_EPH = 1000
for i in range(NB_EPH) :
    population.next_generation()


#print optimal solutions : 

print('Population after '+ str(NB_EPH) +' iterations : ')

scores = []

for p in population.population : 
    scores.append(p.fitness_score(N))
    print(p)

print('Error factor : (+ or - error will give the solution) ', numpy.amin(scores))


        
    


        





            


    



















########################################################	NN 		###################################################
import random
import csv
from random import seed
from random import randrange
from random import random
from math import exp
import numpy as np


# Loading seeds dataset
def load_csv(filename):
	with open(filename, 'r') as file:
		lines = csv.reader(file)
		dataset=list(lines)
	for i in dataset:
		i.pop(0)
	#print(dataset)
	return dataset
 
# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column])
 
# Convert string column to integer
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	#print(unique)
	lookup = dict()
	for i, value in enumerate(unique):
		#print(i,value)
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	#print(lookup)
#	print(dataset)
	return lookup
 
# Find the min and max values for each column
def dataset_minmax(dataset):
	minmax = list()
	stats = [[min(column), max(column)] for column in zip(*dataset)]
	#print(*dataset)
	return stats
 
# Rescale dataset columns to the range 0-1. This is mini-max normalization
def normalize_dataset(dataset, minmax):
	#for row in dataset:
	#	for i in range(len(row)-1):
	#		row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
	#print(dataset)
	print('')
 
# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
		network = list()
		hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
		network.append(hidden_layer)
		output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
		network.append(output_layer)
#		print(network)
		return network

# Calculate neuron activation for an input
def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation

# Transfer neuron activation
#Perform activation through a SIGMOID UNIT
def transfer(activation):
	#Your code here. FILL UP THIS CODE-PART. EXERCISE
	
	#sigmoid
	return (1/float(1+np.exp(-activation)))
	
	#tanh
	#return np.tanh(activation)
	
	#relu
	#return max(0,activation)

# Forward propagate input to a network output
def forward_propagate(network, row):
	#print('forward_propagate(',network,',', row,'):')
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs
 
# Calculate the derivative of a neuron output
def transfer_derivative(output):
	#FILL UP THIS PART OF THE CODE. EXERCISE
	
	#sigmoid
	return (output)*(1-output)
	
	#tanh
	#return (1-(output*output))
	
	#relu
	#if(output<=0):
		#return 0
	#elif(output>0):
		#return 1
	
 
# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		#For Hidden layer
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		#For Output layer
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(expected[j] - neuron['output'])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])
 
# Update network weights with error
def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] += l_rate * neuron['delta']
 
# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
	for epoch in range(n_epoch):
		print("Epoch num: ", epoch)
		for row in train:
			outputs = forward_propagate(network, row)
			expected = [0 for i in range(n_outputs)]
			expected[row[-1]] = 1
			backward_propagate_error(network, expected)
			update_weights(network, row, l_rate)

 # Make a prediction with a network
def predict(network, row):
	outputs = forward_propagate(network, row)
	return outputs.index(max(outputs))
 
# Backpropagation Algorithm With Stochastic Gradient Descent
def applying(train, test, l_rate, n_epoch, n_hidden):
	n_inputs = len(train[0]) - 1
	n_outputs = len(set([row[-1] for row in train]))
	network = initialize_network(n_inputs, n_hidden, n_outputs)
	train_network(network, train, l_rate, n_epoch, n_outputs)
	predictions = list()
	for row in test:
		prediction = predict(network, row)
		predictions.append(prediction)
	return(predictions)
 
#Using all the functions
seed(1)


#Loading and preprocessing data
filename = 'forest_data.csv'
dataset = load_csv(filename)

for i in range(len(dataset[0])-1):
	str_column_to_float(dataset, i)
	
#Converting class column to integers
str_column_to_int(dataset, len(dataset[0])-1)

#Normalize input variables
minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)


n_folds = 10
l_rate = .45
n_epoch = 120
n_hidden = 9
folds = cross_validation_split(dataset, n_folds)

for fold in folds:
	train_set = list(folds)
	train_set.remove(fold)
	train_set = sum(train_set, [])
	test_set = list()
	for row in fold:
		row_copy = list(row)
		test_set.append(row_copy)
		row_copy[-1] = None
		actual = [row[-1] for row in fold]

result=applying(train_set,test_set,l_rate,n_epoch,n_hidden)

#print(result)
acc=accuracy_metric(actual,result)
print(" Accuracy is ",acc)























##################################################SVM ###############################################################################
# coding: utf-8

# In[1]:


import numpy as np

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
# This line is only needed if you have a HiDPI display
get_ipython().magic("config InlineBackend.figure_format = 'retina'")

from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.preprocessing import StandardScaler


# In[2]:


class SMOModel:
    """Container object for the model used for sequential minimal optimization."""
    
    def __init__(self, X, y, C, kernel, alphas, b, errors):
        self.X = X               # training data vector
        self.y = y               # class label vector
        self.C = C               # regularization parameter
        self.kernel = kernel     # kernel function
        self.alphas = alphas     # lagrange multiplier vector
        self.b = b               # scalar bias term
        self.errors = errors     # error cache
        self._obj = []           # record of objective function value
        self.m = len(self.X)     # store size of training set


# In[3]:


def linear_kernel(x, y, b=1):
    """Returns the linear combination of arrays `x` and `y` with
    the optional bias term `b` (set to 1 by default)."""
    
    return x @ y.T + b # Note the @ operator for matrix multiplication


def gaussian_kernel(x, y, sigma=1):
    """Returns the gaussian similarity of arrays `x` and `y` with
    kernel width parameter `sigma` (set to 1 by default)."""
    
    if np.ndim(x) == 1 and np.ndim(y) == 1:
        result = np.exp(- (np.linalg.norm(x - y, 2)) ** 2 / (2 * sigma ** 2))
    elif (np.ndim(x) > 1 and np.ndim(y) == 1) or (np.ndim(x) == 1 and np.ndim(y) > 1):
        result = np.exp(- (np.linalg.norm(x - y, 2, axis=1) ** 2) / (2 * sigma ** 2))
    elif np.ndim(x) > 1 and np.ndim(y) > 1:
        result = np.exp(- (np.linalg.norm(x[:, np.newaxis] - y[np.newaxis, :], 2, axis=2) ** 2) / (2 * sigma ** 2))
    return result


# In[4]:


x_len, y_len = 5, 10


# In[5]:


linear_kernel(np.random.rand(x_len, 1), np.random.rand(y_len, 1)).shape == (x_len,y_len)



# In[6]:


gaussian_kernel(np.random.rand(x_len, 1), np.random.rand(y_len, 1)).shape == (5,10)


# In[7]:


def objective_function(alphas, target, kernel, X_train):
    """Returns the SVM objective function based in the input model defined by:
    `alphas`: vector of Lagrange multipliers
    `target`: vector of class labels (-1 or 1) for training data
    `kernel`: kernel function
    `X_train`: training data for model."""
    
    return np.sum(alphas) - 0.5 * np.sum((target[:, None] * target[None, :]) * kernel(X_train, X_train) * (alphas[:, None] * alphas[None, :]))


# Decision function

def decision_function(alphas, target, kernel, X_train, x_test, b):
    """Applies the SVM decision function to the input feature vectors in `x_test`."""
    
    result = (alphas * target) @ kernel(X_train, x_test) - b
    return result


# In[38]:


def plot_decision_boundary(model, ax, resolution=100, colors=('b', 'k', 'r'), levels=(-1, 0, 1)):
        """Plots the model's decision boundary on the input axes object.
        Range of decision boundary grid is determined by the training data.
        Returns decision boundary grid and axes object (`grid`, `ax`)."""
        # Generate coordinate grid of shape [resolution x resolution]
        # and evaluate the model over the entire space
        xrange = np.linspace(model.X[:,0].min(), model.X[:,0].max(), resolution)
        yrange = np.linspace(model.X[:,1].min(), model.X[:,1].max(), resolution)
        grid = [[decision_function(model.alphas, model.y,
                                   model.kernel, model.X,
                                   np.array([xr, yr]), model.b) for xr in xrange] for yr in yrange]
        grid = np.array(grid).reshape(len(xrange), len(yrange))
        
        # Plot decision contours using grid and
        # make a scatter plot of training data
        ax.contour(xrange, yrange, grid, levels=levels, linewidths=(1, 1, 1),
                   linestyles=('--', '-', '--'), colors=colors)
        ax.scatter(model.X[:,0], model.X[:,1],
                   c=model.y, cmap=plt.cm.viridis, lw=0, alpha=0.25)
        
        # Plot support vectors (non-zero alphas)
        # as circled points (linewidth > 0)
        mask = np.round(model.alphas, decimals=2) != 0.0
        xv=model.X[mask,0]
        yv=model.X[mask,1]
        av=output.alphas[mask]
        print("Support Vectors and alpha values:")
        for i in range(len(xv)):
            print("[%f,%f]\t %f"%(xv[i],yv[i],av[i]))
        ax.scatter(model.X[mask,0], model.X[mask,1],
                   c=model.y[mask], cmap=plt.cm.viridis, lw=1, edgecolors='k')
        
        return grid, ax



# In[9]:


def take_step(i1, i2, model):
    
    # Skip if chosen alphas are the same
    if i1 == i2:
        return 0, model
    
    alph1 = model.alphas[i1]
    alph2 = model.alphas[i2]
    y1 = model.y[i1]
    y2 = model.y[i2]
    E1 = model.errors[i1]
    E2 = model.errors[i2]
    s = y1 * y2
    
    # Compute L & H, the bounds on new possible alpha values
    if (y1 != y2):
        L = max(0, alph2 - alph1)
        H = min(model.C, model.C + alph2 - alph1)
    elif (y1 == y2):
        L = max(0, alph1 + alph2 - model.C)
        H = min(model.C, alph1 + alph2)
    if (L == H):
        return 0, model

    # Compute kernel & 2nd derivative eta
    k11 = model.kernel(model.X[i1], model.X[i1])
    k12 = model.kernel(model.X[i1], model.X[i2])
    k22 = model.kernel(model.X[i2], model.X[i2])
    eta = 2 * k12 - k11 - k22
    
    # Compute new alpha 2 (a2) if eta is negative
    if (eta < 0):
        a2 = alph2 - y2 * (E1 - E2) / eta
        # Clip a2 based on bounds L & H
        if L < a2 < H:
            a2 = a2
        elif (a2 <= L):
            a2 = L
        elif (a2 >= H):
            a2 = H
            
    # If eta is non-negative, move new a2 to bound with greater objective function value
    else:
        alphas_adj = model.alphas.copy()
        alphas_adj[i2] = L
        # objective function output with a2 = L
        Lobj = objective_function(alphas_adj, model.y, model.kernel, model.X) 
        alphas_adj[i2] = H
        # objective function output with a2 = H
        Hobj = objective_function(alphas_adj, model.y, model.kernel, model.X)
        if Lobj > (Hobj + eps):
            a2 = L
        elif Lobj < (Hobj - eps):
            a2 = H
        else:
            a2 = alph2
            
    # Push a2 to 0 or C if very close
    if a2 < 1e-8:
        a2 = 0.0
    elif a2 > (model.C - 1e-8):
        a2 = model.C
    
    # If examples can't be optimized within epsilon (eps), skip this pair
    if (np.abs(a2 - alph2) < eps * (a2 + alph2 + eps)):
        return 0, model
    
    # Calculate new alpha 1 (a1)
    a1 = alph1 + s * (alph2 - a2)
    
    # Update threshold b to reflect newly calculated alphas
    # Calculate both possible thresholds
    b1 = E1 + y1 * (a1 - alph1) * k11 + y2 * (a2 - alph2) * k12 + model.b
    b2 = E2 + y1 * (a1 - alph1) * k12 + y2 * (a2 - alph2) * k22 + model.b
    
    # Set new threshold based on if a1 or a2 is bound by L and/or H
    if 0 < a1 and a1 < C:
        b_new = b1
    elif 0 < a2 and a2 < C:
        b_new = b2
    # Average thresholds if both are bound
    else:
        b_new = (b1 + b2) * 0.5

    # Update model object with new alphas & threshold
    model.alphas[i1] = a1
    model.alphas[i2] = a2
    
    # Update error cache
    # Error cache for optimized alphas is set to 0 if they're unbound
    for index, alph in zip([i1, i2], [a1, a2]):
        if 0.0 < alph < model.C:
            model.errors[index] = 0.0
    
    # Set non-optimized errors based on equation 12.11 in Platt's book
    non_opt = [n for n in range(model.m) if (n != i1 and n != i2)]
    model.errors[non_opt] = model.errors[non_opt] +                             y1*(a1 - alph1)*model.kernel(model.X[i1], model.X[non_opt]) +                             y2*(a2 - alph2)*model.kernel(model.X[i2], model.X[non_opt]) + model.b - b_new
    
    # Update model threshold
    model.b = b_new
    
    return 1, model


# In[10]:


def examine_example(i2, model):
    
    y2 = model.y[i2]
    alph2 = model.alphas[i2]
    E2 = model.errors[i2]
    r2 = E2 * y2

    # Proceed if error is within specified tolerance (tol)
    if ((r2 < -tol and alph2 < model.C) or (r2 > tol and alph2 > 0)):
        
        if len(model.alphas[(model.alphas != 0) & (model.alphas != model.C)]) > 1:
            # Use 2nd choice heuristic is choose max difference in error
            if model.errors[i2] > 0:
                i1 = np.argmin(model.errors)
            elif model.errors[i2] <= 0:
                i1 = np.argmax(model.errors)
            step_result, model = take_step(i1, i2, model)
            if step_result:
                return 1, model
            
        # Loop through non-zero and non-C alphas, starting at a random point
        for i1 in np.roll(np.where((model.alphas != 0) & (model.alphas != model.C))[0],
                          np.random.choice(np.arange(model.m))):
            step_result, model = take_step(i1, i2, model)
            if step_result:
                return 1, model
        
        # loop through all alphas, starting at a random point
        for i1 in np.roll(np.arange(model.m), np.random.choice(np.arange(model.m))):
            step_result, model = take_step(i1, i2, model)
            if step_result:
                return 1, model
    
    return 0, model


# In[11]:


def train(model):
    
    numChanged = 0
    examineAll = 1

    while(numChanged > 0) or (examineAll):
        numChanged = 0
        if examineAll:
            # loop over all training examples
            for i in range(model.alphas.shape[0]):
                examine_result, model = examine_example(i, model)
                numChanged += examine_result
                if examine_result:
                    obj_result = objective_function(model.alphas, model.y, model.kernel, model.X)
                    model._obj.append(obj_result)
        else:
            # loop over examples where alphas are not already at their limits
            for i in np.where((model.alphas != 0) & (model.alphas != model.C))[0]:
                examine_result, model = examine_example(i, model)
                numChanged += examine_result
                if examine_result:
                    obj_result = objective_function(model.alphas, model.y, model.kernel, model.X)
                    model._obj.append(obj_result)
        if examineAll == 1:
            examineAll = 0
        elif numChanged == 0:
            examineAll = 1
        
    return model


# In[69]:


X_train, y = make_blobs(n_samples=1000, centers=2,
                        n_features=2, random_state=1)


# In[70]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train, y)


# In[71]:


y[y == 0] = -1


# In[175]:


# Set model parameters and initial values

C = 1000.0
m = len(X_train_scaled)
initial_alphas = np.zeros(m)
initial_b = 0.0

# Set tolerances
tol = 0.01 # error tolerance
eps = 0.01 # alpha tolerance

# Instantiate model
model = SMOModel(X_train_scaled, y, C, linear_kernel,
                 initial_alphas, initial_b, np.zeros(m))

# Initialize error cache
initial_error = decision_function(model.alphas, model.y, model.kernel,
                                  model.X, model.X, model.b) - model.y
model.errors = initial_error


# In[176]:


np.random.seed(0)
output = train(model)


# In[177]:


print("C=1000")
fig, ax = plt.subplots()
grid, ax = plot_decision_boundary(output, ax)


# In[57]:


for i in output.alphas:
    if(i>0.1e-10):
        print(i)


























######################################################NAIVE BAYES########################
import os
import re
import string
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
DATA_DIR = 'enron'
target_names = ['ham', 'spam']
 
def get_data(DATA_DIR,x):
	#if(x==1):
	#	print("For datasets 1,2,3")
	#	subfolders = ['enron%d' % i for i in range(1,4)]
	#else:
	#	print("For datasets 4,5,6")
	#	subfolders = ['enron%d' % i for i in range(4,7)]
	subfolders = ['enron%d' % i for i in range(1,7)]
 
	data = []
	target = []
	for subfolder in subfolders:
		# spam
		spam_files = os.listdir(os.path.join(DATA_DIR, subfolder, 'spam'))
		for spam_file in spam_files:
			with open(os.path.join(DATA_DIR, subfolder, 'spam', spam_file), encoding="latin-1") as f:
				data.append(f.read())
				target.append(1)
 
		# ham
		ham_files = os.listdir(os.path.join(DATA_DIR, subfolder, 'ham'))
		for ham_file in ham_files:
			with open(os.path.join(DATA_DIR, subfolder, 'ham', ham_file), encoding="latin-1") as f:
				data.append(f.read())

				target.append(0)
 
	return data, target

class SpamDetector(object):
	"""Implementation of Naive Bayes for binary classification"""
	def clean(self, s):
		translator = str.maketrans("", "", string.punctuation)
		return s.translate(translator)
 
	def tokenize(self, text):
		text = self.clean(text).lower()
		return re.split("\W+", text)
 
	def get_word_counts(self, words):
		word_counts = {}
		for word in words:
			word_counts[word] = word_counts.get(word, 0.0) + 1.0
		return word_counts
	
	def fit(self, X, Y):
		self.num_messages = {}
		self.log_class_priors = {}
		self.word_counts = {}
		self.vocab = set()
	 
		n = len(X)
		self.num_messages['spam'] = sum(1 for label in Y if label == 1)
		self.num_messages['ham'] = sum(1 for label in Y if label == 0)
		#self.log_class_priors['spam'] = math.log(self.num_messages['spam'] / n)
		#self.log_class_priors['ham'] = math.log(self.num_messages['ham'] / n)
		
		self.log_class_priors['spam'] = self.num_messages['spam'] / n
		self.log_class_priors['ham'] = self.num_messages['ham'] / n
		self.word_counts['spam'] = {}
		self.word_counts['ham'] = {}
	 
		for x, y in zip(X, Y):
			c = 'spam' if y == 1 else 'ham'
			counts = self.get_word_counts(self.tokenize(x))
			for word, count in counts.items():
				if word not in self.vocab:
					self.vocab.add(word)
				if word not in self.word_counts[c]:
					self.word_counts[c][word] = 0.0
	 
				self.word_counts[c][word] += count
			
			
	def predict(self, X,n):
		result = []
		for x in X:
			counts = self.get_word_counts(self.tokenize(x))
			spam_score = 0
			ham_score = 0
			for word, _ in counts.items():
				if word not in self.vocab: continue
				
				# add Laplace smoothing
				log_w_given_spam = math.log((self.word_counts['spam'].get(word, 0.0) + n) / (self.num_messages['spam'] +n*len(self.vocab)))
				log_w_given_ham =  math.log((self.word_counts['ham'].get(word, 0.0) + n) / (self.num_messages['ham'] + n*len(self.vocab)))
				
				#log_w_given_spam = (self.word_counts['spam'].get(word, 0.0) + n) / (self.num_messages['spam'] +n*len(self.vocab))
				#log_w_given_ham =  (self.word_counts['ham'].get(word, 0.0) + n) / (self.num_messages['ham'] + n*len(self.vocab))
	 
				spam_score += log_w_given_spam
				ham_score += log_w_given_ham
	 
			spam_score += self.log_class_priors['spam']
			ham_score += self.log_class_priors['ham']
	 
			if spam_score > ham_score:
				result.append(1)
			else:
				result.append(0)
		return result

if __name__ == '__main__':
	
	for j in [1,2,3]:
		print("Add-{} smoothing".format(str(j)))
		#print("Without log")
		X, y = get_data(DATA_DIR,j)
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
		MNB = SpamDetector()
		MNB.fit(X_train, y_train)
 
		pred = MNB.predict(X_test,j)
		true = y_test
 
		accuracy = sum(1 for i in range(len(pred)) if pred[i] == true[i]) / float(len(pred))
		print("{0:.4f}".format(accuracy))
		print(confusion_matrix(true,pred)) 






















############################################HMM###################################################
import numpy as np
import datetime as dt
from nsepy import get_history
from matplotlib import pyplot

class HMM(object):
	# Implements discrete 1-st order Hidden Markov Model 
	def __init__(self):
		pass

	def forward(self, pi, A, O, observations):
		N = len(observations)
		S = len(pi)
		alpha = np.zeros((N, S))

		# base case
		alpha[0, :] = pi * O[:,observations[0]]
		
		# recursive case
		for i in range(1, N):
			for s2 in range(S):
				for s1 in range(S):
					alpha[i, s2] += alpha[i-1, s1] * A[s1, s2] * O[s2, observations[i]]
		
		return (alpha, np.sum(alpha[N-1,:]))

	def backward(self, pi, A, O, observations):
		N = len(observations)
		S = len(pi)
		beta = np.zeros((N, S))
		
		# base case
		beta[N-1, :] = 1
		
		# recursive case
		for i in range(N-2, -1, -1):
			for s1 in range(S):
				for s2 in range(S):
					beta[i, s1] += beta[i+1, s2] * A[s1, s2] * O[s2, observations[i+1]]
		
		return (beta, np.sum(pi * O[:, observations[0]] * beta[0,:]))


	def baum_welch(self, o, N, rand_seed=1):
		# Implements HMM Baum-Welch algorithm		
		T = len(o[0])
		M = int(max(o[0]))+1 # now all hist time-series will contain all observation vals, but we have to provide for all

		# Initialise A, B and pi randomly, but so that they sum to one
		np.random.seed(rand_seed)
			
		pi_randomizer = np.ndarray.flatten(np.random.dirichlet(np.ones(N),size=1))/100
		pi=1.0/N*np.ones(N)-pi_randomizer

		a_randomizer = np.random.dirichlet(np.ones(N),size=N)/100
		a=1.0/N*np.ones([N,N])-a_randomizer

		b_randomizer=np.random.dirichlet(np.ones(M),size=N)/100
		b = 1.0/M*np.ones([N,M])-b_randomizer

		pi, A, O = np.copy(pi), np.copy(a), np.copy(b) # take copies, as we modify them
		S = pi.shape[0]
		iterations = 1000
		training = o
		# do several steps of EM hill climbing
		for it in range(iterations):
			pi1 = np.zeros_like(pi)
			A1 = np.zeros_like(A)
			O1 = np.zeros_like(O)
			
			for observations in training:
				# compute forward-backward matrices
				alpha, za = self.forward(pi, A, O, observations)
				beta, zb = self.backward(pi, A, O, observations)
				assert abs(za - zb) < 1e-6, "it's badness 10000 if the marginals don't agree"
				
				# M-step here, calculating the frequency of starting state, transitions and (state, obs) pairs
				pi1 += alpha[0,:] * beta[0,:] / za
				for i in range(0, len(observations)):
					O1[:, observations[i]] += alpha[i,:] * beta[i,:] / za
				for i in range(1, len(observations)):
					for s1 in range(S):
						for s2 in range(S):
							A1[s1, s2] += alpha[i-1,s1] * A[s1, s2] * O[s2, observations[i]] * beta[i,s2] / za
																		
			# normalise pi1, A1, O1
			pi = pi1 / np.sum(pi1)
			for s in range(S):
				A[s, :] = A1[s, :] / np.sum(A1[s, :])
				O[s, :] = O1[s, :] / np.sum(O1[s, :])
		return pi, A, O

	def predict(self, stock_prices, states=3):
		(pi, A, O) = self.baum_welch(np.array([stock_prices]), states)
		(alpha, c) = self.forward(pi, A, O, stock_prices)
		# normalize alpha
		row_sums = alpha.sum(axis=1)
		matrix_1 = alpha / row_sums[:, np.newaxis]
		# probability distribution of last hidden state given data
		matrix_2 = matrix_1[-1, :]
		# probability distribution of last hidden state given data
		matrix_3 = np.matmul(matrix_2, A)
		# probabilty distribution of predicted observation state given past observations
		matrix_4= np.matmul(matrix_3, O)
		return(np.argmax(matrix_4))
		
def get_stock_prices(company_symbol, start_date, end_date):
	# stock price data from nsepy library (closing prices)
	start_date = dt.datetime.strptime(start_date, '%Y-%m-%d').date()
	end_date = dt.datetime.strptime(end_date, '%Y-%m-%d').date()
	stock_prices = get_history(symbol=company_symbol, start=start_date, end=end_date)
	# pandas dataframe to numpy array
	stock_prices.iloc[:,-1].plot()
	pyplot.show()
	stock_prices = stock_prices.values
	
	# return closing prices
	#print(stock_prices)
	
	return stock_prices[:,7]
		
def get_price_movements(stock_prices):
	price_change = stock_prices[1:] - stock_prices[:-1]
	price_movement = np.array(list(map((lambda x: 1 if x>0 else 0), price_change)))
	print(price_movement)
	return price_movement

if __name__ == '__main__':

	hmm = HMM()
	stock_prices = get_stock_prices('SBIN', '2019-01-01', '2019-03-21')
	price_movement = get_price_movements(stock_prices)
	prediction = hmm.predict(price_movement, 5)
	if prediction==1:
		print("You should buy stock.")
	else:
		print("Sell stock if you have.")
