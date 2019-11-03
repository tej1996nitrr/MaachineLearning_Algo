#%%
def separate_byClass(dataset):
    separated =  {}
    for i in range(len(dataset)):
        vector  = dataset[i]
        class_value = vector[-1]
        if(class_value not in separated.keys() ):
            separated[class_value]=[]
             
        separated[class_value].append(vector)
    return separated


'''checking function'''
# dataset = [[3.393533211,2.331273381,0],
# 	[3.110073483,1.781539638,0],
# 	[1.343808831,3.368360954,0],
# 	[3.582294042,4.67917911,0],
# 	[2.280362439,2.866990263,0],
# 	[7.423436942,4.696522875,1],
# 	[5.745051997,3.533989803,1],
# 	[9.172168622,2.511101045,1],
# 	[7.792783481,3.424088941,1],
# 	[7.939820817,0.791637231,1]]
# separated = separate_byClass(dataset)
# separated =
# {0: [[3.393533211, 2.331273381, 0],
#   [3.110073483, 1.781539638, 0],
#   [1.343808831, 3.368360954, 0],
#   [3.582294042, 4.67917911, 0],
#   [2.280362439, 2.866990263, 0]],
#  1: [[7.423436942, 4.696522875, 1],
#   [5.745051997, 3.533989803, 1],
#   [9.172168622, 2.511101045, 1],
#   [7.792783481, 3.424088941, 1],
#   [7.939820817, 0.791637231, 1]]}
# for label in separated:
# 	print(label)
# 	for row in separated[label]:
# 		print(row)

#%%

# Calculating the mean of a list of numbers
def mean(numbers):
	return sum(numbers)/float(len(numbers))

from math import sqrt
 
# Calculating the standard deviation of a list of numbers
def stdev(numbers):
	avg = mean(numbers)
	variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
	return sqrt(variance)
#%%
'''We pass in the dataset to the zip() function with the * operator that separates the dataset (that is a list of lists) into separate lists for each row. The zip() function then iterates over each element of each row and returns a column from the dataset as a list of numbers'''

def summary(dataset):
    summaries = [(mean(column),
                 stdev(column),
                 len(column))  
                 for column in zip(*dataset)]
    del(summaries[-1])
    return summaries
# dataset = [[3.393533211,2.331273381,0],
# 	[3.110073483,1.781539638,0],
# 	[1.343808831,3.368360954,0],
# 	[3.582294042,4.67917911,0],
# 	[2.280362439,2.866990263,0],
# 	[7.423436942,4.696522875,1],
# 	[5.745051997,3.533989803,1],
# 	[9.172168622,2.511101045,1],
# 	[7.792783481,3.424088941,1],
# 	[7.939820817,0.791637231,1]]
# print(summary(dataset))
# summaries =
# [(5.178333386499999, 2.7665845055177263, 10), (2.9984683241, 1.218556343617447, 10)]
#%%
def summarize_by_class(dataset):
	separated = separate_byClass(dataset)
	summaries = dict()
	for class_value, rows in separated.items():
		summaries[class_value] = summary(rows)
	return summaries

# dataset = [[3.393533211,2.331273381,0],
# 	[3.110073483,1.781539638,0],
# 	[1.343808831,3.368360954,0],
# 	[3.582294042,4.67917911,0],
# 	[2.280362439,2.866990263,0],
# 	[7.423436942,4.696522875,1],
# 	[5.745051997,3.533989803,1],
# 	[9.172168622,2.511101045,1],
# 	[7.792783481,3.424088941,1],
# 	[7.939820817,0.791637231,1]]
# s = summarize_by_class(dataset)
#   
# s={0: [(2.7420144012, 0.9265683289298018, 5),
#   (3.0054686692, 1.1073295894898725, 5)],
#  1: [(7.6146523718, 1.2344321550313704, 5),
#   (2.9914679790000003, 1.4541931384601618, 5)]}
# for label in s:
# 	print(label)
# 	for row in s[label]:
# 		print(row)

#%%
from math import exp
def gaussianPDF(x,mean,stdev):
    exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
    return (1 / (sqrt(2 * pi) * stdev)) * exponent

#%%
from math import pi
def class_probability(summaries,row):

    total_rows =  sum([summaries[label][0][2] for label in summaries])
    probabilities = dict()
    for class_value,class_summaries in summaries.items():

        probabilities[class_value]=summaries[class_value][0][2]/float(total_rows)
        for i in range(len(class_summaries)):
            mean,stdev,count = class_summaries[i]
            probabilities[class_value]*=gaussianPDF(row[i],mean,stdev)
    return probabilities
dataset = [[3.393533211,2.331273381,0],
	[3.110073483,1.781539638,0],
	[1.343808831,3.368360954,0],
	[3.582294042,4.67917911,0],
	[2.280362439,2.866990263,0],
	[7.423436942,4.696522875,1],
	[5.745051997,3.533989803,1],
	[9.172168622,2.511101045,1],
	[7.792783481,3.424088941,1],
	[7.939820817,0.791637231,1]]

summaries = summarize_by_class(dataset)
probabilities = class_probability(summaries, dataset[0])
print(probabilities)
#{0: 0.05032427673372076, 1: 0.00011557718379945765}
#%%
'''Applying to iris dataset'''
from csv import reader
from random import seed
from random import randrange
from math import sqrt,exp,pi

def load_csv(filename):
	dataset = list()
	with open(filename, 'rt') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup

def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for _ in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores

def predict(summaries, row):
	probabilities = class_probability(summaries, row)
	best_label, best_prob = None, -1
	for class_value, probability in probabilities.items():
		if best_label is None or probability > best_prob:
			best_prob = probability
			best_label = class_value
	return best_label

def naive_bayes(train, test):
	summarize = summarize_by_class(train)
	predictions = list()
	for row in test:
		output = predict(summarize, row)
		predictions.append(output)
	return(predictions)

#%%
seed(1)
filename = 'Data\iris.data'
dataset = load_csv(filename)
for i in range(len(dataset[0])-1):
	str_column_to_float(dataset, i)
# convert class column to integers
str_column_to_int(dataset, len(dataset[0])-1)
# evaluate algorithm
n_folds = 5
scores = evaluate_algorithm(dataset, naive_bayes, n_folds)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))



#%%
# fit model
model = summarize_by_class(dataset)
# define a new record
row = [5.7,2.9,4.2,1.3]
# predict the label
label = predict(model, row)
print('Data=%s, Predicted: %s' % (row, label))
