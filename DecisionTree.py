#%%
'''Calculating gini index'''
def gini_index(groups,classes):
    n_instances = float(sum([len(group) for group in groups]))
    gini=0.0
    for group in groups:
        size = float(len(group))
        if size==0:
            continue
        score=0.0
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            
            score += p * p
        gini += (1.0 - score) * (size / n_instances)
    return gini   
# print(gini_index([[[1, 1], [1, 0]], [[1, 1], [1, 0]]], [0, 1]))
# print(gini_index([[[1, 0], [1, 0]], [[1, 1], [1, 1]]], [0, 1]))         

# %%
''' iterating over each row, checking if the attribute value is below or above the split value and assigning it to the left or right group respectively.'''

def test_split(index,value,dataset):
    left,right=[],[]
    for row in dataset:
        if row[index]<value:
            left.append(row)
        else:
            right.append(row)
    return left,right


# %%
def get_split(dataset):
	class_values = list(set(row[-1] for row in dataset))
	b_index, b_value, b_score, b_groups = 999, 999, 999, None
	for index in range(len(dataset[0])-1):
		for row in dataset:
			groups = test_split(index, row[index], dataset)
			gini = gini_index(groups, class_values)
			print('X%d < %.3f Gini=%.3f' % ((index+1), row[index], gini))
			if gini < b_score:
				b_index, b_value, b_score, b_groups = index, row[index], gini, groups
	return {'index':b_index, 'value':b_value, 'groups':b_groups}
# split = get_split(dataset)

# %%
dataset = [[2.771244718,1.784783929,0],
	[1.728571309,1.169761413,0],
	[3.678319846,2.81281357,0],
	[3.961043357,2.61995032,0],
	[2.999208922,2.209014212,0],
	[7.497545867,3.162953546,1],
	[9.00220326,3.339047188,1],
	[7.444542326,0.476683375,1],
	[10.12493903,3.234550982,1],
	[6.642287351,3.319983761,1]]
# import matplotlib.pyplot as plt
# x1 = [dataset[r][0] for r in range(len(dataset))]
# x2  = [dataset[r][1] for r in range(len(dataset))]
# y = [dataset[r][2] for r in range(len(dataset))]
# color= ['red' if l == 0 else 'green' for l in y]
# plt.scatter(x1, x2, color=color)

# %%
# split = get_split(dataset)
# print('Split: [X%d < %.3f]' % ((split['index']+1), split['value']))

# %%
#Building a Tree
'''Maximum Tree Depth: This is the maximum number of nodes from the root node of the tree. Once a maximum depth of the tree is met, we must stop splitting adding new nodes. Deeper trees are more complex and are more likely to overfit the training data.'''
'''Minimum Node Records. This is the minimum number of training patterns that a given node is responsible for.'''

# Create a terminal node value
def to_terminal(group):
	outcomes = [row[-1] for row in group]
	return max(set(outcomes), key=outcomes.count)
''' It returns the most common output value in a list of rows'''
''' taking the group of rows assigned to that node and selecting the most common class value in the group. This will be used to make predictions.'''

# %%

def split(node, max_depth, min_size, depth):
	left, right = node['groups']
	del(node['groups'])
	# check for a no split
	if not left or not right:
		node['left'] = node['right'] = to_terminal(left + right)
		return
	# check for max depth
	if depth >= max_depth:
		node['left'], node['right'] = to_terminal(left), to_terminal(right)
		return
	# process left child
	if len(left) <= min_size:
		node['left'] = to_terminal(left)
	else:
		node['left'] = get_split(left)
		split(node['left'], max_depth, min_size, depth+1)
	# process right child
	if len(right) <= min_size:
		node['right'] = to_terminal(right)
	else:
		node['right'] = get_split(right)
		split(node['right'], max_depth, min_size, depth+1)


# %%
def build_tree(train, max_depth, min_size):
	root = get_split(train)
	split(root, max_depth, min_size, 1)
	return root

# %%
def print_tree(node, depth=0):
	if isinstance(node, dict):
		print('%s[X%d < %.3f]' % ((depth*' ', (node['index']+1), node['value'])))
		print_tree(node['left'], depth+1)
		print_tree(node['right'], depth+1)
	else:
		print('%s[%s]' % ((depth*' ', node)))

# %%
tree = build_tree(dataset, 3, 1)
print_tree(tree)

# %%
