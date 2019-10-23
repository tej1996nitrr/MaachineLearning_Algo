#%%
import csv
#opening file in text mode
with open('Data\iris.data', 'rt') as csvfile:
	lines = csv.reader(csvfile)
	for row in lines:
		print( ', '.join(row))
#%%


