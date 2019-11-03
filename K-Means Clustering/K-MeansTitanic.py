#%%
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
# %%
import pandas as pd 
df = pd.read_excel(r'Data\titanic.xlsx')
df.drop(['body','name'],1,inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0,inplace=True)



# %%

def handle_non_numericalData(df):
    columns = df.columns.values
    for column in columns:
        text_digit_vals={}
        def convert_toInt(val):
            return text_digit_vals[val]
        
        if df[column].dtype!=np.float and df[column].dtype!=np.int:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x=0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique]=x
                    x+=1
            df[column] = list(map(convert_toInt,df[column]))

    return df

df = handle_non_numericalData(df)    
#%%
x = np.array(df.drop(['survived'],1).astype(float))
y = np.array(df['survived'])
x=preprocessing.scale(x)
clf = KMeans(n_clusters=2)
# %%
clf.fit(x)
correct = 0
for i in range(len(x)):
    predict_me = np.array(x[i].astype(float))
    predict_me = predict_me.reshape(-1,len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0]==y[i]:
        correct+=1
print(correct/len(x))

