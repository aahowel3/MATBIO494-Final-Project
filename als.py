#!/usr/bin/python 
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix

def strip(text):
    try:
        return text.strip()
    except AttributeError:
        return text


df=pd.read_csv("als_data2.csv", sep=',', header=None, encoding="utf-8", 
names=['gene', 'genetype', 'bp', 'country', 'age', 'symptoms', 'UMN', 'cognitive'],
converters = {'genetype' : strip,'country':strip, 'symptoms':strip}) 

def replace_most_common(x):
    if x=="NaN" or x=="N/A" or x=="":
        return most_common
    else:
        return x
      
def mut(line):
    try:
        x,y=line.split(">")
        q=str(x)
        a=str(y)
        x1=q[-1]
        y1=a[0]
        strr= x1 + y1
        return(strr) 
    except ValueError:
        return "indel" 

bp2=[]
for x in df['bp']:
    r=str(x)
    rt=mut(r) 
    bp2.append(rt) 

df['bp2'] = bp2
df1=pd.get_dummies(df['bp2'], prefix="mutationtype")

#hot-encoding for symptoms column 
regex_pattern = r',\s*(?![^()]*\))'

# we want to replace *those* commas with empty string (aka delete them)
replacement = '//'

flip_B7=[] 

for s in df['symptoms']:
    r=re.sub(regex_pattern, replacement, s)
    flip_B7.append(r)


df['symptoms2'] = flip_B7
df2=df['symptoms2']
most_common = df2.str.get_dummies(sep='//').sum().sort_values(ascending=False).index[0]
new_m = df2.map(replace_most_common)
df2_final = new_m.str.get_dummies(sep='//')
df2_final= df2_final.add_prefix('symptoms_')


def fill(line):
    try:
        x=re.sub(regex_pattern, replacement, line) 
        return(x)
    except AttributeError:
        return "NaN"
        
countrynew=[]
for s in df['country']:
    r=fill(s) 
    countrynew.append(r)



df['country2'] = countrynew
df4=df['country2']


most_common = df4.str.get_dummies(sep='//').sum().sort_values(ascending=False).index[0]
new_m = df4.map(replace_most_common)
df4_final = new_m.str.get_dummies(sep='//')
df4_final = df4_final.add_prefix('country_')

df5=df['UMN']
most_common=df5.str.get_dummies().sum().sort_values(ascending=False).index[0]
new_m = df5.map(replace_most_common)
df5_final = new_m.str.get_dummies()
df5_final = df5_final.add_prefix('UMN_')


flip_df6=[] 

for s in df['cognitive']:
    s2=str(s)
    r=re.sub(regex_pattern, replacement, s2)
    flip_df6.append(r)

df['cognitive2']=flip_df6
df6=df['cognitive2']
most_common=df6.str.get_dummies(sep=',').sum().sort_values(ascending=False).index[0]
new_m = df6.map(replace_most_common)
df6_final = new_m.str.get_dummies(sep=',')
df6_final = df6_final.add_prefix('cognitiveimpairment_')

df12=df['genetype']
df12_final = df12.str.get_dummies()
df12_final = df12_final.add_prefix('genetype_')

df13=df['gene']
df13_final = df13.str.get_dummies()
df13_final = df13_final.add_prefix('gene_')

flip_B9=[]
df['age'] = df['age'].fillna('NaN')

for s in df['age']:    
    aa=str(s)
    r=re.sub('-',',',aa)
    q=[x.strip() for x in r.split(',')]
    q= [float(i) for i in q]
    qq=((sum(q))/(len(r)))
    flip_B9.append(qq)

    

df['age'] = flip_B9
df3=df['age']
df3 = df3.fillna(df3.mean())



df_target=df['gene']
df_predictors=pd.concat([df2_final,df4_final,df3,df5_final,df6_final], axis=1)

rfc = RandomForestClassifier(n_estimators=100,oob_score=True)
df_target.value_counts()[:24]
top_10_cell_types = list(df_target.value_counts(normalize=True).index)[:24]
top_10_cell_type_indices = df_target.isin(top_10_cell_types)
df_target = df_target[top_10_cell_type_indices]
#with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
#    print(df_target)


df_predictors = df_predictors[top_10_cell_type_indices]


rfc.fit(df_predictors,df_target)
# Assess the in-sample performance
print("The predicted cross-validation classification performance is %.3g%%" % (rfc.oob_score_*100)) 
scores = cross_val_score(rfc, df_predictors, df_target)
for i,score in enumerate(scores):
    print("On cross-validation split #%d, the classification performance was %.3g%%" % (i+1,score*100))  
   
def plot_confusion_matrix(predictions,actual):
    # Split the data into a training set and a test set
    predictions = cross_val_predict(rfc, df_predictors, df_target) # Make out-of-sample predictions
    classes = list(set(df_target)) # A list of all the possible classes (the neuron names)
    cm = confusion_matrix(df_target, predictions, labels=classes) # Generate the confusion matrix
    cm = 100*cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # Normalize and convert to percentage
    plt.pcolor(cm,cmap='Reds') # Plot using the red colormap
    plt.colorbar().set_label('Percent classified as this') # Add the color scale
    ticks = np.arange(0.5,len(classes)) # The numeric valies of the axis tick positions
    plt.xticks(ticks,classes,rotation=90) # Add the x-axis ticks
    plt.yticks(ticks,classes); # Add the y-axis ticks  
    plt.tight_layout()
    plt.show()
    
predictions = cross_val_predict(rfc, df_predictors, df_target) # Make out-of-sample predictions
plot_confusion_matrix(predictions,df_target)

df_importances = pd.Series(rfc.feature_importances_,index=df_predictors.columns).sort_values(ascending=False)
df_importances.iloc[:20].plot.bar(title='Feature importances',figsize=(12,4));
plt.tight_layout()
plt.show()
            
              
 




