#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib as mtb


# In[2]:


os.chdir(r"C:/Users/user/Documents")
os.getcwd()


# In[179]:


df=pd.read_csv(r"C:/Users/user/Downloads/train.csv")


# In[181]:


df1=pd.read_csv(r"C:/Users/user/Downloads/test.csv")



# In[5]:


df.columns


# In[6]:


df1.columns


# In[7]:


type(df)
type(df1)


# In[8]:


df.shape


# In[9]:


df1.shape


# In[10]:


df.describe()


# In[11]:


df1.describe()


# In[182]:


df=df.set_index('ID_code')


# In[183]:


df1=df1.set_index('ID_code')


# In[16]:


miss_val=pd.DataFrame(df.isnull().sum())
miss_val


# In[17]:


df['var_0'].loc[70]


# In[18]:


df['var_0'].loc[70]=np.nan
df['var_0'].loc[70]


# In[20]:


df['var_0']=df['var_0'].fillna(df['var_0'].mean())
df['var_0'].loc[70]


# In[21]:


df['var_0']=df['var_0'].fillna(df['var_0'].median())
df['var_0'].loc[70]


# In[22]:


df['var_0'].loc[70]=11.0572


# In[26]:


df_corr=df.iloc[:,:]
df_corr


# In[28]:


import seaborn as sns
corr = df.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);
ax


# In[125]:


df['target']=df['target'].astype('category')
import statsmodels.api as sm
logit=sm.Logit(df['target'],df[df.columns[1:100]])
logit=logit.fit()
logit.summary()


# In[199]:


df1.insert(0,'target',1)
for i in range (0,df1.shape[1]):
    df1.iloc[i,0]=df.iloc[i,0]
df2=df2.iloc[0:,0:]

# In[192]:



df2['Actual_prob']=logit.predict(df2[df.columns[1:100]])
df2['Actual_val']=1
df2.loc[df2.Actual_prob <0.05,'Actual_val']=0


# In[196]:




CM=pd.crosstab(df2['target'],df2['Actual_val'])


# In[173]:


CM


# In[162]:


TN=CM.iloc[0,0]
FN=CM.iloc[1,0]
TP=CM.iloc[1,1]
FP=CM.iloc[0,1]


# In[ ]:


Accuracy=((TP+TN)*100)/(TP+TN+FN+FP)
FNR=(FN*100)/(FN+TP)
Recall=(TP*100)/(FN+TP)
Accuracy


# In[ ]:


FNR


# In[ ]:


Recall


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
x1=df.values[0:,1:]
y1=df.values[0:,0]
from sklearn.model_selection import train_test_split
X1_train, X1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.20)
KNN_Model=KNeighborsClassifier(n_neighbors=1)
KNN_Model.fit(X1_train,y1_train)
KNN_Model


# In[ ]:


KNN_pred=KNN_Model.predict(X1_test)
KNN_pred


# In[ ]:


CM=pd.crosstab(y1_test,KNN_pred)
CM


# In[ ]:


TN=CM.iloc[0,0]
FN=CM.iloc[1,0]
TP=CM.iloc[1,1]
FP=CM.iloc[0,1]
Accuracy=((TP+TN)*100)/(TP+TN+FN+FP)
FNR=(FN*100)/(FN+TP)
Recall=(TP*100)/(FN+TP)
Accuracy


# In[ ]:


FNR


# In[ ]:


Recall


# In[ ]:


from sklearn.naive_bayes import GaussianNB
NB_model=GaussianNB().fit(X1_train,y1_train)
NB_pred=NB_model.predict(X1_test)
CM=pd.crosstab(y1_test,NB_pred)
CM


# In[ ]:


TN=CM.iloc[0,0]
FN=CM.iloc[1,0]
TP=CM.iloc[1,1]
FP=CM.iloc[0,1]
Accuracy=((TP+TN)*100)/(TP+TN+FN+FP)
FNR=(FN*100)/(FN+TP)
Recall=(TP*100)/(FN+TP)
Accuracy


# In[ ]:


FNR


# In[ ]:


Recall


# In[ ]:

df1['target']=NB_pred
