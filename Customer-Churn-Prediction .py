#!/usr/bin/env python
# coding: utf-8

# In[33]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn import metrics
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score


# In[34]:


data=pd.read_csv('Telco-Customer-Churn.csv')


# In[35]:


data.head()


# In[36]:


data.tail()


# In[37]:


data.isnull().sum()


# In[38]:


data.dtypes


# In[39]:


data['TotalCharges']=data["TotalCharges"].replace(" ",np.nan)


# In[40]:


data = data[data["TotalCharges"].notnull()]
data = data.reset_index()[data.columns]


# In[41]:


data["TotalCharges"] = data["TotalCharges"].astype(float)


# In[42]:


data['Churn'].value_counts()


# In[43]:


cont=[]
disc=[]

for col in data.columns:
    if data[col].nunique()>5:
        cont.append(col)
        
    else:
        disc.append(col)
        
print ('Continuous variables are ',cont)
print('-----------------------------------')
print ('Discrete variables are ' ,disc)


# In[44]:


for col in disc:
  print(data[col].value_counts())
  print('------------------------------------------')


# In[45]:


data.describe()


# In[46]:


data.nunique()


# In[47]:


sns.set(style="whitegrid", color_codes=True)

fig, axes = plt.subplots(nrows = 8,ncols = 2,figsize = (30,40))
sns.countplot(x = "gender", data = data, hue=data.Churn, ax=axes[0][0])
sns.countplot(x="SeniorCitizen",data=data,hue=data.Churn, ax=axes[0][1])
sns.countplot(x = "Partner", data = data, hue=data.Churn, ax=axes[1][0])
sns.countplot(x = "Dependents", data = data, hue=data.Churn, ax=axes[1][1])
sns.countplot(x = "PhoneService", data = data, hue=data.Churn, ax=axes[2][0])
sns.countplot(x = "MultipleLines", data = data, hue=data.Churn, ax=axes[2][1])
sns.countplot(x = "InternetService", data = data, hue=data.Churn, ax=axes[3][0])
sns.countplot(x = "OnlineSecurity", data = data, hue=data.Churn, ax=axes[3][1])
sns.countplot(x = "OnlineBackup", data = data, hue=data.Churn, ax=axes[4][0])
sns.countplot(x = "DeviceProtection", data = data, hue=data.Churn, ax=axes[4][1])
sns.countplot(x = "TechSupport", data = data, hue=data.Churn, ax=axes[5][0])
sns.countplot(x = "StreamingTV", data = data, hue=data.Churn, ax=axes[5][1])
sns.countplot(x = "StreamingMovies", data = data, hue=data.Churn, ax=axes[6][0])
sns.countplot(x = "Contract", data = data, hue=data.Churn, ax=axes[6][1])
sns.countplot(x = "PaperlessBilling", data = data, hue=data.Churn, ax=axes[7][0])
ax = sns.countplot(x = "PaymentMethod", data = data, hue=data.Churn, ax=axes[7][1])
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)

plt.show(fig)


# In[48]:


data.hist(cont, bins=50, figsize=(20,15))
plt.show()


# In[49]:


data.drop(['customerID'], axis=1, inplace=True)


# In[50]:


y=data.Churn


# In[51]:


data.drop(['Churn'], axis=1, inplace=True)


# In[52]:


data=pd.get_dummies(data)


# In[53]:


data.head()


# In[54]:


X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.20,random_state=83)
print(X_train.shape,y_train.shape)
print(X_test.shape,y_test.shape)


# In[55]:


X_train.head()


# In[56]:


y_train.head()


# In[57]:


logmodel=LogisticRegression(max_iter=200,C=1,solver='liblinear')
logmodel.fit(X_train,y_train)


# In[58]:


pred1 = logmodel.predict(X_test)
print('Accuracy of Logistic Regression on test set:',accuracy_score(y_test, pred1))


# In[59]:


results1=confusion_matrix(y_test, pred1)
print(results1)

