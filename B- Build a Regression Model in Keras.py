#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


#The dataset is about the compressive strength of different samples of concrete based on the volumes of the different ingredients that were used to make them. Ingredients include:

#1. Cement

#2. Blast Furnace Slag

#3. Fly Ash

#4. Water

#5. Superplasticizer

#6. Coarse Aggregate

#7. Fine Aggregate


# In[3]:


concrete_data = pd.read_csv('C://Users//Concrete_Data.csv')
concrete_data.head()


# In[4]:


concrete_data.shape


# In[5]:


concrete_data.describe()


# In[6]:


concrete_data.isnull().sum()


# In[7]:


#Split data into predictors and target
#The target variable in this problem is the concrete sample strength. Therefore, our predictors will be all the other columns.

concrete_data_columns = concrete_data.columns
predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']] # all columns except Strength
target = concrete_data['CMS'] # Strength column


# In[8]:


predictors.head()


# In[9]:


target.head()


# In[10]:


n_cols = predictors.shape[1] # number of predictors
n_cols


# In[11]:


pip install keras


# In[12]:


pip install tensorflow


# In[13]:


import tensorflow as tf


# In[14]:


import keras
from keras.models import Sequential
from keras.layers import Dense


# In[15]:


# define regression model
def regression_model():
    # create model
    model = Sequential()
    model.add(Dense(10, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(1))
    
    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# In[16]:


from sklearn.model_selection import train_test_split


# In[17]:


X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size=0.3, random_state=42)


# In[18]:


# build the model
model = regression_model()


# In[19]:


# fit the model
epochs = 50
model.fit(X_train, y_train, epochs=epochs, verbose=1)


# In[20]:


loss_val = model.evaluate(X_test, y_test)
y_pred = model.predict(X_test)
loss_val


# In[21]:


from sklearn.metrics import mean_squared_error


# In[22]:


mean_square_error = mean_squared_error(y_test, y_pred)
mean = np.mean(mean_square_error)
standard_deviation = np.std(mean_square_error)
print(mean, standard_deviation)


# In[23]:


total_mean_squared_errors = 50
epochs = 50
mean_squared_errors = []
for i in range(0, total_mean_squared_errors):
    X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size=0.3, random_state=i)
    model.fit(X_train, y_train, epochs=epochs, verbose=0)
    MSE = model.evaluate(X_test, y_test, verbose=0)
    print("MSE "+str(i+1)+": "+str(MSE))
    y_pred = model.predict(X_test)
    mean_square_error = mean_squared_error(y_test, y_pred)
    mean_squared_errors.append(mean_square_error)

mean_squared_errors = np.array(mean_squared_errors)
mean = np.mean(mean_squared_errors)
standard_deviation = np.std(mean_squared_errors)

print('\n')
print("Below is the mean and standard deviation of " +str(total_mean_squared_errors) + " mean squared errors without normalized data. Total number of epochs for each training is: " +str(epochs) + "\n")
print("Mean: "+str(mean))
print("Standard Deviation: "+str(standard_deviation))


# In[ ]:





# In[ ]:




