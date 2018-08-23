
# coding: utf-8

# In[4]:


import sys
import pandas
import matplotlib
import seaborn
import sklearn


# In[5]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# In[6]:


# Load the data

games = pandas.read_csv("datasets/games.csv")


# In[7]:


print(games.columns)


# In[8]:


print(games.shape)


# In[9]:


# we will make a histogram of all the ratings in the average_rating column

plt.hist(games["average_rating"])
plt.show()


# In[10]:


# Print the first row of all the games with zero scores
print(games[games["average_rating"]==0].iloc[0])

# Print the first row of games with scores greater than 0
print(games[games["average_rating"]>0].iloc[0])


# In[11]:


# Remove any rows without user review

games = games[games["users_rated"]>0]

# Remove any rows with missing values
games = games.dropna(axis=0)

#Make a histogram of all the average ratings

plt.hist(games["average_rating"])
plt.show()


# In[12]:


print(games.columns)


# In[18]:


# develop correlation matrix

corrmat = games.corr()
fig = plt.figure(figsize = (12, 8))

sns.heatmap(corrmat, vmax = 0.8 , square = True)
plt.show()

# According to correlation matrix average_rating is the prominent factor


# In[19]:


# Get all the columns from the dataframe

columns = games.columns.tolist()

#Filter the columns to remove data we do not want

columns = [c for c in columns if c not in ["bayes_average_rating","average_rating","type","name","id"]]

#store the variable we will be predicting on

target = "average_rating"


# In[20]:


# Generate training and testing datasets

from sklearn.cross_validation import train_test_split

# Generate training set

train = games.sample(frac=0.8,random_state = 1)

# Generate test datasets data not present in training will come to testing

test = games.loc[~games.index.isin(train.index)]

#Print shapes
print(train.shape)
print(test.shape)


# In[21]:


# Import linear regression model

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Initialize the model data

LR = LinearRegression()

#Fit the model the training data
LR.fit(train[columns],train[target])


# In[22]:


# Generate prediction for the test dataset

predictions = LR.predict(test[columns])

# Compute error between our test predictions and actual values
mean_squared_error(predictions,test[target])


# In[23]:


# Import the rndom forest model

from sklearn.ensemble import RandomForestRegressor

# Initialize the model

RFR = RandomForestRegressor(n_estimators = 100, min_samples_leaf=10,random_state = 1)

# Fit to the data

RFR.fit(train[columns],train[target])


# In[24]:


# make predictions

predictions = RFR.predict(test[columns])

# compute the error between our test predictions and actual values
mean_squared_error(predictions,test[target])


# In[25]:


test[columns].iloc[0]


# In[27]:


# Make precdictions with both models
rating_LR = LR.predict(test[columns].iloc[0].values.reshape(1,-1))
rating_RFR = RFR.predict(test[columns].iloc[0].values.reshape(1,-1))

# Print out the predictions
print(rating_LR)
print(rating_RFR)


# In[28]:


test[target].iloc[0]

