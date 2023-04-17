#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

weather = pd.read_csv("3121464.csv", index_col="DATE")


# In[2]:


weather


# In[3]:


weather.apply(pd.isnull).sum()/weather.shape[0]


# In[4]:


core_weather = weather[["PRCP", "SNOW", "SNWD", "TMAX", "TMIN"]].copy()
core_weather.columns = ["precip", "snow", "snow_depth", "temp_max", "temp_min"]
core_weather


# In[5]:


core_weather.apply(pd.isnull).sum()/core_weather.shape[0]


# In[6]:


core_weather["snow"].value_counts()


# In[7]:


core_weather["snow_depth"].value_counts()


# In[8]:


del core_weather["snow"]
del core_weather["snow_depth"]
core_weather[pd.isnull(core_weather["precip"])]


# In[9]:


core_weather.loc["1983-10-29":"2017-10-28",:]


# In[10]:


core_weather["precip"].value_counts() 


# In[11]:


core_weather["precip"] = core_weather["precip"].fillna(0)
core_weather.apply(pd.isnull).sum()


# In[12]:


core_weather[pd.isnull(core_weather["temp_min"])]


# In[13]:


core_weather.loc["2011-12-18":"2011-12-28"]


# In[14]:


core_weather = core_weather.fillna(method="ffill")
core_weather.apply(pd.isnull).sum()


# In[15]:


# Check for missing value defined in data documentation
core_weather.apply(lambda x: (x == 9999).sum())


# In[16]:


core_weather.dtypes


# In[17]:


core_weather.index


# In[18]:


core_weather.index = pd.to_datetime(core_weather.index)
core_weather.index


# In[19]:


core_weather[["temp_max","temp_min"]].plot()


# In[20]:


core_weather.index.year.value_counts().sort_index()


# In[21]:


core_weather.groupby(core_weather.index.year).sum()


# In[22]:


#predict tomorrows temperature using previous day data
core_weather["target"] = core_weather.shift(-1)["temp_max"]
core_weather


# In[23]:


core_weather = core_weather.iloc[:-1,:].copy()


# In[24]:


from sklearn.linear_model import Ridge

reg = Ridge(alpha=.1)


# In[25]:


#select colums for prediction
predictors = ["precip", "temp_max", "temp_min"]

#seperating train and test data
train = core_weather.loc["2000-01-01":"2020-12-31"]
test = core_weather.loc["2021-01-01":]


# In[26]:


reg.fit(train[predictors], train["target"])


# In[27]:


predictions = reg.predict(test[predictors])

from sklearn.metrics import mean_absolute_error

mean_absolute_error(test["target"], predictions)


# In[28]:


combined = pd.concat([test["target"], pd.Series(predictions, index=test.index)], axis=1)
combined.columns = ["actual", "predictions"]


# In[29]:


combined


# In[30]:


reg.coef_


# In[31]:


def create_predictions(predictors, core_weather, reg):
    train = core_weather.loc["2000-01-01":"2020-12-31"]
    test = core_weather.loc["2021-01-01":]

    reg.fit(train[predictors], train["target"])
    predictions = reg.predict(test[predictors])

    error = mean_absolute_error(test["target"], predictions)
    
    combined = pd.concat([test["target"], pd.Series(predictions, index=test.index)], axis=1)
    combined.columns = ["actual", "predictions"]
    return error, combined


# In[33]:


combined.plot()


# In[ ]:




