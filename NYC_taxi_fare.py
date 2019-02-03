
# coding: utf-8

# In[2]:


I have worked with first 6 million data rows throughout the notebook for train the model. 
In future I will work with the entire dataset for a better prediction of the model.


# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
file_path="C:/Users/SHREYA/Desktop/Fall_SEM/DSF/all/train.csv"
df = pd.read_csv(file_path,nrows = 6000000)


# In[ ]:


The entire test data was loaded into the dataframe.


# In[2]:


file_path="C:/Users/SHREYA/Desktop/Fall_SEM/DSF/all/test.csv"
df_test = pd.read_csv(file_path)


# In[ ]:


Data Cleaning : After some exploration of the data, I proceeded with the below cleaning which for me was 
either noisy data or outliers.


# In[2]:


df = df.dropna(how='any',axis=0)


# In[3]:


df = df.drop(df[df['fare_amount']<0].index, axis=0)


# In[4]:


df = df.drop(df[(df.passenger_count >8) | (df.passenger_count<0)].index, axis=0)


# In[5]:


df=df[(df.pickup_latitude > 37) & (df.pickup_latitude < 43) & (df.dropoff_latitude > 37) & (df.dropoff_latitude < 42)]


# In[6]:


df=df[(df.pickup_longitude > -75) & (df.pickup_longitude < -72) & (df.dropoff_longitude > -75) & (df.dropoff_longitude < -72)]


# In[ ]:


Feature Engineering : My first feature was to find the distance of the trip and map it with the fare amount to
    see the relation.
Here i am finding both the euclidean and haversine distnce(since this will give me a distance in Kilometers)


# In[7]:


def euc_distance(pick_lat, pick_lng, drop_lat, drop_lng):
    x = (drop_lat-pick_lat)
    y = (drop_lng - pick_lng)
    return np.sqrt(x*x + y*y)


def hav_distance(pick_lat, pick_lng, drop_lat, drop_lng):
    r=6371
    phi1=np.radians(pick_lat)
    phi2=np.radians(drop_lat)
    delta_phi=np.radians(drop_lat-pick_lat)
    delta_lamda=np.radians(drop_lng-pick_lng)
    
    a=np.sin(delta_phi/2) * np.sin(delta_phi/2) + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lamda/2) * np.sin(delta_lamda/2)
    c= 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d= r*c
    return d


# In[8]:


dist=euc_distance(df['pickup_latitude'], df['pickup_longitude'], df['dropoff_latitude'], df['dropoff_longitude'])
hav_dist=hav_distance(df['pickup_latitude'], df['pickup_longitude'], df['dropoff_latitude'], df['dropoff_longitude'])


# In[9]:


df['euclidean_distance']=dist


# In[10]:


df['haversine_distance']=hav_dist


# In[12]:


dist=euc_distance(df_test['pickup_latitude'], df_test['pickup_longitude'], df_test['dropoff_latitude'], df_test['dropoff_longitude'])
hav_dist=hav_distance(df_test['pickup_latitude'], df_test['pickup_longitude'], df_test['dropoff_latitude'], df_test['dropoff_longitude'])


# In[13]:


df_test['euclidean_distance']=dist


# In[14]:


df_test['haversine_distance']=hav_dist


# In[15]:


data = df[['euclidean_distance','fare_amount']]
correlation = data.corr(method='pearson')


# In[16]:


correlation


# In[17]:


df['euclidean_distance'].corr(df['fare_amount'])


# In[18]:


df['haversine_distance'].corr(df['fare_amount'])


# In[19]:


plot=df.plot.scatter('euclidean_distance','fare_amount')


# In[ ]:


There is a field given as 'pickup_datetime' which can be used to extract vital information pertaining
to time such as hour of travel,date,day of week,month.So we consider this also as an feature for our model


# In[11]:


import math
import re
get_ipython().run_line_magic('timeit', '')
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'],infer_datetime_format=True)


# In[12]:


df['date'] = df['pickup_datetime'].dt.day
df['day of week'] = df['pickup_datetime'].dt.dayofweek
df['year'] = df['pickup_datetime'].dt.year
df['month'] = df['pickup_datetime'].dt.month
df['hour'] = df['pickup_datetime'].dt.hour


# In[22]:


df_test['pickup_datetime'] = pd.to_datetime(df_test['pickup_datetime'],infer_datetime_format=True)


# In[23]:


df_test['date'] = df_test['pickup_datetime'].dt.day
df_test['day of week'] = df_test['pickup_datetime'].dt.dayofweek
df_test['year'] = df_test['pickup_datetime'].dt.year
df_test['month'] = df_test['pickup_datetime'].dt.month
df_test['hour'] = df_test['pickup_datetime'].dt.hour


# In[ ]:


Adding another feature 'time' which is basically the hour of travel, converting minutes and seconds to hours.


# In[13]:


df['time'] = df['pickup_datetime'].dt.hour + df['pickup_datetime'].dt.minute / 60 + df['pickup_datetime'].dt.second / 3600


# In[25]:


df_test['time'] = df_test['pickup_datetime'].dt.hour + df_test['pickup_datetime'].dt.minute / 60 + df_test['pickup_datetime'].dt.second / 3600


# In[26]:


df['passenger_count'].corr(df['fare_amount'])


# In[28]:


df['date'].corr(df['fare_amount'])


# In[29]:


df['day of week'].corr(df['fare_amount'])


# In[30]:


df['hour'].corr(df['fare_amount'])


# In[31]:


df['time'].corr(df['fare_amount'])


# In[32]:


df['time'].corr(df['euclidean_distance'])


# In[33]:


plot=df.plot.scatter('time','fare_amount')


# In[34]:


plot=df.plot.scatter('time','euclidean_distance')


# In[35]:


def manhattan_distance(pick_lat, pick_lng, drop_lat, drop_lng):
    x = np.abs(drop_lat-pick_lat)
    y = np.abs(drop_lng - pick_lng)
    return (x + y)


# In[36]:


df['manhattan_distance']=manhattan_distance(df['pickup_latitude'], df['pickup_longitude'], df['dropoff_latitude'], df['dropoff_longitude'])


# In[37]:


plot=df.plot.scatter('manhattan_distance','fare_amount')


# In[38]:


df['manhattan_distance'].corr(df['fare_amount'])


# In[ ]:


loc=df[['fare_amount','jfk_dropoff','jfk_pickup','lga_pickup','lga_dropoff']].copy()
sns.pairplot(loc)


# In[39]:


df_test['manhattan_distance']=manhattan_distance(df_test['pickup_latitude'], df_test['pickup_longitude'], df_test['dropoff_latitude'], df_test['dropoff_longitude'])


# In[40]:


df = df.drop(df[(df['haversine_distance']==0)&(df['fare_amount']==0)].index, axis = 0)


# In[41]:


df.loc[(df['haversine_distance']!=0) & (df['fare_amount']==0)].count()


# In[42]:


df.loc[(df['haversine_distance']!=0) & (df['fare_amount']==0)]


# In[44]:


impute1=df.loc[(df['haversine_distance']!=0) & (df['fare_amount']==0)]


# In[45]:


impute1['fare_amount'] = impute1.apply(lambda row: ((row['haversine_distance'] * 1.5534) + 2.50), axis=1)


# In[46]:


df.update(impute1)


# In[47]:


impute2=df.loc[(df['haversine_distance']==0) & (df['fare_amount']>3)]


# In[48]:


impute2['haversine_distance'] = impute2.apply(lambda row: ((row['fare_amount'] -2.5)/1.5534), axis=1)


# In[49]:


df.update(impute2)


# In[50]:


plot=df.plot.scatter('haversine_distance','fare_amount')


# In[51]:


df=df.drop(['key'],axis=1)


# In[52]:


df=df.drop(['pickup_datetime'],axis=1)


# In[53]:


df_test=df_test.drop(['key'],axis=1)


# In[54]:


df_test=df_test.drop(['pickup_datetime'],axis=1)


# In[57]:


x_df = df.iloc[:,df.columns!='fare_amount']
y_df = df['fare_amount'].values


# In[58]:


from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
lm=LinearRegression()
lm.fit(x_df, y_df)
lm.predict=lm.predict(df_test)


# In[59]:


sample = pd.read_csv("C:/Users/SHREYA/Desktop/Fall_SEM/DSF/all/sample_submission.csv")
sample['fare_amount']= lm.predict
sample.to_csv('C:/Users/SHREYA/Desktop/Fall_SEM/DSF/all/submission_10.csv', index=False)


# In[ ]:


Next we calculate the airport trips, or trips made in the vicinity of airport as a new feature for the model.


# In[11]:


jfk = (-73.7781, 40.6413)
ewr = (-74.175,40.69)
lga = (-73.87, 40.77)

def nyc_pl_dist(coord):
    df['jfk_dropoff'] = (hav_distance(df['dropoff_latitude'], df['dropoff_longitude'], coord[1], coord[0]))
    df['jfk_pickup'] = (hav_distance(coord[1], coord[0],df['pickup_latitude'], df['pickup_longitude']))
    df['ewr_dropoff'] = (hav_distance(df['dropoff_latitude'], df['dropoff_longitude'], coord[1], coord[0]))
    df['ewr_pickup'] = (hav_distance(coord[1], coord[0],df['pickup_latitude'], df['pickup_longitude']))
    df['lga_dropoff'] = (hav_distance(df['dropoff_latitude'], df['dropoff_longitude'], coord[1], coord[0]))
    df['lga_pickup'] = (hav_distance(coord[1], coord[0],df['pickup_latitude'], df['pickup_longitude']))
    #idx.plot('haversine_distance','fare_amount')
    

nyc_pl_dist(jfk)
nyc_pl_dist(ewr)
nyc_pl_dist(lga)


# In[61]:


jfk = (-73.7781, 40.6413)
ewr = (-74.175,40.69)
lga = (-73.87, 40.77)

def nyc_pl_dist(coord):
    df_test['jfk_dropoff'] = (hav_distance(df_test['dropoff_latitude'], df_test['dropoff_longitude'], coord[1], coord[0]))
    df_test['jfk_pickup'] = (hav_distance(coord[1], coord[0],df_test['pickup_latitude'], df_test['pickup_longitude']))
    df_test['ewr_dropoff'] = (hav_distance(df_test['dropoff_latitude'], df_test['dropoff_longitude'], coord[1], coord[0]))
    df_test['ewr_pickup'] = (hav_distance(coord[1], coord[0],df_test['pickup_latitude'], df_test['pickup_longitude']))
    df_test['lga_dropoff'] = (hav_distance(df_test['dropoff_latitude'], df_test['dropoff_longitude'], coord[1], coord[0]))
    df_test['lga_pickup'] = (hav_distance(coord[1], coord[0],df_test['pickup_latitude'], df_test['pickup_longitude']))
    #idx.plot('haversine_distance','fare_amount')
    

nyc_pl_dist(jfk)
nyc_pl_dist(ewr)
nyc_pl_dist(lga)


# In[ ]:


df[['jfk_pickup']] = df[['jfk_pickup']].apply(lambda value:np.add(df['jfk_pickup'], df['jfk_dropoff']))


# In[62]:


plot=df.plot.scatter('jfk_pickup','fare_amount')


# In[65]:


x_df = df.iloc[:,df.columns!='fare_amount']
y_df = df['fare_amount'].values


# In[67]:


from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
lm=LinearRegression()
lm.fit(x_df, y_df)
lm.predict=lm.predict(df_test)


# In[68]:


sample = pd.read_csv("C:/Users/SHREYA/Desktop/Fall_SEM/DSF/all/sample_submission.csv")
sample['fare_amount']= lm.predict
sample.to_csv('C:/Users/SHREYA/Desktop/Fall_SEM/DSF/all/submission_20.csv', index=False)


# In[ ]:


from sklearn.feature_selection import SelectKBest
feature_importances = pd.DataFrame({'feature': x_df,
                                        'importance': lm.feature_importances_}).\
                           sort_values('importance', ascending = False).set_index('feature')


# In[69]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(x_df, y_df)
rf_predict = rf.predict(df_test)


# In[71]:


sample = pd.read_csv("C:/Users/SHREYA/Desktop/Fall_SEM/DSF/all/sample_submission.csv")
sample['fare_amount']= rf_predict
sample.to_csv('C:/Users/SHREYA/Desktop/Fall_SEM/DSF/all/submission_30.csv', index=False)


# In[72]:


from sklearn.feature_selection import SelectKBest
feature_importances = pd.DataFrame({'feature': x_df,
                                        'importance': rf.feature_importances_}).\
                           sort_values('importance', ascending = False).set_index('feature')


# In[73]:


feature_importances


# In[74]:


weather=pd.read_csv("C:/Users/SHREYA/Desktop/Fall_SEM/DSF/all/weather_train.csv",nrows=6000000)


# In[77]:


weather.head(2)


# In[82]:


weather['DATE'] = pd.to_datetime(weather['DATE'],infer_datetime_format=True)


# In[83]:


weather['year'] = weather['DATE'].dt.year
weather['month'] = weather['DATE'].dt.month
weather['day of week'] = weather['DATE'].dt.day


# In[86]:


df = pd.merge(df, weather, how='left', on=['year','month'])


# In[87]:


test


# In[ ]:


test = df.dropna(how='any',axis=0)


# In[ ]:


test.head(1)


# In[ ]:


plot=df.plot.scatter('SNOW','fare_amount')


# In[ ]:


df['SNOW'].corr(df['fare_amount'])

