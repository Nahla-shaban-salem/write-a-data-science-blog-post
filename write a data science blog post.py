#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import necessarily libraries and create data frame 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv('listings.csv')
pd.options.display.max_rows = 4000


# # First, try to understand data from shape , data inside it and nulls values

# In[2]:


#know size of dataframe
df.shape


# In[3]:


#check data inside DF
df.head()


# In[4]:


#Check Data Type and Nulls
df.info()


# In[5]:


#Check count of null  to detrmine way to fill it
df.isnull().sum().sort_values(ascending=False)


# # after checking there are four questions to my mind.
# 
# 1- what's the relationship between scores cleanliness and scores rating?
# 
# 2-what's the highest property type in Seattle Airbnb?
# 
# 3- what's the distribution of amenities? what's on the top?
# 
# 4-Can you predict the price from available futures? 
# 

# #after checking data select main futures from my prespective

# In[6]:


#select main futures in list
col_needed=['id','property_type','room_type','accommodates','bathrooms','bedrooms','beds','bed_type','amenities','price'
            ,'security_deposit','cleaning_fee','review_scores_rating','review_scores_accuracy', 'review_scores_cleanliness',
             'review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value']


# In[17]:


#create new DataFrame by main futures
df2=df[col_needed]
df2.head()


# In[8]:


#doble check data type and nulls to determine most proper way to fix it
df2.info()


# # second prepared data to use it
# 
# 1- for object Kpis use mode to fill nun.
# 
# 2- for object Kpis with special character remove it and convert to float.
# 
# 3- for numeric kpis use mean or zero to fill nun.
# 
# 4- for object Kpis with more than values in filed sprated values in difftent columns with values 0,1 if user have value in orginal data
# 
# 5- for object Kpis use get_dummies to make dummies values .
# 

# In[18]:


df2['property_type'].fillna(df2['property_type'].mode()[0],inplace=True )
df2['bathrooms'].fillna(df2['bathrooms'].mean(),inplace=True)
df2['bedrooms'].fillna(df2['bedrooms'].mean(),inplace=True)
df2['beds'].fillna(df2['beds'].mean(),inplace=True)

df2 = pd.concat([df2.drop(columns = ['price']), df2['price'].str.replace('$','',regex=True).str.replace(',','',regex=True).astype(float)], axis = 1)
df2 = pd.concat([df2.drop(columns = ['security_deposit']), df2['security_deposit'].str.replace('$','',regex=True).str.replace(',','',regex=True).astype(float)], axis = 1)
df2['security_deposit'].fillna(0,inplace=True)
df2 = pd.concat([df2.drop(columns = ['cleaning_fee']), df2['cleaning_fee'].str.replace('$','',regex=True).str.replace(',','',regex=True).astype(float)], axis = 1)
df2['cleaning_fee'].fillna(0,inplace=True)

df2['review_scores_rating'].fillna(df2['review_scores_rating'].mean(),inplace=True)
df2['review_scores_accuracy'].fillna(df2['review_scores_accuracy'].mean(),inplace=True)
df2['review_scores_cleanliness'].fillna(df2['review_scores_cleanliness'].mean(),inplace=True)
df2['review_scores_checkin'].fillna(df2['review_scores_checkin'].mean(),inplace=True)
df2['review_scores_communication'].fillna(df2['review_scores_communication'].mean(),inplace=True)
df2['review_scores_location'].fillna(df2['review_scores_location'].mean(),inplace=True)
df2['review_scores_value'].fillna(df2['review_scores_value'].mean(),inplace=True)

amenities = df2['amenities']
amenities_list = []

for index, row in amenities.items():
    amenities_list.append(row.replace('{','').replace('}','').replace('"','').split(','))
    
amenities_df = pd.Series(amenities_list, name = 'amenities').to_frame()
dummies_amenities_df = amenities_df.drop('amenities', 1).join(
    pd.get_dummies(
        pd.DataFrame(amenities_df.amenities.tolist()).stack()
    ).astype(int).sum(level=0)
)

df2.drop('amenities','columns',inplace=True)
ML_DF = pd.concat([df2,dummies_amenities_df],axis=1)
ML_DF


# In[21]:


def create_dummy_df(df, cat_cols, dummy_na):
  
    for col in cat_cols:
        try:
            
            
            df=pd.concat([df.drop(col,axis=1),pd.get_dummies(df[col],prefix=col,prefix_sep='_',drop_first=True,dummy_na=dummy_na)],axis=1)

        except:
            continue
    return df


# In[22]:


df_name=df2.select_dtypes(include=['object'])
ML_DF=create_dummy_df(ML_DF,df_name,dummy_na=False)
ML_DF.head()


# In[15]:


#check new size
ML_DF.shape


# In[16]:


#double checking ,there's not null values
ML_DF.isnull().sum().sort_values(ascending=False)


# #start to answer the selected question
# 
# # 1-what's the relationship between scores cleanliness and scores rating?

# In[11]:


#Q 1 there's +ve relationship beteween review_scores_cleanliness and review_scores_rating

sns.scatterplot(data=ML_DF, x="review_scores_cleanliness", y="review_scores_rating");


# # 2-what's the highest property type in Seattle Airbnb?

# In[24]:


#Q2 House is the most property type in Seattle
df2.groupby('property_type')['id'].count().sort_values(ascending=False).plot(kind='bar',color='purple');


# # 3-what's the distribution of amenities? what's on the top?

# In[26]:


#Q3 as shown bellow the internet in the top of amenities
dummies_amenities_df.sum().sort_values(ascending=False).plot(kind='bar',figsize =(9,5),color='crimson');
plt.title('the distribution of amenities');
plt.ylabel('count')
plt.xlabel('amenities')
plt.show()


# # 4-Can you predict the price from available futures?

# In[27]:


# after use the selected futures the accuracy of preduction is 0.59
X = ML_DF[['accommodates', 'bathrooms', 'bedrooms', 'beds',
       'review_scores_rating', 'review_scores_accuracy',
       'review_scores_cleanliness', 'review_scores_checkin',
       'review_scores_communication', 'review_scores_location',
       'review_scores_value', 'security_deposit', 'cleaning_fee', '',
       '24-Hour Check-in', 'Air Conditioning', 'Breakfast',
       'Buzzer/Wireless Intercom', 'Cable TV', 'Carbon Monoxide Detector',
       'Cat(s)', 'Dog(s)', 'Doorman', 'Dryer', 'Elevator in Building',
       'Essentials', 'Family/Kid Friendly', 'Fire Extinguisher',
       'First Aid Kit', 'Free Parking on Premises', 'Gym', 'Hair Dryer',
       'Hangers', 'Heating', 'Hot Tub', 'Indoor Fireplace', 'Internet', 'Iron',
       'Kitchen', 'Laptop Friendly Workspace', 'Lock on Bedroom Door',
       'Other pet(s)', 'Pets Allowed', 'Pets live on this property', 'Pool',
       'Safety Card', 'Shampoo', 'Smoke Detector', 'Smoking Allowed',
       'Suitable for Events', 'TV', 'Washer', 'Washer / Dryer',
       'Wheelchair Accessible', 'Wireless Internet',
       'property_type_Bed & Breakfast', 'property_type_Boat',
       'property_type_Bungalow', 'property_type_Cabin',
       'property_type_Camper/RV', 'property_type_Chalet',
       'property_type_Condominium', 'property_type_Dorm',
       'property_type_House', 'property_type_Loft', 'property_type_Other',
       'property_type_Tent', 'property_type_Townhouse',
       'property_type_Treehouse', 'property_type_Yurt',
       'room_type_Private room', 'room_type_Shared room', 'bed_type_Couch',
       'bed_type_Futon', 'bed_type_Pull-out Sofa', 'bed_type_Real Bed']]
y = ML_DF['price']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3,random_state=42)
ML = LinearRegression(normalize=True)
ML.fit(X_train,y_train)

y_test_preds =ML.predict(X_test)
y_test_preds = ML.predict(X_test) 
r2_score(y_test, y_test_preds)

