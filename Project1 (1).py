#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
import tensorflow as tf
import datetime as dt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Download data from 2010 to 2021 for training
dataset_train = yf.download('^BSESN', start='2010-03-31', end='2021-03-31')

# Check for missing values
print("Null values:", dataset_train.isnull().values.sum())
print("NA values:", dataset_train.isna().values.any())
dataset_train.info()

# Data visualization
train = dataset_train.reset_index()

# Normalization
sc = MinMaxScaler(feature_range=(0,1))
train_set = train.iloc[:, 1:2].values
train_set_scaled = sc.fit_transform(train_set)

# Creating X_train and y_train
X_train = []
y_train = []
for i in range(60, len(train_set_scaled)):
    X_train.append(train_set_scaled[i-60:i, 0])
    y_train.append(train_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshape the data
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Model building
lstm_model = Sequential()
lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(units=50, return_sequences=True))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(units=50, return_sequences=True))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(units=1))

# Compile the model
lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
lstm_model.fit(X_train, y_train, epochs=100, batch_size=50)

# Download test data from 2021 to 2022
dataset_test = yf.download('^BSESN', start='2021-04-01', end='2023-03-31')
test = dataset_test.reset_index()
test = test.drop(['Adj Close'], axis=1)

# Preparing the input data for the model
dataset_total = pd.concat((train['Open'], test['Open']), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

X_test = []
for i in range(60, len(inputs)):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Predicting stock values
predicted_bse_stock_price = lstm_model.predict(X_test)
predicted_bse_stock_price = sc.inverse_transform(predicted_bse_stock_price)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(test['Open'].values, color='#0b1d78', label='BSE Stock Price')
plt.plot(predicted_bse_stock_price, color='#1fe074', label='Predicted BSE Stock Price')
plt.title('BSE Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Actual BSE Stock Price')
plt.legend()
plt.show()


# In[69]:


lstm_model.summary()


# In[2]:


get_ipython().system('pip install vaderSentiment')


# In[3]:


get_ipython().system('pip install textblob')


# In[4]:


#sentimental anlysis
from textblob import TextBlob
import nltk
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from nltk.corpus import stopwords
from textblob import Word


# In[5]:


indf = pd.read_csv("india-news-headlines.csv")
indf.head(10)


# In[6]:


indf['publish_date'] = pd.to_datetime(indf['publish_date'], format = '%Y%m%d')


# In[7]:


indf.head()


# In[8]:


indf.tail()


# In[9]:


indf = indf.drop(['headline_category'], axis=1)
indf.head()


# In[10]:


indf = indf.rename(columns={'publish_date':'date','headline_text': 'headline'})
indf.head()


# In[11]:


indf.columns


# In[12]:


start_date = pd.to_datetime('2010-03-31')
end_date = pd.to_datetime('2021-03-31')
indf = indf.loc[(indf['date'] >= start_date) & (indf['date'] <= end_date)]


# In[13]:


indf = indf.reset_index()
indf = indf.drop('index', axis=1)
indf.head


# In[14]:


indf['headline'] = indf.groupby(['date']).transform(lambda x: ' '.join(x))
indf = indf.drop_duplicates()
indf.reset_index(inplace = True, drop = True)


# In[15]:


indf.head()


# In[16]:


#merging
train_new = train.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low',
                         'Close': 'close', 'Volume': 'volume'})

train_new.head()


# In[17]:


bse_merge = pd.merge(train_new, indf, how='inner', on='date')
bse_merge


# In[18]:


# Case Conversion
bse_merge['headline'] = bse_merge['headline'].apply(lambda x: " ".join(x.lower() for x in x.split()))


# In[19]:


# Numbers
bse_merge['headline'] = bse_merge['headline'].str.replace('\d','')


# In[20]:


# Punctuations
bse_merge['headline'] = bse_merge['headline'].str.replace('[^\w\s]','')


# In[21]:


nltk.download('stopwords')


# In[22]:


sw = stopwords.words('english')
bse_merge['headline'] = bse_merge['headline'].apply(lambda x: " ".join(x for x in x.split() if x not in sw))


# In[23]:


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


# In[24]:


bse_merge['headline'] = bse_merge['headline'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))


# In[25]:


bse_merge.head()


# In[26]:


bse_merge['headline'][0:15]


# In[27]:


# Get subjectivity
def get_subjectivity(text):
    return TextBlob(text).sentiment.subjectivity

# Get polarity:

def get_polarity(text):
    return TextBlob(text).sentiment.polarity


# In[28]:


bse_merge['subjectivity'] = bse_merge['headline'].apply(get_subjectivity)
bse_merge['polarity'] = bse_merge['headline'].apply(get_polarity)


# In[29]:


bse_merge.head()


# In[30]:


pd.DataFrame(bse_merge['polarity']).plot(kind='density', figsize=(10, 6), 
                             color=['#0b1d78'])

plt.title("Density Plot of Polarity")
plt.show()


# In[31]:


pd.DataFrame(bse_merge['subjectivity']).plot(kind='density', figsize=(10, 6), 
                             color=['#1fe074'])

plt.title("Density Plot of Subjectivity")
plt.show()


# In[32]:


def getSIA(text):
    sia = SentimentIntensityAnalyzer()
    sentiment= sia.polarity_scores(text)
    return sentiment


# In[33]:


compound=[]
neg=[]
pos=[]
neu=[]
SIA=0

for i in range (0, len(bse_merge['headline'])):
    SIA= getSIA(bse_merge['headline'][i])
    compound.append(SIA['compound'])
    neg.append(SIA['neg'])
    pos.append(SIA['pos'])
    neu.append(SIA['neu'])


# In[34]:


bse_merge['compound']= compound
bse_merge['neg']= neg
bse_merge['pos']=pos
bse_merge['neu']=neu

bse_merge.head()


# In[73]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load data
data = pd.read_csv("datafile.csv")

# Define a dictionary to convert month names to numbers
month_dict = {
    'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
    'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12
}

# Strip any extra spaces and convert to lowercase
data['Month'] = data['Month'].str.strip().str.lower()

# Correct any misspelled month names
misspelled_months = {'marcrh': 'march'}  # Add any other known misspellings here
data['Month'] = data['Month'].replace(misspelled_months)

# Verify that all month names are valid
invalid_months = data[~data['Month'].isin(month_dict.keys())]
if not invalid_months.empty:
    print("Invalid month names found:\n", invalid_months)
else:
    # Map the Month names to numbers and create a new column with the date in 'yyyy-mm-dd' format
    data['Date'] = pd.to_datetime(data.apply(lambda row: f"{row['Year']}-{month_dict[row['Month']]:02d}-01", axis=1))

    data.drop(columns=['Year', 'Month'], inplace=True)

    # Show the resulting DataFrame
    #print(data.head())
data = data[data['Sector'] == 'Rural+Urban']
# Drop the 'Sector' column as it's no longer needed
data = data.drop(columns=['Sector'])
print(data.head())


# In[74]:


from sklearn.metrics import mean_squared_error
# Handle missing values
#data.fillna(method='ffill', inplace=True)

# Replace '-' with NaN
data.replace('-', np.nan, inplace=True)

# Convert all columns to numeric
data = data.apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN values
data.dropna(inplace=True)

# Select relevant features and target variable
features = data[['Cereals and products', 'Meat and fish', 'Egg', 'Milk and products', 'Oils and fats', 'Fruits', 'Vegetables', 'Pulses and products', 'Sugar and Confectionery', 'Spices', 'Non-alcoholic beverages', 'Prepared meals, snacks, sweets etc.', 'Food and beverages', 'Pan, tobacco and intoxicants', 'Clothing', 'Footwear', 'Clothing and footwear', 'Housing', 'Fuel and light', 'Household goods and services', 'Health', 'Transport and communication', 'Recreation and amusement', 'Education', 'Personal care and effects', 'Miscellaneous']]
target = data['General index']

# Normalize the data
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)
target_scaled = scaler.fit_transform(target.values.reshape(-1, 1))

# Convert to supervised learning problem
def create_supervised_data(features, target, time_steps=1):
    X, y = [], []
    for i in range(len(features) - time_steps):
        X.append(features[i:i+time_steps, :])
        y.append(target[i + time_steps])
    return np.array(X), np.array(y)

time_steps = 3
X, y = create_supervised_data(features_scaled, target_scaled, time_steps)

# Split into training and testing sets
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build and train the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Make predictions
predictions = model.predict(X_test)

# Inverse transform predictions and true values
predictions_inv = scaler.inverse_transform(predictions)
y_test_inv = scaler.inverse_transform(y_test)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test_inv, predictions_inv))
print("RMSE:", rmse)

# Evaluate the model
#import matplotlib.pyplot as plt

plt.plot(y_test_inv, label='True CPI')
plt.plot(predictions_inv, label='Predicted CPI')
plt.legend()
plt.show()



# In[76]:


model.summary()


# In[37]:


#To convert these Unix timestamps back to a human-readable date format
data['Date'] = pd.to_datetime(data['Date'], unit='ns')

columns_to_drop = ['Cereals and products', 'Meat and fish', 'Egg', 'Milk and products',
       'Oils and fats', 'Fruits', 'Vegetables', 'Pulses and products',
       'Sugar and Confectionery', 'Spices', 'Non-alcoholic beverages',
       'Prepared meals, snacks, sweets etc.', 'Food and beverages',
       'Pan, tobacco and intoxicants', 'Clothing', 'Footwear',
       'Clothing and footwear', 'Housing', 'Fuel and light',
       'Household goods and services', 'Health', 'Transport and communication',
       'Recreation and amusement', 'Education', 'Personal care and effects',
       'Miscellaneous']

# Drop the specified columns
data.drop(columns=columns_to_drop, inplace=True)
print(data)



# In[38]:


# Convert 'date' columns to datetime format
bse_merge['date'] = pd.to_datetime(bse_merge['date'])
data['Date'] = pd.to_datetime(data['Date'])

# Extract year and month components
bse_merge['Year_Month'] = bse_merge['date'].dt.to_period('M')
data['Year_Month'] = data['Date'].dt.to_period('M')

# Merge the datasets on the 'Year_Month' column
merged_data = pd.merge(bse_merge, data, on='Year_Month', how='inner')

# Drop the 'Year_Month' column as it's no longer needed
merged_data.drop(columns=['Year_Month'], inplace=True)
# Display the merged dataset in table format

# Display the merged dataset
print(merged_data.head())


# In[41]:


df = merged_data[['close', 'subjectivity', 'polarity', 'compound', 'neg', 'pos', 'neu', 'General index']]
df.head()


# In[44]:


'''sc2 = MinMaxScaler()
df_scaled = pd.DataFrame(sc2.fit_transform(df))
df_scaled.columns = df.columns
df_scaled.index = df.index
df_scaled.head(100)
'''


# In[45]:


X = df_scaled.drop("close", axis=1) # drop close column
y = df_scaled["close"] # only the close column


# In[55]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[56]:


get_ipython().system('pip install catboost')


# In[57]:


from catboost import CatBoostRegressor
cb = CatBoostRegressor(iterations = 200,
                      learning_rate = 0.1,
                      depth = 5)
cb.fit(X_train, y_train)
y_pred_cat = cb.predict(X_test)



# In[1]:


import matplotlib.pyplot as plt

# Scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_cat, alpha=0.6, color='b')

# Add a line for perfect predictions
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='r', linewidth=2)

# Add labels and title
plt.xlabel('Actual Values (y_test)')
plt.ylabel('Predicted Values (y_pred_cat)')
plt.title('Actual vs Predicted Values CatBoost mode')
plt.show()



# In[68]:


residuals = y_test - y_pred_cat

# Residual plot
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_rf, residuals, alpha=0.6, color='b')
plt.hlines(y=0, xmin=min(y_pred_rf), xmax=max(y_pred_cat), colors='r', linewidth=2)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot for CatBoost')
plt.show()


# In[58]:


from sklearn import metrics

mse_cb = metrics.mean_squared_error(y_test, y_pred_cat)
rmse_cb = np.sqrt(mse_cb)

print("RMSE for CatBoost Model: ",rmse_cb)


# In[59]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Assuming y_true contains the true target values and y_pred contains the predicted values
# Calculate RMSE
rmse = mean_squared_error(y_test, y_pred_cat, squared=False)

# Calculate MAE
mae = mean_absolute_error(y_test, y_pred_cat)

# Calculate R-squared
r2 = r2_score(y_test, y_pred_cat)

print("RMSE:", rmse)
print("MAE:", mae)
print("R-squared:", r2)



# In[60]:


from sklearn.ensemble import RandomForestRegressor


# In[65]:


import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np

# Fit and predict with Random Forest
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Calculate RMSE
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
print("RMSE for Random Forest Model: ", rmse_rf)

# Scatter plot for Random Forest predictions
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.6, color='b')

# Add a line for perfect predictions
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='r', linewidth=2)

# Add labels and title
plt.xlabel('Actual Values (y_test)')
plt.ylabel('Predicted Values (y_pred_rf)')
plt.title('Actual vs Predicted Values for Random Forest Model')
plt.show()


# In[62]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Assuming y_true contains the true target values and y_pred contains the predicted values
# Calculate RMSE
rmse = mean_squared_error(y_test, y_pred_rf, squared=False)

# Calculate MAE
mae = mean_absolute_error(y_test, y_pred_rf)

# Calculate R-squared
r2 = r2_score(y_test, y_pred_rf)
print("RMSE:", rmse)
print("MAE:", mae)
print("R-squared:", r2)


# In[66]:


# Calculate residuals
residuals = y_test - y_pred_rf

# Residual plot
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_rf, residuals, alpha=0.6, color='b')
plt.hlines(y=0, xmin=min(y_pred_rf), xmax=max(y_pred_rf), colors='r', linewidth=2)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot for Random Forest Model')
plt.show()


# In[ ]:




