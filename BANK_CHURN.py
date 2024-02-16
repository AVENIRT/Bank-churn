#!/usr/bin/env python
# coding: utf-8

# # The purpose of this work , is to train the data set to find the portion of customer succeptible to left the bank or to stay
# 
# 

# In[16]:


#### import neccessary library

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error


# In[17]:


#load data to a panda data frame

df = pd.read_csv('train_churn.csv')


# In[18]:


### check data head
df.head(10)


# In[19]:


### shape of our data
df.shape


# In[20]:


# return an array of column names

df.columns.values 


# In[21]:


df.info()


# In[23]:


## describe all data types, not limiting to numeric (floats) columns

df.describe(include=object)


# In[24]:


df.describe()


# In[25]:


#### FIND ANY MISSING VALUES
df.isnull().sum()


# In[26]:


#### count active value

df['IsActiveMember'].value_counts()


# In[27]:


### count Exited Member

df['Exited'].value_counts()


# # EDA: DATA VISUALIZATION

# In[28]:


# Histograms viz for different features
df.hist(bins=10, figsize=(13, 12))
plt.show()


# In[29]:


#### lets do some vizualisation
#### check to see Exited menber distribution


sns.countplot(x='Exited',hue='Exited', data=df)

plt.title('Exited status')


# In[14]:


# Plotting the bar chart to see the churn by gender

# Create a catplot with gender' on the x-axis, 'Value' as hue, and kind='count'
sns.catplot(x='Gender', hue='Exited', kind='count', data=df, height=8, aspect=2)
plt.title('gender status')
# Show the plot
plt.show()


# In[15]:


# Plotting the bar chart to see the churn geograohically

# Create a catplot with 'geography' on the x-axis, 'Value' as hue, and kind='count'
sns.catplot(x='Geography', hue='Exited', kind='count', data=df, height=8, aspect=2)
plt.title('Geography status')
# Show the plot
plt.show()


# In[16]:


# Create a catplot with 'having credit card  on the x-axis, 'Value' as hue, and kind='count'
sns.catplot(x='HasCrCard', hue='Exited', kind='count', data=df, height=8, aspect=2)
plt.title('HasCrCard')
# Show the plot
plt.show()


# In[110]:


# Calculating the mean credit score for each region
mean_Credit_Score  = df.groupby('Geography')['CreditScore'].mean()


# In[111]:


# Plotting the pie chart to see credit score average by region
plt.figure(figsize=(8, 6))
plt.pie(mean_Credit_Score, labels=mean_Credit_Score.index, autopct='%1.1f%%', startangle=140)
plt.title('credit score by Region')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.tight_layout()
plt.show()


# # DATA PREPROCESSING

# In[30]:


####### lest drop the columns that with no has a strong impact on our study.
# List of columns to be removed
columns_to_remove = ['id', 'Surname','CustomerId','Geography']

# Remove columns
df = df.drop(columns=columns_to_remove)

print(df)


# In[31]:


# One-hot encode the 'Gender' column
df_encoded = pd.get_dummies(df, columns=['Gender'], drop_first=True)


# In[32]:


# Separate features (X) and target variable (y)
X = df_encoded.drop('Exited', axis=1)  # Adjust 'target_column_name' to the name of your target column
y = df_encoded['Exited']


# In[33]:


# test and train model: We divide our data in to 2 set , using 33 percent for test.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33,random_state=42)


print("X_train shape: ", X_train.shape)
print("y_train shape: ", y_train.shape)
print("X_test shape: ", X_test.shape)
print("y_test shape: ", y_test.shape)


# In[34]:


# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[52]:


# Initialize the logistic regression model
LR = LogisticRegression()

# Train the model
LR.fit(X_train_scaled, y_train)

# Predictions on the test set
y_pred = LR.predict(X_test_scaled)

# Evaluate the model
LR_accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[53]:


# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Classification report
class_report = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(class_report)


# In[54]:


lr_mse = mean_squared_error(y_test, y_pred)
lr_rmse = np.sqrt(lr_mse)
print("Logistic Regression MSE:", lr_mse)
print("Logistic Regression RMSE:", lr_rmse)


# In[59]:


# Initialize the Gaussian Naive Bayes model
GB = GaussianNB()

# Train the model
GB.fit(X_train_scaled, y_train)

# Predictions on the test set
pred = GB.predict(X_test_scaled)

# Evaluate the model
GB_accuracy = accuracy_score(y_test,pred)
print("Accuracy:", GB_accuracy)


# In[60]:


# Confusion matrix
conf_matrix = confusion_matrix(y_test, pred)
print("Confusion Matrix:")
print(conf_matrix)

# Classification report
class_report = classification_report(y_test, pred)
print("\nClassification Report:")
print(class_report)


# In[63]:


GB_mse = mean_squared_error(y_test, pred)
GB_rmse = np.sqrt(GB_mse)
print("Gaussian Regression MSE:", GB_mse)
print("Gaussian Regression RMSE:", GB_rmse)


# In[46]:


# Initialize the DecisionTreeClassifier model
DTC = DecisionTreeClassifier(random_state=42)

# Train the model
DTC.fit(X_train_scaled, y_train)

# Predictions on the test set
DTC_pred = DTC.predict(X_test_scaled)

# Evaluate the model
DTC_accuracy = accuracy_score(y_test, DTC_pred)
print("Accuracy:", DTC_accuracy)


# In[47]:


# Confusion matrix
conf_matrix = confusion_matrix(y_test, DTC_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Classification report
class_report = classification_report(y_test, DTC_pred)
print("\nClassification Report:")
print(class_report)


# In[48]:


# Initialize the  RandomForestClassifier model
RFC =  RandomForestClassifier(random_state=42)

# Train the model
RFC.fit(X_train_scaled, y_train)

# Predictions on the test set
RFC_pred = RFC.predict(X_test_scaled)

# Evaluate the model
RFC_accuracy = accuracy_score(y_test, RFC_pred)
print("Accuracy:", RFC_accuracy)


# In[49]:


# Confusion matrix
conf_matrix = confusion_matrix(y_test, RFC_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Classification report
class_report = classification_report(y_test, RFC_pred)
print("\nClassification Report:")
print(class_report)


# In[50]:


#####Train with MLPClassifier(Multi-layer Perceptron Classifier)

MLP = MLPClassifier(random_state=42)

# Train the model
MLP.fit(X_train_scaled, y_train)

# Predictions on the test set
MLP_pred = MLP.predict(X_test_scaled)

# Evaluate the model
MLP_accuracy = accuracy_score(y_test, MLP_pred)
print("Accuracy:", MLP_accuracy)


# In[51]:


# Confusion matrix
conf_matrix = confusion_matrix(y_test, MLP_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Classification report
class_report = classification_report(y_test, MLP_pred)
print("\nClassification Report:")
print(class_report)


# In[65]:


predictors_group = ('Logistic Regression','MLP','Decision tree','Random Forest','XGB')
x_pos = np.arange(len(predictors_group))
accuracies = [LR_accuracy,MLP_accuracy,GB_accuracy,DTC_accuracy,RFC_accuracy]
    
plt.bar(x_pos, accuracies, align='center', color='blue')
plt.xticks(x_pos, predictors_group, rotation='vertical')
plt.ylabel('Accuracy (%)',)
plt.title(' Accuracies')
plt.show()


# To summarise, in this notebook, we have learned the root cause of churn in our nank from from different visualization.
# We also seen the benefits of performing feature encoding  the Pandas library.
# 
# Finally, we compared the accuracy of five machine algorithm, to predicting the customer EXIT . We concluded that XGBoost Classifier  is slighly more accurate than 4 others.due to thebetter accuracy score.

# In[ ]:




