# Creator: Israel Showell
# Start Date: 4/11/2024
# End Date: 4/11/2024
# Project: Music Predictor Machine Learning
# Version: 1.00

# Description:
"""
This is a machine learning program that will predict what kind of music a person may like.
I learned how to use Anaconda, Jupyter Notebooks, and how to implement, train, and test a basic machine learning model in Python.
"""

# Guide by @Programming with Mosh
# https://www.youtube.com/watch?v=7eh4d6sabA0
# Steps:
# 1. Import the Data
# 2. Clean the Data
# 3. Split the Data into Training/Test Sets
# 4. Create a Model
# 5. Train the Model
# 6. Make Predictions
# 7. Evaluate and Improve

# NOTE: Gender 1 is male, Gender 0 is female

# Imports the panda library as the abbreviation pd
import pandas as pd
# Imports the DecisionTreeClassifier from sklearn.tree that will find patterns in the data
from sklearn.tree import DecisionTreeClassifier

#This allows us to easily split our dataset into 2 sets for training and testing
from sklearn.model_selection import train_test_split

#This allows us to measure how accurate our predictions are
from sklearn.metrics import accuracy_score

# 1. Importing the data
music_data = pd.read_csv('music.csv')

# 2. Cleaning data is not needed, but we need to seperate the first 2 columns from the 3rd one.
# The first 2 columns will be the input data, and the 3rd will be the output 

# 3. Spliting the data up!
# X has first 2 columns, and this is our input data
X = music_data.drop(columns=['genre'])

# y has 3rd column, and this is our output data
y = music_data['genre']

# Here we pass the input and output data to the function
# We then allocate 20% of both of them to testing
# This will return a tuple that we split up in these 4 variables
# Each time we run this program, we get different accuracy score due to 
# this function taking random data from the input and output datasets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

#The less data we give our model, the less accurate it will be!

# 4. Creating a Model
model = DecisionTreeClassifier()

# This method takes input data and their values as arguments and adjusts the parameters of the model 
# to minimize the difference between the predicted output and the actual target values.
# We just pass the training datasets now
model.fit(X_train,y_train)

# 6. Make predictions
# This takes in a 2D array. 
# The [21,1] is asking the model, what does a 21 year old male like in music?
# The [22,0] is asking the model, what does a 22 year old female like in music?
predictions = model.predict(X_test)

# 7. Evaluate and Improve
# Allocate 70%-80% of your data to training
# Allocate 20%-30 of your data to testing

# We compare the predictions we get from X_test with the output data we have in y_test!
# The score is measured from 0 to 1
score = accuracy_score(y_test,predictions)
# Prints out the score based on the predictions by the model
# Currently the accuracy is between 75% to 100%
score
