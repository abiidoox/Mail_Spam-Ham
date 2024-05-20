#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


# In[2]:


#DATA LOADING


# In[3]:


mail = pd.read_csv('spam.csv', encoding='latin-1')


# In[4]:


mail


# In[5]:


#sms.drop(['Unnamed: 2','Unnamed: 3', 'Unnamed: 4'], axis = 1, inplace = True)
mail.dropna(how="any", inplace=True, axis=1)
mail.columns = ['label', 'message']
mail


# In[6]:


#Exploratory Data Analysis (EDA)


# In[7]:


mail.describe()


# In[8]:


mail.groupby('label').describe()


# In[9]:


mail['label_num'] = mail.label.map({'ham':0, 'spam':1})
mail


# In[10]:


mail['message_len'] = mail.message.apply(len)
mail


# In[11]:


plt.figure(figsize=(12, 8))

mail[mail.label=='ham'].message_len.plot(bins=35, kind='hist', color='blue', 
                                       label='Ham messages', alpha=0.6)
mail[mail.label=='spam'].message_len.plot(kind='hist', color='red', 
                                       label='Spam messages', alpha=0.6)
plt.legend()
plt.xlabel("Message Length")


# In[12]:


mail[mail.label=='ham'].describe()


# In[13]:


mail[mail.label=='spam'].describe()


# In[14]:


mail[mail.message_len == 910].message.iloc[0]


# In[15]:


#TEXT PreProcessing


# In[16]:


# Initialize the stemmer and stopwords
stemmer = PorterStemmer()
stopwords = set(stopwords.words('english') + ['u', 'ü', 'ur', '4', '2', 'im', 'dont', 'doin', 'ure'])

def clean_text(text):
    # Remove HTML tags
    text = re.sub('<.*?>', '', text)
    # Remove non-alphabetic characters and convert to lowercase
    text = re.sub('[^a-zA-Z]', ' ', text).lower()
    # Remove Mentions
    text = re.sub(r'@\S+', '', text)
    # Remove Hashtags
    text = re.sub(r'#\S+', '', text)
    # Remove stopwords and stem the words
    words = [stemmer.stem(w) for w in text.split() if w not in stopwords]
    # Join the words back into a string
    cleaned_text = ' '.join(words)
    return text


# In[17]:


mail['clean_msg'] = mail.message.apply(clean_text)

mail


# In[18]:


words = mail[mail.label=='ham'].clean_msg.apply(lambda x: [word.lower() for word in x.split()])
ham_words = Counter()

for msg in words:
    ham_words.update(msg)
    
print(ham_words.most_common(50))


# In[19]:


words = mail[mail.label=='spam'].clean_msg.apply(lambda x: [word.lower() for word in x.split()])
spam_words = Counter()

for msg in words:
    spam_words.update(msg)
    
print(spam_words.most_common(50))


# In[20]:


#SplitDATA and Vectorize


# In[21]:


# split X and y into training and testing sets 

X = mail.clean_msg
y = mail.label_num
print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[22]:


# instantiate the vectorizer
vect = CountVectorizer()

# learn training data vocabulary, then use it to create a document-term matrix
X_train_dtm = vect.fit_transform(X_train)

# examine the document-term matrix
print(type(X_train_dtm), X_train_dtm.shape)

# transform testing data (using fitted vocabulary) into a document-term matrix
X_test_dtm = vect.transform(X_test)
print(type(X_test_dtm), X_test_dtm.shape)


# In[23]:


#Building and evaluating the model


# In[24]:


SVM = SVC()


# In[25]:


SVM.fit(X_train_dtm, y_train)


# In[26]:


# make class predictions for X_test_dtm
y_pred_class = SVM.predict(X_test_dtm)

# calculate accuracy of class predictions
print("=======Accuracy Score===========")
print(metrics.accuracy_score(y_test, y_pred_class))

# print the confusion matrix
print("=======Confision Matrix===========")
metrics.confusion_matrix(y_test, y_pred_class)


# In[27]:


KN = KNeighborsClassifier()


# In[28]:


KN.fit(X_train_dtm, y_train)


# In[29]:


y_pred_class = KN.predict(X_test_dtm)

# calculate accuracy of class predictions
print("=======Accuracy Score===========")
print(metrics.accuracy_score(y_test, y_pred_class))

# print the confusion matrix
print("=======Confision Matrix===========")
print(metrics.confusion_matrix(y_test, y_pred_class))


# In[30]:


DT = DecisionTreeClassifier()


# In[31]:


DT.fit(X_train_dtm, y_train)


# In[32]:


y_pred_class = DT.predict(X_test_dtm)

# calculate accuracy of class predictions
print("=======Accuracy Score===========")
print(metrics.accuracy_score(y_test, y_pred_class))

# print the confusion matrix
print("=======Confision Matrix===========")
print(metrics.confusion_matrix(y_test, y_pred_class))


# In[33]:


#Comparing the models


# In[34]:


# Classifiers
names = [
    "KNN Classifier",
    "Decision Tree",
    "SVM",  
]

models = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    SVC(),
    
]


# In[35]:


def score(X_train, y_train, X_test, y_test, names, models):
    score_df = pd.DataFrame()
    score_train = []
    confusion_matrices = []
    
    for name, model in zip(names, models):
        model.fit(X_train, y_train)
        y_pred_class = model.predict(X_test)
        score_train.append(metrics.accuracy_score(y_test, y_pred_class))
        confusion_matrices.append(metrics.confusion_matrix(y_test, y_pred_class))
    
    score_df["Classifier"] = names
    score_df["Training accuracy"] = score_train
    score_df["Confusion matrix"] = confusion_matrices
    score_df.sort_values(by='Training accuracy', ascending=False, inplace=True)
    
    return score_df


# In[36]:


score(X_train_dtm, y_train, X_test_dtm, y_test, names=names, models=models)


# In[37]:


#Test the models


# In[40]:


def classify_email(email):
    cleaned_email = clean_text(email)
    email_dtm = vect.transform([cleaned_email])
    prediction = SVM.predict(email_dtm)
    if prediction[0] == 0:
        return 'DT : ham'
    else:
        return 'DT : spam'


# In[41]:


email = "Hello, this is a legitimate email."
classification = classify_email(email)
print(classification)  

email = "Congratulations! You have won a prize. Click here to win it for free."
classification = classify_email(email)
print(classification)


# In[ ]:




