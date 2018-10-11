import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

#reading data from csv file with pandas library and changing label of columns
train_set = pd.read_csv("/Users/aisanaghazade/DM_HW3/spam.csv",encoding='latin-1')
train_set.columns = ['label','content','v3','v4','v5']
n = int(train_set.shape[0])

#cleaning data
del (train_set['v3'])
del (train_set['v4'])
del (train_set['v5'])

train_set['label'] = train_set.label.map({'ham':0, 'spam':1})

#dividing train_set to train and test to have more exact error rate
x_train, x_test, y_train, y_test = train_test_split(train_set['content'], train_set['label'], random_state = 1)
print(x_train.shape[0])
print(x_test.shape[0])

#creating bag of words
count_vector = CountVectorizer()
training_data = count_vector.fit_transform(x_train)
print(training_data)
testing_data = count_vector.transform((x_test))

#creating model
naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, y_train)

#predicting and assessment
preditions = naive_bayes.predict(testing_data)
print(accuracy_score(y_test, preditions))
print(recall_score(y_test, preditions))
print(precision_score(y_test, preditions))
print(f1_score(y_test,preditions))



