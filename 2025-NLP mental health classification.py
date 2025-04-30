path = r"D:\datasets\NLP\Sentiment Analysis for Mental Health\Sentiment analysis - Combined Data.csv"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

# create our df
df = pd.read_csv(path)
print(df.head())
df.isna().sum()
df.dropna(inplace=True)
print(len(df))
print(df['status'].unique())
print(len(df['status'].unique()))
print(df['status'].value_counts())

# convert our status to one_hot_encoded 
one_hot_df = pd.get_dummies(df['status'] , prefix='status' , dtype= int)
print(one_hot_df.sample(5))

final_df = pd.concat([df,one_hot_df] , axis=1)
print(final_df.sample(5))

# create X,y
X = df['statement'].values
y = final_df.drop(columns=['Unnamed: 0','statement','status'] ).values
print(y[random.randint(10,1000)])
print(f'some random values of x : {X[random.randint(0,len(X)-1)]} ')

#split x,y values
from sklearn import metrics
from sklearn.model_selection import train_test_split
x_train,x_ts,y_train,y_ts = train_test_split(X,y,test_size=0.30)
x_test,x_valid ,y_test,y_valid = train_test_split(x_ts , y_ts , test_size=0.5)
# print(f" shape of x-train : {x_train.shape} , shape of x-test : {x_test.shape} , shape of x-validation : {x_valid.shape} ")
print(np.info(x_train))
# import transformers libraries
import tensorflow as tf
import keras
from keras import optimizers,losses
from transformers import AutoTokenizer,TFAutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
encoded_train = tokenizer(list(x_train) , truncation=True , padding= True , max_length=128 , return_tensors='tf')
encoded_test = tokenizer(list(x_test) , truncation=True , padding= True , max_length=128 , return_tensors='tf')
encoded_valid = tokenizer(list(x_valid) , truncation=True , padding= True , max_length=128 , return_tensors='tf')

y_train_tf = tf.convert_to_tensor(y_train , dtype=tf.int32)
y_test_tf = tf.convert_to_tensor(y_test , dtype=tf.int32)
y_valid_tf = tf.convert_to_tensor(y_valid , dtype=tf.int32)


# now we can creqte our dataset
train_data = tf.data.Dataset.from_tensor_slices((dict(encoded_train), y_train_tf))
test_data = tf.data.Dataset.from_tensor_slices((dict(encoded_test) , y_test_tf))
valid_data = tf.data.Dataset.from_tensor_slices((dict(encoded_valid) , y_valid_tf))

# shuffle data
train_data = train_data.shuffle(100).batch(16)
test_data = test_data.batch(16)
valid_data= valid_data.batch(16)



# create our model
model = TFAutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased' , num_labels = 7)
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

model.compile(optimizer= optimizer ,
              loss = loss,
              metrics = ['accuracy'])
history = model.fit(train_data , validation_data = valid_data , epochs = 3 )

test_loss , test_acc = model.evaluate(test_data)
print(f"our accuracy is : {test_acc:.4f} , and loss is : {test_loss:.4f}")

predictions = model.predict(test_data)
predicted_labels = tf.argmax(predictions.logits , axis=1).numpy()
y_true = np.argmax(y_test_tf , axis=1)


import seaborn as sns
from sklearn.metrics import confusion_matrix,classification_report
cm = confusion_matrix(y_true= y_true , y_pred = predicted_labels)

sns.heatmap(cm , cmap = 'coolwarm')

print(classification_report(cm) )