import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import timeit

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV


df = pd.read_csv("train.csv")
df_test = pd.read_csv("test_df.csv")
full_data = [df,df_test]

print(df_test.info())


#To See Age Distribution
sns.distplot(df.Age.dropna()) 
plt.show()

# Distributions according to Pclass
g = sns.FacetGrid(df, row = "Survived" , col = "Pclass")
g.map(sns.distplot,"Age")
plt.show()


#Correlation Matrix 
plt.subplots(figsize=(6,6))
sns.heatmap(df.corr(), annot=True, fmt=".2f")
plt.show()

#Filling Missing Values of Age
for dataset in full_data:
    age_avg 	   = dataset['Age'].mean()
    age_std 	   = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)
    
df['CategoricalAge'] = pd.cut(df['Age'], 5)

print (df[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean())

# Correlation Matrix 
k_m = df.pivot_table(index= "Survived", columns = "CategoricalAge" ,values = "Pclass" ,  aggfunc = np.median)
plt.subplots(figsize=(6,10))
sns.heatmap(k_m, annot=True, fmt=".4f")
plt.show()

print(df.isnull().sum()) 


# Categorization of Cabin Feature
  
df['Cabin']=df['Cabin'].fillna(0)
df.loc[df.Cabin.str[0]=="A", "Cabin"] = 1
df.loc[df.Cabin.str[0]=="B", "Cabin"] = 2
df.loc[df.Cabin.str[0]=="C", "Cabin"] = 3
df.loc[df.Cabin.str[0]=="D", "Cabin"] = 4
df.loc[df.Cabin.str[0]=="E", "Cabin"] = 5
df.loc[df.Cabin.str[0]=="F", "Cabin"] = 6
df.loc[df.Cabin.str[0]=="G", "Cabin"] = 7
df.loc[df.Cabin.str[0]=="T", "Cabin"] = 8


import re as re
def get_title(name):
	title_search = re.search(' ([A-Za-z]+)\.', name)
	# If the title exists, extract and return it.
	if title_search:
		return title_search.group(1)
	return ""

#Categorization of Name Feature
df['Title'] = df['Name'].apply(get_title)

df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df['Title'] = df['Title'].replace('Mlle', 'Miss')
df['Title'] = df['Title'].replace('Ms', 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
df['Title'] = df['Title'].map(title_mapping)

#Categorization of Sex Feature
df['Sex'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

#Categorization of Embarked Feature
df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0]) 
df['Embarked'] = df['Embarked'].map( {'S': 0, 'Q': 1,"C":2} ).astype(int)
    

#Categorization of Age Feature
df.loc[ df['Age'] <= 16, 'Age']= 0
df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age'] = 1
df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age']= 2
df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age']= 3
df.loc[ df['Age'] > 64, 'Age']= 4

#Dropped Zero Correlated Features
df.drop(['PassengerId',"Fare","Ticket","Name","CategoricalAge"],axis=1,inplace=True)


survived = df["Survived"]   
df.drop(['Survived'],axis=1,inplace=True)

######################################################################################################################◘3
# MODELLER
#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(df,survived,test_size=0.33, random_state=0)


## RANDOM FOREST CLASSIFIER
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 10, criterion = "entropy")
rfc.fit(x_train, y_train)
y_pred = rfc.predict(x_test)

cm = confusion_matrix(y_test,y_pred)
print("Random Forest")
print(cm)
###########################################################################################################################
#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(4,init = "uniform",activation = "relu", input_dim = 8))
classifier.add(Dense(4,init = "uniform",activation = "relu"))
classifier.add(Dense(1,init = "uniform",activation = "softmax"))

classifier.compile(optimizer = "adam", loss = "binary_crossentropy" , metrics = ["accuracy"])

classifier.fit(x_train,y_train,epochs = 50)
y_predYSA = classifier.predict(x_test)

y_predYSA = (y_pred > 0.5)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
sc = accuracy_score(y_test, y_predYSA, normalize=False)
print(sc)
cm = confusion_matrix(y_test, y_predYSA)
print("YSA")
print(cm)
print( 'Accuracy Score :',accuracy_score(y_test, y_predYSA))
#####################################################################################################################
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(x_train,y_train)
y_pred_nihai = classifier.predict(x_test)
#y_pred_nihai = (y_pred > 0.5)
cm = confusion_matrix(y_test, y_pred_nihai)
print("gxbost")
print(cm)
print( 'Accuracy Score GxBoost :',accuracy_score(y_test, y_pred_nihai) )
xgboostScore=accuracy_score(y_test, y_pred)

print( 'Accuracy Score :',accuracy_score(y_test, y_pred_nihai) )
#####################################################################
df_nihai = pd.read_csv("gender_submission.csv")
df_nihai.drop(['Survived'],axis=1,inplace=True)

y_pred_final = rfc.predict(df_test)
y_pred_final_df= pd.DataFrame(data = y_pred_final, index = range(418), columns=["Survived"] )
test_csv = pd.concat([df_nihai,y_pred_final_df],axis=1)
#s2.to_csv('titanic_nihai.csv',index=False)











