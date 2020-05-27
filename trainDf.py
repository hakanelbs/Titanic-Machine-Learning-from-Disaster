# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 20:38:13 2020

@author: Predator
"""

### BU ÇALIŞMADA 3 PARAMETRE VE DECISION TREE VE RF ILE CLASSIFICATION TAHMINLEMESİ YAPTIK


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

df3 = pd.read_csv("train.csv")
df2 = pd.read_csv("train.csv")
df = pd.read_csv("train.csv")
df_test = pd.read_csv("testDF.csv")
full_data = [df,df_test]

print(df_test.info())

#Yaş dağılımını görmek için
sns.distplot(df.Age.dropna()) 
plt.show()


# FacetGrid : Bir Grafiği kategorilere ayırır
g = sns.FacetGrid(df, row = "Survived" , col = "Pclass")
g.map(sns.distplot,"Age")
plt.show()




# =============================================================================
# jointplot : Bu metot, veri değişikliklerinin hem dağılımları hem de çekirdek yoğunluğu tahmin edicileri ve verilere uyan bir opsiyonel
# regresyonile birlikte iki değişkene göre görüntülenmesi için kullanılır. 
# Reg ile, verilere uygun bir regresyon istediğimizi belirtiyoruz. Bu durumda regresyon gösterdiği yukarı doğru küçük bir eğilim olduğu
# görünsede, Pearson korelasyon katsayısı ile gösterildiği gibi "yaş" ve "ücret" değişkenleri arasında hemen hemen hiçbir ilişki yoktur
# 
# =============================================================================

sns.jointplot(data=df , x="Age" , y = "Fare" , kind = "reg" , color = "green")
plt.show()
sns.jointplot(data=df , x="Age" , y = "Survived" , kind = "reg" , color = "blue")
plt.show()





#Korelasyon matrisi : 
plt.subplots(figsize=(6,6))
sns.heatmap(df.corr(), annot=True, fmt=".2f")
plt.show()


for dataset in full_data:
    age_avg 	   = dataset['Age'].mean()
    age_std 	   = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)
    
df2['CategoricalAge'] = pd.cut(df2['Age'], 5)

print (df2[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean())



k_m = df2.pivot_table(index= "Survived", columns = "CategoricalAge" ,values = "Pclass" ,  aggfunc = np.median)
plt.subplots(figsize=(6,10))
sns.heatmap(k_m, annot=True, fmt=".4f")
plt.show()
print(df.isnull().sum()) 








# Veri DÜZENLEME
  

df['Cabin']=df['Cabin'].fillna(0)
df.loc[df.Cabin.str[0]=="A", "Cabin"] = 1
df.loc[df.Cabin.str[0]=="B", "Cabin"] = 2
df.loc[df.Cabin.str[0]=="C", "Cabin"] = 3
df.loc[df.Cabin.str[0]=="D", "Cabin"] = 4
df.loc[df.Cabin.str[0]=="E", "Cabin"] = 5
df.loc[df.Cabin.str[0]=="F", "Cabin"] = 6
df.loc[df.Cabin.str[0]=="G", "Cabin"] = 7
df.loc[df.Cabin.str[0]=="T", "Cabin"] = 8


#sex = df.iloc[:,3:4].values
#print(sex)
#from sklearn.preprocessing import LabelEncoder
#le = LabelEncoder()
#sex[:,0] = le.fit_transform(sex[:,0])
#print(sex)
#sex = pd.DataFrame(data = sex, index = range(891), columns=["sex"] )
#df= pd.concat([sex,df],axis=1)
#df.drop(['Sex'],axis=1,inplace=True)




#df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])      # Tek Tek Düzeltmek Stringler
#emberked = df.iloc[:,11:12].values
#from sklearn.preprocessing import LabelEncoder
#le2 = LabelEncoder()
#emberked[:,0] = le2.fit_transform(emberked[:,0])
#emberked = pd.DataFrame(data = emberked, index = range(891), columns=["emberked"] )
#df= pd.concat([emberked,df],axis=1)
#df.drop(['Embarked'],axis=1,inplace=True)




import re as re
def get_title(name):
	title_search = re.search(' ([A-Za-z]+)\.', name)
	# If the title exists, extract and return it.
	if title_search:
		return title_search.group(1)
	return ""


df['Title'] = df['Name'].apply(get_title)

df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df['Title'] = df['Title'].replace('Mlle', 'Miss')
df['Title'] = df['Title'].replace('Ms', 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
df['Title'] = df['Title'].map(title_mapping)





df['Sex'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)




df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0]) 
df['Embarked'] = df['Embarked'].map( {'S': 0, 'Q': 1,"C":2} ).astype(int)
    



   
    
#df['CategoricalAge'] = pd.cut(df['Age'], 5)

df.loc[ df['Age'] <= 16, 'Age']= 0
df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age'] = 1
df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age']= 2
df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age']= 3
df.loc[ df['Age'] > 64, 'Age']= 4



df.drop(['PassengerId',"Fare","Ticket","Name"],axis=1,inplace=True)


survived = df["Survived"]   
df.drop(['Survived'],axis=1,inplace=True)



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






from xgboost import XGBClassifier
classifier = XGBClassifier()

import xgboost
classifier=xgboost.XGBRegressor()


regressor=xgboost.XGBRegressor()

classifier.fit(x_train,y_train)
y_pred_nihai = classifier.predict(x_test)
y_pred_nihai = (y_pred > 0.5)
cm = confusion_matrix(y_test, y_pred_nihai)
print("gxbost")
print(cm)
print( 'Accuracy Score GxBoost :',accuracy_score(y_test, y_pred_nihai) )
xgboostScore=accuracy_score(y_test, y_pred)
#plt.xlabel('Accuracy')
#plt.title('Classifier Accuracy')
#
#sns.set_color_codes("muted")
#sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")




print( 'Accuracy Score :',accuracy_score(y_test, y_pred_nihai) )








rfc2 = RandomForestClassifier(n_estimators = 120,max_depth = 5, criterion = "entropy",min_samples_leaf = 2)
rfc2.fit(x_train, y_train)
y_pred2 = rfc2.predict(x_test)

cm = confusion_matrix(y_test,y_pred2)
print("Random Forest Parametreli")
print("cm 2" ,cm)    


from sklearn.metrics import accuracy_score 
rfc3 = RandomForestClassifier(n_estimators = 300,max_depth = 5, criterion = "gini",min_samples_leaf = 2)
rfc3.fit(x_train, y_train)
y_pred3 = rfc3.predict(x_test)

cm = confusion_matrix(y_test,y_pred3)
print("Random Forest Parametreli 2")
print("cm 3 " , cm)    
print( 'Accuracy Score RandomForest:',accuracy_score(y_test, y_pred3) )
rfScore=accuracy_score(y_test, y_pred)






#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
#Yapay Sinir Ağı

#import keras
#from keras.models import Sequential
#from keras.layers import Dense
#
#classifier = Sequential()
#classifier.add(Dense(4,init = "uniform",activation = "relu", input_dim = 8))
#classifier.add(Dense(4,init = "uniform",activation = "relu"))
#classifier.add(Dense(1,init = "uniform",activation = "sigmoid"))
#
#classifier.compile(optimizer = "adam", loss = "binary_crossentropy" , metrics = ["accuracy"])
#
#classifier.fit(X_train,y_train,epochs = 100)
#y_predYSA = classifier.predict(X_test)
#
#y_predYSA = (y_pred > 0.5)
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import accuracy_score
#sc = accuracy_score(y_test, y_predYSA, normalize=False)
#print(sc)
#cm = confusion_matrix(y_test, y_predYSA)
#print("YSA")
#print(cm)
#
#
#
#
#df_nihai = pd.read_csv("gender_submission.csv")
#df_nihai.drop(['Survived'],axis=1,inplace=True)
#
#
#rfc2 = RandomForestClassifier(n_estimators = 120,max_depth = 5, criterion = "entropy",min_samples_leaf = 2)
#rfc2.fit(x_train, y_train)
#y_pred22 = rfc2.predict(df_test)
#
#y_pred222 = pd.DataFrame(data = y_pred22, index = range(418), columns=["Survived"] )
#s2= pd.concat([df_nihai,y_pred222],axis=1)
#s2.to_csv('titanic_nihai.csv',index=False)
#




import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(4,init = "uniform",activation = "relu", input_dim = 8))
classifier.add(Dense(4,init = "uniform",activation = "relu"))
classifier.add(Dense(1,init = "uniform",activation = "sigmoid"))

classifier.compile(optimizer = "adam", loss = "binary_crossentropy" , metrics = ["accuracy"])

classifier.fit(X_train,y_train,epochs = 250)
y_predYSA = classifier.predict(X_test)

y_predYSA = (y_pred > 0.5)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
sc = accuracy_score(y_test, y_predYSA, normalize=False)
print(sc)
cm = confusion_matrix(y_test, y_predYSA)
print("YSA")
print(cm)




df_nihai = pd.read_csv("gender_submission.csv")
df_nihai.drop(['Survived'],axis=1,inplace=True)


rfc2 = RandomForestClassifier(n_estimators = 120,max_depth = 5, criterion = "entropy",min_samples_leaf = 2)
rfc2.fit(x_train, y_train)
y_pred22 = rfc2.predict(df_test)

y_pred222 = pd.DataFrame(data = y_pred22, index = range(418), columns=["Survived"] )
s2= pd.concat([df_nihai,y_pred222],axis=1)
#s2.to_csv('titanic_nihai.csv',index=False)





