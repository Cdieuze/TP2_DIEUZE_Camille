#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# importation des bibliothèques

import pandas as pd
import sqldf
import matplotlib.pyplot as plt
import numpy as np
import nltk
import seaborn as sns

from datetime import datetime
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.ensemble import RandomForestRegressor


# In[28]:


# PARTIE 1 Importation données

df = pd.read_csv('C:\\Users\\dieuz\\Desktop\\Professionnels\\ESGF M2\\Machine_Learning\\TP2\\ONLINE_RETAILS.csv')
df.head()


# In[29]:


## Etape pré-processing

#enregistrements négatifs
df = df[df['Quantity']>=0] 

#doublons
df = df.drop_duplicates()

#valeurs vides = 0
df[['InvoiceNo', 'Quantity', 'UnitPrice', 'CustomerID']]= df[['InvoiceNo', 'Quantity', 'UnitPrice', 'CustomerID']].fillna(0)   

#traitement de la date
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
df['InvoiceDate']= df['InvoiceDate'].dt.strftime("%Y-%m-%d")
df['JJ'] = pd.DatetimeIndex(df['InvoiceDate']).day
df['MM'] = pd.DatetimeIndex(df['InvoiceDate']).month
df['AAAA'] = pd.DatetimeIndex(df['InvoiceDate']).year


df.head()


# In[85]:


#PARTIE 2 QUESTIONS

## QUESTION 1

#moyenne produits achetés par client pour une dépense
sql_customers = """
    SELECT CustomerID, InvoiceNo, avg(Quantity) as avg_nb_products
    FROM df
    group by CustomerID, InvoiceNo
"""


df_customers_avg = sqldf.run(sql_customers)
df_customers_avg

resultat = df_customers_avg.groupby('CustomerID').mean().reset_index()
resultat.head()


# In[112]:


#distribution sur top10 des clients (meilleur visibilité)
distri_clients = resultat.sort_values('avg_nb_products',ascending=False).head(10)
distri_clients.plot(kind='bar', x='CustomerID', y='avg_nb_products',color='b')


# In[97]:


#médianne et moyenne produits par pays

volume = pd.DataFrame()
volume['average'] = df.groupby('Country')['Quantity'].mean()
volume['median'] = df.groupby('Country')['Quantity'].apply(np.median)

volume = volume.reset_index()
volume.head()


# In[133]:


#distribution Top10 des pays (meilleur visibilité)

distri_volume = volume.head(10)

ax = plt.gca()
distri_volume.plot(kind='bar',x='Country',y='average',color='b',ax=ax)
distri_volume.plot(kind='bar',x='Country',y='median', color='skyblue', ax=ax)

plt.show()


# In[36]:


## QUESTION 2

#montant dépensé par client
sql_customer_UnitPrice = """
    SELECT CustomerID, InvoiceNo, avg(UnitPrice*Quantity) as mean
    FROM df
    group by CustomerID, InvoiceNo
"""

df_customer_avg_price = sqldf.run(sql_customer_UnitPrice)
df_customer_avg_price.head()


# In[37]:


#moyenne générale montant par client

amount_customers = df_customer_avg_price.groupby('CustomerID').mean().reset_index()
amount_customers.head()


# In[113]:


#distribution top10

distri_amount_customers = amount_customers.sort_values('mean',ascending=False).head(10)
distri_amount_customers.plot(kind='bar', x='CustomerID', y='mean',color='b')


# In[45]:


#montant moyen et médian par pays
df_test = df.copy(deep=True)
amount = pd.DataFrame()


#montant dépensé = €*qté
df_test['amount_depense'] = df_test[['Quantity', 'UnitPrice']].apply(lambda x : (x['Quantity'] * x['UnitPrice']), 1)

#montant moyen dépensé par pays
amount['average'] = df_test.groupby('Country')['amount_depense'].mean()

#montant médian dépensé par pays
amount['median'] = df_test.groupby('Country')['amount_depense'].apply(np.median)

amount= amount.reset_index()

amount


# In[132]:


#distribution


fig, ax = plt.subplots(figsize=(30,30))
fig.tight_layout(pad=5)

def plot_hor_bar(subplot, df):
    plt.subplot(1,2,subplot)
    if subplot==1:
        ax = sns.barplot(y='Country', x='average', data=df,
                     color='b')
        plt.title("Montant moyen d'un panier client par pays",
          fontsize=30)
    else : 
        ax = sns.barplot(y='Country', x='median', data=df,
                     color='skyblue')
        plt.title("Montant median d'un panier client par pays",
          fontsize=30)
    plt.xticks(fontsize=30)
    plt.ylabel(None)
    plt.yticks(fontsize=30)
    sns.despine(left=True)
    ax.grid(False)
    ax.tick_params(bottom=True, left=False)
    return None

plot_hor_bar(1, amount[['Country', 'average']])
plot_hor_bar(2, amount[['Country', 'median']])

plt.show()


# In[55]:


## QUESTION 3

#Top5 trimestre 1
top_5_products_T1 = """
    SELECT StockCode as code_produits, sum(Quantity) as total_vente
    FROM df
    where AAAA=2011 and MM between 1 and 3
    group by StockCode 
    order by total_vente desc
    limit 5
"""

df_top_5_products_T1 = sqldf.run(top_5_products_T1)
df_top_5_products_T1


# In[57]:


#Top5 trimestre 1

top_5_products_T2 = """
    SELECT StockCode, sum(Quantity) as total_quantity
    FROM df
    where AAAA=2011 and MM between 4 and 6
    group by StockCode 
    order by total_quantity desc
    limit 5
"""
df_top_5_products_T2= sqldf.run(top_5_products_T2)
df_top_5_products_T2



# In[126]:


## QUESTION 4

#Top5 pays plus gros chiffre d'affaires

CA_country = """
    SELECT Country, sum(Quantity * UnitPrice) as CA
    FROM df
    where AAAA=2011 and MM between 1 and 3
    group by Country 
    order by CA desc
"""

df_CA_country = sqldf.run(CA_country)
df_CA_country = df_CA_country.head(5)
df_CA_country


# In[123]:


## QUESTION 5

#montant moyen et médian d'un panier client /Top5 pays

df_1 = df[(df['Country'].isin(df_CA_country.Country)) & (df['AAAA']==2011) & (df['MM'].between(1,3))]
df['amount_depense'] = df[['Quantity', 'UnitPrice']].apply(lambda x : (x['Quantity'] * x['UnitPrice']),1)
   



# In[124]:


evol = pd.DataFrame()

evol['median'] = df.groupby(['Country', 'MM'])['amount_depense'].apply(np.median)
evol['average'] = df.groupby(['Country', 'MM'])['amount_depense'].mean()

evol = evol.reset_index()

evol_Australia = evol[evol['Country']=='Australia']
evol_EIRE = evol[evol['Country']=='EIRE']
evol_France = evol[evol['Country']=='France']
evol_Netherlands = evol[evol['Country']=='Netherlands']
evol_United_Kingdom = evol[evol['Country']=='United Kingdom']


evol


# In[131]:


fig, ax = plt.subplots(figsize=(20,5))
x = np.arange(len(evol))
width = 0.5
plt.bar(x-0.2, evol['median'],
        width, color='tab:red', label='median')
plt.bar(x+0.2, evol['average'],
        width, color='gold', label='average')
plt.title('montant moyen et median panier client par pays', fontsize=25)
plt.xlabel(None)
plt.xticks(evol.index, evol['Country'], fontsize=8)
plt.yticks(fontsize=17)
sns.despine(bottom=True)
ax.grid(False)
ax.tick_params(bottom=False, left=True)
plt.legend(frameon=False, fontsize=10)

plt.show()


# In[129]:


# QUESTION 6

#Fréquence de dépense des Top100 clients du UK
united_kingdom_top100 = """
    SELECT CustomerID, sum(Quantity) quantity, sum(Quantity*UnitPrice) price
    FROM df
    where Country='United Kingdom'
    group by CustomerID 
    order by quantity desc
    limit 100
"""

top100_united_kingdom = sqldf.run(united_kingdom_top100)
top100_united_kingdom


# In[130]:


#quantité moyenne
print(top100_united_kingdom.quantity.mean())

#prix moyen
print(top100_united_kingdom.price.mean())


# In[135]:


#PARTIE 3 MODELE

## CA global de Décembre 2011

df_test = df.copy(deep=True)
df_test['amount_depense'] = df_test[['Quantity', 'UnitPrice']].apply(lambda x : (x['Quantity'] * x['UnitPrice']), 1)
df_model = df_test[['InvoiceDate', 'UnitPrice', 'Quantity', 'amount_depense', 'Country', 'AAAA', 'MM', 'JJ']]
df_model.head()


# In[137]:


df_model = df_model.groupby(['InvoiceDate', 'AAAA', 'MM', 'JJ']).agg({'amount_depense':'sum'}).reset_index()
df_model


# In[141]:


#model training comme TP1
x = df_model[['AAAA', 'MM', 'JJ']]
y = df_model['amount_depense']

X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.5)    

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[144]:


#regression
model = RandomForestRegressor()

rF = model.fit(X_train, Y_train)
predict = rF.predict(X_test)
rF.score(X_train, Y_train)


# In[149]:


#prédiction
model = RandomForestRegressor()
rF = model.fit(x, y)

#jours complets décembre 2011
list_day = [i for i in range(10,32)]
list_day.append(3)

df_december = pd.DataFrame.from_dict({'AAAA': [2011 for i in range(len(list_day))],
                'MM': [12 for i in range(len(list_day))],
                'JJ': list_day
                })

df_december_complet = df_december.copy(deep=True)
df_december_complet['amount_depense']=rf.predict(df_december)
df_december_complet.head()


# In[155]:


df_model_december = df_model[(df_model['AAAA']==2011) & (df_model['MM']==12)][['AAAA', 'MM', 'JJ','amount_depense']]

df_december_complet = pd.concat([df_model_december, df_december_complet], ignore_index=True)

df_december_complet.sort_values('JJ')


# In[158]:


print("CA de décembre sera de : ")
df_december_complet.amount_depense.sum()


# In[159]:


##  Au moins un achat en Décembre 2011 par clients
df_purchase = df[['InvoiceNo', 'CustomerID', 'InvoiceDate', 'UnitPrice', 'Quantity', 'Country', 'AAAA', 'MM', 'JJ']]

df_purchase = df_purchase.groupby(['Country', 'CustomerID', 'InvoiceDate', 'AAAA', 'MM', 'JJ']).agg({'InvoiceNo':'count'})

df_purchase = df_purchase.reset_index()

df_purchase = df_purchase.rename({'InvoiceNo': 'purchase_count'}, axis=1)  

df_purchase


# In[161]:


#model training

x_puchase = df_purchase[['AAAA', 'MM', 'JJ']]
y_purchase = df_purchase['purchase_count']

rf_purchase = model.fit(x_puchase, y_purchase)

list_day_purchase = [i for i in range(10,32)]
list_day_purchase.append(3)


# In[169]:



december_purchase = pd.DataFrame.from_dict({'AAAA': [2011 for i in range(len(list_day_purchase))],
                'MM': [12 for i in range(len(list_day_purchase))],
                'JJ': list_day_purchase
                })
december_purchase.sort_values('JJ')
december_purchase['purchase_count']=rf.predict(december_purchase)

december_purchase


# In[175]:


df_purchase_december = df_purchase[(df_purchase['AAAA']==2011) & (df_purchase['MM']==12 ) & (df_purchase['Country']=='France')][['CustomerID','AAAA', 'MM', 'JJ','purchase_count']]


df_purchase_december = pd.concat([df_purchase_december, december_purchase], ignore_index=True)


df_purchase_december.sort_values('JJ')


# In[172]:


print("Nombre clients qui réalisera au moins un achat en France en décembre 2011 : ")
df_purchase_december.count()

