#Downloading data
import numpy as np

medical_charges_url='https://raw.githubusercontent.com/JovianML/opendatasets/master/data/medical-charges.csv'
from urllib.request import urlretrieve
urlretrieve(medical_charges_url, 'medical.csv')

#we now create pandas dataframe to store the data.

import pandas as pd
medical_df=pd.read_csv('medical.csv')

#we can get some info on the dataset
medical_df.info()

medical_df.describe() # this is going to give us the mean,median ...... this tells us how sensible our data is.

# Now we will do some visualization of the data
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# these are settings for our graphs
sns.set_style('darkgrid')
matplotlib.rcParams['font.size']=14
matplotlib.rcParams['figure.figsize']=(10,6)
matplotlib.rcParams['figure.facecolor']='#00000000'

# Age
"""medical_df.age.describe()
fig=px.histogram(medical_df,
    x='age',
    
    
    marginal='box',
    nbins=47,
    title='Distribution of Age'

)
fig.update_layout(bargap=0.1)
fig.show()

medical_df.charges.corr(medical_df.age)"""

def rmse(targets, predictions):
    return np.sqrt(np.mean(np.square(targets-predictions)))


#we create a dataframe from the main dataframe for non smokers
"""non_smoker_df=medical_df[medical_df.smoker=='no']
ages=non_smoker_df.age
target=non_smoker_df.charges

from sklearn.linear_model import LinearRegression
inputs, targets=non_smoker_df[['age', 'bmi','children']], non_smoker_df['charges']

#Create and train model
model= LinearRegression().fit(inputs,targets)

#Generate predictions
predictions=model.predict(inputs)

#Print loss
loss=rmse(targets,predictions)
print('loss:',loss)

#we create a dataframe from the main dataframe for smokers
smoker_df=medical_df[medical_df.smoker=='yes']

from sklearn.linear_model import LinearRegression
inputs, targets=smoker_df[['age','bmi','children']], smoker_df['charges']

#Create and train model
model=LinearRegression().fit(inputs, targets)

#Generate predictions
predictions=model.predict(inputs)

#Print loss
loss=rmse(targets,predictions)
print('loss:',loss) """

#Now we do for all
#we convert the categorical smoker column to binary
smoker_code={'no':0,'yes':1}
medical_df['smoker_code']=medical_df.smoker.map(smoker_code)

#we convert the categorical sex clumn to binary
sex_code={'female':0,'male':1}
medical_df['sex_code']=medical_df.sex.map(sex_code)

# We want to include region but we would have to use one-hot encoding
"""from sklearn import preprocessing
enc=preprocessing.OneHotEncoder()
enc.fit(medical_df[['region']])
enc.categories_
one_hot=enc.transform(medical_df[['region']]).toarray()
medical_df[['northeast','northwest','southeast','southwest']]=one_hot"""

#shorter form of one hot encoding
medical_df[['northeast','northwest','southeast','southwest']]=pd.get_dummies(medical_df.region)
from sklearn.linear_model import LinearRegression

"""inputs, targets=medical_df[['age', 'bmi', 'children','smoker_code','sex_code','northeast','northwest','southeast','southwest']], medical_df['charges']

#create and train model
model=LinearRegression().fit(inputs,targets)

#Generate predictions
predictions=model.predict(inputs)

#print loss
loss=rmse(targets,predictions)
print('loss:',loss)

#print weights and bias
print(model.coef_, model.intercept_)"""


#Feature Scaling
from sklearn.preprocessing import StandardScaler

numeric_cols=['age', 'bmi','children']
scaler=StandardScaler()
scaler.fit(medical_df[numeric_cols])

scaled_inputs=scaler.transform(medical_df[numeric_cols])
cat_cols=['smoker_code','sex_code','northeast','northwest','southeast','southwest']
categorical_data=medical_df[cat_cols].values

inputs=np.concatenate((scaled_inputs,categorical_data), axis=1)
targets=medical_df.charges
model=LinearRegression().fit(inputs, targets)

#Generate predictions
predictions=model.predict(inputs)

#Print loss
loss=rmse(targets,predictions)
print('loss:',loss)

print(model.coef_, model.intercept_)


#Creating a test set
from sklearn.model_selection import train_test_split

inputs_train, inputs_test, target_train, target_test=train_test_split(inputs,targets,test_size=0.1)

model=LinearRegression().fit(inputs,targets)
predictions_test=model.predict(inputs_test)
loss=rmse(target_test,predictions_test)
print('Test Loss:', loss)

predictions_train=model.predict(inputs_train)
loss=rmse(target_train,predictions_train)
print(loss)