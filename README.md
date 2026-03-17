<H3> NAME : HEMNATH R </H3>
<H3>REGISTER NO.: 212224240057 </H3>
<H3>EX. NO.1</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
from google.colab import files
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
df = pd.read_csv("/content/Churn_Modelling.csv")
df
x = df.iloc[:, :-1].values
print(x)
y  = df.iloc[:, -1].values
print(y)
df.duplicated() 
print(df.describe())
scaler = MinMaxScaler()
numeric_cols = df.select_dtypes(include=['number']).columns
df_scaled = pd.DataFrame(scaler.fit_transform(df[numeric_cols]), columns=numeric_cols)
print(df_scaled)
x_train , x_test ,y_train , y_test = train_test_split(x,y,test_size = 0.2)
print(x_train)
print(y_train)
x_train , x_test ,y_train , y_test = train_test_split(x,y,test_size = 0.2)
print(x_test)
print(y_test)
```


## OUTPUT:
# Dataset :

<img width="1455" height="457" alt="image" src="https://github.com/user-attachments/assets/48b71114-c312-476d-8d40-cb3b6730d015" />

# Spliting the data :

<img width="487" height="149" alt="image" src="https://github.com/user-attachments/assets/8fcbb6a0-79d8-430f-88bc-9578e0dafc9a" />


<img width="257" height="35" alt="image" src="https://github.com/user-attachments/assets/5e60f17d-8ac9-4408-949c-8fa556277b9d" />


# Finding Missing Value:

<img width="271" height="339" alt="image" src="https://github.com/user-attachments/assets/bef3c8f7-790f-4b32-90b3-aea649d0ac9f" />

# Finding Duplicates:

<img width="248" height="515" alt="image" src="https://github.com/user-attachments/assets/7506b7eb-7aef-4855-8949-ae5381f0d3fb" />

# Describe:

<img width="661" height="223" alt="image" src="https://github.com/user-attachments/assets/54c0faa4-6936-4f7c-96d6-10b30f45f9d3" />

# Label Encoding :

<img width="574" height="287" alt="image" src="https://github.com/user-attachments/assets/7c949db7-1566-4ec0-aca1-f85b22995927" />

# Training the dataset :

<img width="521" height="169" alt="image" src="https://github.com/user-attachments/assets/ac1d522d-7429-44ae-aed9-d7bc0ccf0db3" />

# Testing the dataset :

<img width="468" height="171" alt="image" src="https://github.com/user-attachments/assets/fc66f172-5163-41c0-ab52-f68350a9ac5f" />







## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


