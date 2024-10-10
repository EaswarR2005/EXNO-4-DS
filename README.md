# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
## FEATURE SCALING:
```
import pandas as pd
from scipy import stats
import numpy as np
import pandas as pd
df=pd.read_csv("/content/bmi.csv")
df
```
![image](https://github.com/user-attachments/assets/740577a1-3c52-4ef5-95b6-b776db55a8d5)

```
df.head()
```
![image](https://github.com/user-attachments/assets/5a48e796-99be-476c-b598-0f918e6bc598)

```
import numpy as np
max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals
```
![image](https://github.com/user-attachments/assets/e4dbc3dd-527b-49a6-ba9e-4c8874280304)

```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)

```
![image](https://github.com/user-attachments/assets/91f16cf6-ea1c-47a9-9c11-116e96c6703d)

```
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/f616413f-d7ca-4319-aa70-c9dd80134d9a)

```
from sklearn.preprocessing import Normalizer
Scaler=Normalizer
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df
```
![image](https://github.com/user-attachments/assets/ca17bed4-36e1-4375-bec9-aece8768c3c9)

```
from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df
```
![image](https://github.com/user-attachments/assets/2faa88f4-ab9f-4700-adb9-bcb5556a69c8)

```
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head()
```
![image](https://github.com/user-attachments/assets/65bdb4d9-fdaa-4c91-83c1-8f70ad439be0)

## FEATURE SELECTION:
```
import pandas as pd
import numpy as np
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
df=pd.read_csv("/content/income(1) (1).csv",na_values=[" ?"])
df
```
![image](https://github.com/user-attachments/assets/1cff2c3a-ff14-42ab-a51d-f854059df935)

```
df.isnull().sum()
```
![image](https://github.com/user-attachments/assets/139753ce-31df-4e2d-8743-1a4011e05e81)

```
missing=df[df.isnull().any(axis=1)]
missing
```
![image](https://github.com/user-attachments/assets/e6096fc1-2f3e-4b4f-8021-9d196b4a8d68)

```
df2=df.dropna(axis=0)
df2
```
![image](https://github.com/user-attachments/assets/a5a00fd3-bcfa-4178-9904-d795e0290c72)

```
sal=df['SalStat']
df2['SalStat']=df2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(df2['SalStat'])
df2
```
![image](https://github.com/user-attachments/assets/dc505153-0bcd-403f-9681-2a18307f7117)

```
sal2=df2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![image](https://github.com/user-attachments/assets/f9d28b32-6516-491a-ac87-0fb455d5b041)

```
new_data=pd.get_dummies(df2,drop_first=True)
new_data
```
![image](https://github.com/user-attachments/assets/e744c3f1-e72f-41bb-8cf5-57ea0fc73db7)

```
columns_list=list(new_data.columns)
print(columns_list)
```
![image](https://github.com/user-attachments/assets/a7fd5c8a-32bb-44b6-93da-d1635e572c27)

```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![image](https://github.com/user-attachments/assets/90e2c77f-48d9-408b-ab98-d3ac13762151)

```
x=new_data[features].values
print(x)
```
![image](https://github.com/user-attachments/assets/db854775-aafe-494a-8aa4-e991bcfc85e1)

```
y=new_data['SalStat'].values
print(y)
```
![image](https://github.com/user-attachments/assets/dbe7fa93-0fe3-4b37-8350-3c6b6a1eeebf)

```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors=5)
KNN_classifier.fit(train_x,train_y)
```
![image](https://github.com/user-attachments/assets/b61ae834-4f41-4826-9dc2-5f9231b6a871)

```
prediction=KNN_classifier.predict(test_x)
print(prediction)
```
![image](https://github.com/user-attachments/assets/63fe3755-65b5-4ce0-af9d-a961b9b272e0)

```
confusionMmatrix=confusion_matrix(test_y,prediction)
print(confusionMmatrix)
```
![image](https://github.com/user-attachments/assets/375bb237-2d8e-42d5-96ce-f8e152b366e0)

```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
![image](https://github.com/user-attachments/assets/f83f921d-c228-47d1-996e-7e985542695a)

```
print('Misclassified samples: %d' %(test_y!=prediction).sum())
```
![image](https://github.com/user-attachments/assets/724ff038-9457-4f79-8480-52a78428e219)

```
df.shape
```
![image](https://github.com/user-attachments/assets/fa298d58-8bde-4f06-accd-65701a770612)

```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![image](https://github.com/user-attachments/assets/94450f24-42d4-4ed1-87b2-7f02c2ffd192)

```
contigency_table=pd.crosstab(tips['sex'],tips['time'])
print(contigency_table)
```
![image](https://github.com/user-attachments/assets/21d0abb8-336d-4c14-b5f8-a925d7554e69)

```
chi2, p, _, _=chi2_contingency(contigency_table)
print(f"Chi-square statistics:{chi2}")
print(f"p-value:{p}") 
```
![image](https://github.com/user-attachments/assets/1eade9a2-d87b-48d4-aafe-6425642ce067)

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif,f_classif
data={
    'Feature1':[1,2,3,4,5],
    'Feature2':['A','B','C','A','B'],
    'Feature3':[0,1,1,0,1],
    'Target':[0,1,1,0,1]
}
df=pd.DataFrame(data)
df
```
![image](https://github.com/user-attachments/assets/d5bc7b34-9e8e-4f17-af65-7f8ee12b030a)

```
x=df[['Feature1','Feature3']]
y=df['Target']
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![image](https://github.com/user-attachments/assets/59c81ad6-6813-4c89-af00-a7b6371c3384)

# RESULT:

Hence,Feature Scaling and Feature Selection process has been performed on the given data set.
