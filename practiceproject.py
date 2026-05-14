import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

#loading dataset
df=pd.read_csv("house_prices.csv")
print(df.head())

#checking size and type 
print("Shape:",df.shape) #rows,columns
print(df.info())           #column name + types + non-null count
print(df.dtypes)         #only for checking types

#statistical summary
print(df.describe())

#checking missing values
print(df.isnull().sum())

# Histogram
df.hist(figsize=(10,6), bins=20, color='steelblue', edgecolor='white')
plt.suptitle('Distribution of all features')
plt.tight_layout()
plt.show()

# Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

# Scatter plot
plt.figure(figsize=(8,6))
sns.scatterplot(x='area_sqft', y='price', data=df)
plt.title('Area vs Price')
plt.tight_layout()
plt.show()

#Encoding 'garage' yes/no-->> 0/1
print("Before Encoding")
print(df['garage'].value_counts())

df['garage']=df['garage'].map({'yes':1,'no':0})

print("After Encoding")
print(df['garage'].value_counts())

#Feature-Target Separation 
X = df[['area_sqft', 'bedrooms', 'bathrooms', 'age_years', 'garage', 'location_score']]
y = df['price']          
print("X shape",X.shape)
print("y shape",y.shape)

print("Features",X.columns.tolist())

#Train Test Split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

print("TRAINING SET:",X_train.shape)
print("TESTING SET:",X_test.shape)

#Feature Scaling
Scaler_OBJ=StandardScaler()
X_train=Scaler_OBJ.fit_transform(X_train) #On train fit and transform both
X_test=Scaler_OBJ.transform(X_test)       #On test only transform

lr_obj=LinearRegression()
lr_obj.fit(X_train,y_train)    #model training
lr_pred=lr_obj.predict(X_test) #predict on test data

dt_obj=DecisionTreeRegressor(max_depth=5,random_state=42)
dt_obj.fit(X_train,y_train)
dt_pred=dt_obj.predict(X_test)

print("\n MODEL TRAIN SUCCESSFULLY! ")
print(f"LR first 5 predictions:{lr_pred[:5].round(0)} ")
print(f"DT first 5 predictions:{dt_pred[:5].round(0)}")
print(f"Actual first 5 values:{y_test.values[:5]}")

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Evaluation Function
def evaluate(y_test, y_pred):
    print("MAE :", mean_absolute_error(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("R2 :", r2_score(y_test, y_pred))

# Linear Regression result
print("Linear Regression")
evaluate(y_test, lr_pred)

# Decision Tree result
print("\nDecision Tree")
evaluate(y_test, dt_pred)

#Actual VS Predicted
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test, lr_pred, alpha=0.6, color='steelblue',s=60)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Linear Regression')

plt.subplot(1, 2, 2)
plt.scatter(y_test, dt_pred, alpha=0.6, color='seagreen',s=60)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Decision Tree')

formatter = FuncFormatter(lambda x, pos: f'{x/1000:.0f}K')

plt.subplot(1,2,1)
plt.gca().xaxis.set_major_formatter(formatter)
plt.gca().yaxis.set_major_formatter(formatter)

plt.subplot(1,2,2)
plt.gca().xaxis.set_major_formatter(formatter)
plt.gca().yaxis.set_major_formatter(formatter)
plt.tight_layout()
plt.show()









