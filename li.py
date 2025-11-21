import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv(r"D:\New folder\salary_Data.csv")
x=data.iloc[:,:-1].values
y=data.iloc[:,4].values
from sklearn.model_selection import train_test_split
x_tr,x_ts,y_tr,y_ts=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x,y)
y_pred=regressor.predict(x_tr)
x_pred=regressor.predict(x_ts)
plt.scatter(x_tr,y_tr,color='red')
plt.plot(x_tr,y_pred,color="blue")
plt.xlabel('sal')
plt.ylabel('exp')
plt.title('sal vs exp(tr ds)')
plt.show()
