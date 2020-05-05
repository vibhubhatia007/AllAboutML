
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:,:1]
y = dataset.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit( x_train, y_train)

y_pred = reg.predict(x_test)

plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,reg.predict(x_train),color='blue')
plt.title('Rnd vs Profit')
plt.xlabel('Rnd')
plt.ylabel("Profit")
plt.show()

plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,reg.predict(x_train),color='blue')
plt.title('Rnd vs Profit')
plt.xlabel('Rnd')
plt.ylabel("Profit")
plt.show()

