import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

df_train = pd.read_csv('train.csv')
df_train=df_train[~np.isnan(df_train).any(axis=1)]

df_test = pd.read_csv('test.csv')

x_train = df_train['x']
y_train = df_train['y']
x_test = df_test['x']
y_test = df_test['y']

x_train = np.array(x_train)

y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

x_train = x_train.reshape(-1,1)
x_test = x_test.reshape(-1,1)
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score

clf = LinearRegression()
clf.fit(x_train,y_train)
y_pred = clf.predict(np.array(sorted(x_test)))
print(r2_score(y_test,y_pred))
plt.figure(figsize=(10,10))
plt.plot(y_pred)
plt.plot(np.array(sorted(y_test)))
plt.legend()
plt.show()





import numpy as np

n = 699
alpha = 0.00001
b=0
m=0
epochs = 0

while(epochs < 10000):
    y = b + m * x_train
    error = y - y_train
    mean_sq_er = np.sum(error**2)
    mean_sq_er = mean_sq_er/n
    b = b - alpha * 2 * np.sum(error)/n 
    m = m - alpha * 2 * np.sum(error * x_train)/n
    epochs += 1

        
         
     
import matplotlib.pyplot as plt 

y_prediction = b + m * x_test
print('R2 Score:',r2_score(y_test,y_prediction))

Numero=[]
y_plot = []
y_Mod_plot= []
for i in range(100):
    Numero.append(i)
    
Numero=np.array(Numero)
y_plot_Manu=(b + m * Numero)
y_plot_Model= clf.predict(Numero.reshape(-1, 1))
plt.figure(figsize=(10,10))
plt.scatter(x_test,y_test)
plt.plot(range(len(y_plot_Manu)),y_plot_Manu,color='black',label = 'pred_manu')
plt.plot(range(len(y_plot_Model)),y_plot_Model,color='red',label = 'pred_func')

plt.legend()
plt.show()


