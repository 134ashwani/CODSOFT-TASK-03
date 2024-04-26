import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("advertising.csv")
print(df.head())

print(df.shape)
print(df.describe)

sns.pairplot(df, x_vars=['TV', 'Radio','Newspaper'], y_vars='Sales', kind='scatter')
plt.show()
df['TV'].plot.hist(bins=10)
plt.show()

df['Radio'].plot.hist(bins=10, color="green", xlabel="Radio")
plt.show()

df['Newspaper'].plot.hist(bins=10, color="purple", xlabel="newspaper")
plt.show()

sns.heatmap(df.corr(),annot = True)
plt.show()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df[['TV']], df[['Sales']], test_size=0.3, random_state=0)
print(X_train)
print(y_train)
print(X_test)
print(y_test)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)

LinearRegression()

res= model.predict(X_test)
print(res)
print(model.coef_)
print(model.intercept_)
print(0.05473199* 69.2 + 7.14382225)
plt.plot(res)
plt.show()



plt.scatter(X_test, y_test)
plt.plot(X_test, 7.14382225 + 0.05473199 * X_test, 'r')
plt.show()