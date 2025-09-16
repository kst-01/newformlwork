import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
iris_data = datasets.load_iris()

df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)
df['target'] = iris_data.target
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df[iris_data.feature_names], df['target'], test_size=0.2, random_state=42)
print(X_train,"\n this is y_test \n",y_test)
scaler = MinMaxScaler()
scaler.fit(X_train)
Xtrain_scaled = scaler.transform(X_train)
Xtest_scaled = scaler.transform(X_test)
