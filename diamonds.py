import pandas as pd
df=pd.read_csv("diamonds.csv", index_col=0)
#df.set_index("carat", inplace=True)
df.head()
# to convert into numerical form 
df["cut"].unique()
#df["cut"].astype("category").cat.codes
df["clarity"].unique()
df["color"].unique()
# to assign numbers to string values

cut_class_dict={'Fair':1, 'Good':2, 'Very Good':3, 'Premium':4, 'Ideal':5}
clarity_dict={'FL':11,'IF':10, 'VVS1':9, 'VVS2':8, 'VS1':7, 'VS2':6, 'SI1':5, 'SI2':4, 'I1':3, 'I2':2, 'I3':1}
color_dict={'J':1, 'I':2, 'H':3, 'G':4, 'F':5, 'E':6, 'D':7}

df['cut']=df['cut'].map(cut_class_dict)
df['clarity']=df['clarity'].map(clarity_dict)
df['color']=df['color'].map(color_dict)

df.head()
import sklearn
from sklearn import svm,preprocessing

# shuffle the df for better results

df = sklearn.utils.shuffle(df)
X=df.drop("price", axis=1).values

# scale the data to reduce the complexity and huf=ge range
X=preprocessing.scale(X)

y=df["price"].values
test_size = 200

X_train= X[:-test_size]
y_train= y[:-test_size]

X_test= X[-test_size:]
y_test= y[-test_size:]

clf=svm.SVR(kernel="linear")
clf.fit(X_train,y_train)
clf.score(X_test, y_test)
for X,y in zip(X_test, y_test):
    print(f"Model:{clf.predict([X])[0]}, Actual:{y}")

# check using radial basis function

clf=svm.SVR(kernel="rbf")
clf.fit(X_train,y_train)
print(clf.score(X_test, y_test))

for X,y in zip(X_test, y_test):
    print(f"Model:{clf.predict([X])[0]}, Actual:{y}")