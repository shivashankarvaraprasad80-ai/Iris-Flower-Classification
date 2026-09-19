import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
iris = load_iris()
df=pd.DataFrame(iris.data,columns=iris.feature_names)
df["species"]=iris.target
print(df.head())

print("Dataset shapes:")
print(df.shape)
print("Dataset Information:")
print(df.info())
print("Missing values:")
print(df.isnull().sum())

sns.scatterplot(
    x="sepal length (cm)",
    y="sepal width (cm)",
    hue="species",
    data=df
)
plt.title("sepal length vs sepal width")
plt.show()

x=iris.data
y=iris.target

print("x Shape:",x.shape)
print("y Shape:",y.shape)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print("The Training data:",x_train.shape)
print("The Testing data:",x_test.shape)

scaler=StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model=KNeighborsClassifier(n_neighbors=5)
model.fit(x_train,y_train)

y_pred = model.predict(x_test)
print("predicted values:",y_pred)

accuracy = accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy*100,"%")

cm = confusion_matrix(y_test,y_pred)
print("confusion matrix:",cm)

plt.imshow(cm)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion matrix")


plt.xticks(
    range(3),
    iris.target_names
)


plt.xticks(
    range(3),
    iris.target_names
)
for i in range(3):
    for j in range(3):
     plt.text(j,i,cm[i,j],ha="center",va="center")
        
plt.show()        

print("Enter the Flower Measurements:")

sepal_length=float(input("Sepal Length (cm):"))
sepal_width=float(input("Sepal Width (cm):"))
petal_length=float(input("Petal Length (cm):"))
petal_width=float(input("Petal Width (cm):"))

new_flower=[[sepal_length,sepal_width,petal_length,petal_width]]

new_flower_scaled = scaler.transform(new_flower)

prediction = model.predict(new_flower_scaled)

species=iris.target_names[prediction[0]]

print("\nPredicted Species:",species)