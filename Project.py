import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report




data = {
    "Math": [78, 85, 90, 60, 72, 88, 95, 55, 67, 80],
    "Science": [82, 80, 88, 65, 70, 90, 92, 60, 75, 85],
    "English": [75, 78, 85, 58, 68, 86, 90, 55, 70, 80],
    "Attendance": [85, 90, 95, 70, 80, 92, 98, 65, 75, 88]
}
df=pd.DataFrame(data)
df['Average'] = df[["Math","Science","English"]].mean(axis=1)

def categorize(score):
    if score>=85:
        return 'Good'
    elif score>=70:
        return 'Average'
    else:
        return 'Poor'
df["Performance"]=df["Average"].apply(categorize)
x=df[["Math","Science","English","Attendance"]]
y=df["Performance"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=LogisticRegression(max_iter=2000)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print('Accuracy:',accuracy_score(y_test,y_pred))

print("Classification Report:")
print(classification_report(y_test, y_pred))
new_student = pd.DataFrame([[85, 88, 82, 90]],columns=["Math","Science","English","Attendance"])
prediction=model.predict(new_student)
print("prediction of new student is :",prediction[0])
