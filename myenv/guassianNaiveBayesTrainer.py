from sklearn.naive_bayes import GaussianNB
import pickle
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


X_train = pickle.load(open('data\\vehicle_evaluation\\vehicle_evaluation_X_train','rb'))
X_test = pickle.load(open('data\\vehicle_evaluation\\vehicle_evaluation_X_test','rb'))
y_train = pickle.load(open('data\\vehicle_evaluation\\vehicle_evaluation_y_train','rb'))
y_test = pickle.load(open('data\\vehicle_evaluation\\vehicle_evaluation_y_test','rb'))

print(X_train)
print(X_test)
print(y_train)
print(y_test)




X_train_scaled = X_train
X_test_scaled = X_test
y_train_scaled = y_train
y_test_scaled = y_test

gnb = GaussianNB()
gnb.fit(X_train_scaled, y_train_scaled)

y_pred = gnb.predict(X_test_scaled)

print(y_pred)
accuracy = metrics.accuracy_score(y_test_scaled, y_pred)
print(accuracy)
precision = metrics.precision_score(y_test_scaled,y_pred, average='micro', zero_division='warn')
print(precision)
recall = metrics.recall_score(y_test_scaled,y_pred,average='micro')
print(recall)
f1_score = metrics.f1_score(y_test_scaled,y_pred, average='micro')
print(f1_score)
