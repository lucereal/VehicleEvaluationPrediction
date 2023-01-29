from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pickle
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

# create an instance of the StandardScaler
# scaler = StandardScaler()

# # fit the scaler to your data
# scaler.fit(X_train)

# # use the scaler to transform the data
# X_train_scaled = scaler.transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# scaler.fit(y_train.reshape(-1,1))
# y_train_scaled = scaler.transform(y_train.reshape(-1,1))
# y_test_scaled = scaler.transform(y_test.reshape(-1,1))

# y_train_scaled = y_train_scaled.reshape(-1)
# y_test_scaled = y_test_scaled.reshape(-1)

# print(X_train_scaled)
# print(X_test_scaled)
# print(y_train_scaled)
# print(y_test_scaled)
# Create an instance of the LogisticRegression class
clf = LogisticRegression()
#clf = LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', multi_class='auto')

# Fit the model to your training data
le = LabelEncoder()

# y_train_scaled = le.fit_transform(y_train_scaled)
# y_test_scaled = le.fit_transform(y_test_scaled)
# print(y_train_scaled)
# print(y_test_scaled)
clf.fit(X_train, y_train)

# Use the model to make predictions on your test data
y_pred = clf.predict(X_test)

accuracy = metrics.accuracy_score(y_test, y_pred)
print(accuracy)
precision = metrics.precision_score(y_test,y_pred, average='micro', zero_division='warn')
print(precision)
recall = metrics.recall_score(y_test,y_pred,average='micro')
print(recall)
f1_score = metrics.f1_score(y_test,y_pred, average='micro')
print(f1_score)


