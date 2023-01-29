from sklearn.naive_bayes import GaussianNB
import pickle
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

X_train = pickle.load(open('data\\vehicle_evaluation\\vehicle_evaluation_X_train','rb'))
X_test = pickle.load(open('data\\vehicle_evaluation\\vehicle_evaluation_X_test','rb'))
y_train = pickle.load(open('data\\vehicle_evaluation\\vehicle_evaluation_y_train','rb'))
y_test = pickle.load(open('data\\vehicle_evaluation\\vehicle_evaluation_y_test','rb'))


clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

pickle.dump(clf, open('data\\vehicle_evaluation\\decisionTreeClassifier\\model','wb'))

y_pred = clf.predict(X_test)

accuracy = metrics.accuracy_score(y_test, y_pred)
print(accuracy)
precision = metrics.precision_score(y_test,y_pred, average='micro', zero_division='warn')
print(precision)
recall = metrics.recall_score(y_test,y_pred,average='micro')
print(recall)
f1_score = metrics.f1_score(y_test,y_pred, average='micro')
print(f1_score)

# low,low,4,4,big,med,good
# low,low,4,4,big,high,vgood
# low,low,4,more,small,low,unacc
# low,low,4,more,small,med,acc
# low,low,4,more,small,high,good
# low,low,4,more,med,low,unacc
# low,low,4,more,med,med,good
# low,low,4,more,med,high,vgood