import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


data = pd.read_csv("..\data\carDataWHeader.data", header='infer')

le_fit_evaluation = LabelEncoder()
le_fit_evaluation = le_fit_evaluation.fit(data['evaluation'])
pickle.dump(le_fit_evaluation, open('data\\vehicle_evaluation\\le_fit_evaluation','wb'))

le_fit_buying = LabelEncoder()
le_fit_buying = le_fit_buying.fit(data['buying'])
pickle.dump(le_fit_buying, open('data\\vehicle_evaluation\\le_fit_buying','wb'))

le_fit_maint = LabelEncoder()
le_fit_maint = le_fit_maint.fit(data['maint'])
pickle.dump(le_fit_maint, open('data\\vehicle_evaluation\\le_fit_maint','wb'))

le_fit_doors = LabelEncoder()
le_fit_doors = le_fit_doors.fit(data['doors'])
pickle.dump(le_fit_doors, open('data\\vehicle_evaluation\\le_fit_doors','wb'))

le_fit_persons = LabelEncoder()
le_fit_persons = le_fit_persons.fit(data['persons'])
pickle.dump(le_fit_persons, open('data\\vehicle_evaluation\\le_fit_persons','wb'))

le_fit_lug_boot = LabelEncoder()
le_fit_lug_boot = le_fit_lug_boot.fit(data['lug_boot'])
pickle.dump(le_fit_lug_boot, open('data\\vehicle_evaluation\\le_fit_lug_boot','wb'))

le_fit_safety = LabelEncoder()
le_fit_safety = le_fit_safety.fit(data['safety'])
pickle.dump(le_fit_safety, open('data\\vehicle_evaluation\\le_fit_safety','wb'))


evaluation_column = data['evaluation']

data = data.drop(columns=['evaluation'], inplace=False);


evaluation_column = le_fit_evaluation.transform(evaluation_column)
# evaluation_column_unique = list(dict.fromkeys(evaluation_column))
# print("evaluation_column_unique")
# print(evaluation_column_unique)

data['buying'] = le_fit_buying.transform(data['buying'])
data['maint'] = le_fit_maint.transform(data['maint'])
data['doors'] = le_fit_doors.transform(data['doors'])
data['persons'] = le_fit_persons.transform(data['persons'])
data['lug_boot'] = le_fit_lug_boot.transform(data['lug_boot'])
data['safety'] = le_fit_safety.transform(data['safety'])

print("data and evaluation after transform")
print(data)
print(evaluation_column)
X = data
y = evaluation_column


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('foo')
print(X_train)
print(X_test)
print(y_train)
print(y_test)

pickle.dump(X_train, open('data\\vehicle_evaluation\\vehicle_evaluation_X_train','wb'))
pickle.dump(X_test, open('data\\vehicle_evaluation\\vehicle_evaluation_X_test','wb'))
pickle.dump(y_train, open('data\\vehicle_evaluation\\vehicle_evaluation_y_train','wb'))
pickle.dump(y_test, open('data\\vehicle_evaluation\\vehicle_evaluation_y_test','wb'))


