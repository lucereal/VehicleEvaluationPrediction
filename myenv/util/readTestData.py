import pickle


X_train = pickle.load(open('data\\vehicle_evaluation\\vehicle_evaluation_X_train','rb'))
X_test = pickle.load(open('data\\vehicle_evaluation\\vehicle_evaluation_X_test','rb'))
y_train = pickle.load(open('data\\vehicle_evaluation\\vehicle_evaluation_y_train','rb'))
y_test = pickle.load(open('data\\vehicle_evaluation\\vehicle_evaluation_y_test','rb'))

print("X_train")
print(X_train)
print("X_test")
print(X_test)

print("y_train")
print(y_train)
print("y_test")
print(y_test)

le_fit_buying = pickle.load(open('data\\vehicle_evaluation\\le_fit_buying','rb'))
le_fit_evaluation = pickle.load(open('data\\vehicle_evaluation\\le_fit_evaluation','rb'))
le_fit_doors = pickle.load(open('data\\vehicle_evaluation\\le_fit_doors','rb'))
le_fit_lug_boot = pickle.load(open('data\\vehicle_evaluation\\le_fit_lug_boot','rb'))
le_fit_persons = pickle.load(open('data\\vehicle_evaluation\\le_fit_persons','rb'))
le_fit_maint = pickle.load(open('data\\vehicle_evaluation\\le_fit_maint','rb'))
le_fit_safety = pickle.load(open('data\\vehicle_evaluation\\le_fit_safety','rb'))

X_test["buying"] = le_fit_buying.inverse_transform(X_test["buying"])
X_test["maint"] = le_fit_maint.inverse_transform(X_test["maint"])
X_test["doors"] = le_fit_doors.inverse_transform(X_test["doors"])
X_test["persons"] = le_fit_persons.inverse_transform(X_test["persons"])
X_test["lug_boot"] = le_fit_lug_boot.inverse_transform(X_test["lug_boot"])
X_test["safety"] = le_fit_safety.inverse_transform(X_test["safety"])

y_test = le_fit_evaluation.inverse_transform(y_test)

testData = X_test
testData['evaluation'] = y_test
# X_test and y_test
print("testData")
print(testData.head(10))

