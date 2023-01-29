import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

le_fit_buying = pickle.load(open('data\\vehicle_evaluation\\le_fit_buying','rb'))
le_fit_evaluation = pickle.load(open('data\\vehicle_evaluation\\le_fit_evaluation','rb'))
le_fit_doors = pickle.load(open('data\\vehicle_evaluation\\le_fit_doors','rb'))
le_fit_lug_boot = pickle.load(open('data\\vehicle_evaluation\\le_fit_lug_boot','rb'))
le_fit_persons = pickle.load(open('data\\vehicle_evaluation\\le_fit_persons','rb'))
le_fit_maint = pickle.load(open('data\\vehicle_evaluation\\le_fit_maint','rb'))
le_fit_safety = pickle.load(open('data\\vehicle_evaluation\\le_fit_safety','rb'))

# valuesDF = pd.DataFrame(values,columns=["buying","maint","doors","persons","lug_boot","safety"])

# valuesDF["buying"] = le_fit_buying.transform(valuesDF["buying"])
# valuesDF["maint"] = le_fit_maint.transform(valuesDF["maint"])
# valuesDF["doors"] = le_fit_doors.transform(valuesDF["doors"])
# valuesDF["persons"] = le_fit_persons.transform(valuesDF["persons"])
# valuesDF["lug_boot"] = le_fit_lug_boot.transform(valuesDF["lug_boot"])
# valuesDF["safety"] = le_fit_safety.transform(valuesDF["safety"])

clf = DecisionTreeClassifier()
gnb = pickle.load(open('data\\vehicle_evaluation\\decisionTreeClassifier\\model','rb'))

# y_pred = gnb.predict(valuesDF)

# actual_value = le_fit_evaluation.inverse_transform(y_pred)

def getModelOutput(inputArray):
    valuesDF = pd.DataFrame(inputArray,columns=["buying","maint","doors","persons","lug_boot","safety"])
    valuesDF["buying"] = le_fit_buying.transform(valuesDF["buying"])
    valuesDF["maint"] = le_fit_maint.transform(valuesDF["maint"])
    valuesDF["doors"] = le_fit_doors.transform(valuesDF["doors"])
    valuesDF["persons"] = le_fit_persons.transform(valuesDF["persons"])
    valuesDF["lug_boot"] = le_fit_lug_boot.transform(valuesDF["lug_boot"])
    valuesDF["safety"] = le_fit_safety.transform(valuesDF["safety"])
    gnb.predict(valuesDF)
    y_pred = gnb.predict(valuesDF)
    actual_value = le_fit_evaluation.inverse_transform(y_pred)
    return actual_value

# values = [["high","high","4","2","med","high"]]
values = []
while True:
    user_input = input("Enter '1' to test another (or type '0' to quit): ")
    if user_input == "0":
        break
    elif user_input == "1":
        print("You entered: ", user_input)
        testValues = []
        for i in range(6):
            value = input("Enter string value: ")
            testValues.append(value)
            print(testValues)
        testValuesWrapper = [testValues]
        resultValue = getModelOutput(testValuesWrapper)
        print(resultValue)