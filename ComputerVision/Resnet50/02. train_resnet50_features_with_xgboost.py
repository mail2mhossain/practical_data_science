import numpy as np
from datetime import datetime
from matplotlib import pyplot
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report,confusion_matrix,plot_confusion_matrix

def get_RandomizedSearch_model(parameters, X_train, y_train):
    model = XGBClassifier(objective='binary:logistic')
    xgb_grid = RandomizedSearchCV(model, 
                                parameters,
                                scoring='f1_micro',
                                cv = 2,
                                n_jobs = 10,
                                verbose=False)
    xgb_grid.fit(X_train, y_train)
    return xgb_grid

init_time = datetime.now()
#X_train = np.load('resnet50_dogs_vs_cats_train_features.npy')
#y_train = np.load('resnet50_dogs_vs_cats_train_labels.npy')
#np.savez_compressed('resnet50_dogs_vs_cats_train_data.npz', features, labels)
# load the face dataset
data = np.load('resnet50_dogs_vs_cats_train_data.npz')
X_train, y_train = data['arr_0'], data['arr_1']

#X_test = np.load('resnet50_dogs_vs_cats_test_features.npy')
#y_test = np.load('resnet50_dogs_vs_cats_test_labels.npy')
data = np.load('resnet50_dogs_vs_cats_test_data.npz')
X_test, y_test = data['arr_0'], data['arr_1']

#print(X_train.shape, y_train.shape)
#print(y_train)

fin_time = datetime.now()
print("Data Loading time : ", (fin_time-init_time))

init_time = datetime.now()
clf_xgb = XGBClassifier()
clf_xgb.fit(X_train, y_train, verbose=False, 
            early_stopping_rounds=10, 
            eval_set=[(X_test, y_test)])
fin_time = datetime.now()
print("Execution time : ", (fin_time-init_time))
print(classification_report(y_test, clf_xgb.predict(X_test)))

# Test Result 98%

#init_time = datetime.now()
#results = dict()
#models = dict()
#for _ in range(10):
    #grid_model = get_RandomizedSearch_model(parameters, X_train, y_train)
    #grid_predictions = grid_model.predict(X_test)
    #score = accuracy_score(y_test, grid_predictions)
    #results[score] = grid_model.best_params_
    #models[score] = grid_model.best_estimator_
#fin_time = datetime.now()
#print("Grid Search Execution time : ", (fin_time-init_time))

#max_score = max(results.keys())
#print(f"Minimum Error: {max_score}")
#print(f"Parameters: {results[max_score]}")

# save model
f = open('resnet50_xgb_best_model', "wb")
f.write(pickle.dumps(clf_xgb))
f.close()

#load model
model = pickle.loads(open('resnet50_xgb_best_model', "rb").read())
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))
plot_confusion_matrix(model, X_test, y_test, values_format='d', display_labels=["Dog","Cat"])
pyplot.savefig("resnet50_confusion_matrix_dogs_cats_plot.png")
# show the figure
pyplot.show()