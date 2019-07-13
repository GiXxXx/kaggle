import pandas as pd
import numpy as np

from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from mlxtend.classifier import StackingClassifier

train_data = pd.read_csv("~/kaggle-workspace/kaggle/digital-recognizer/train.csv")
test_data = pd.read_csv("~/kaggle-workspace/kaggle/digital-recognizer/test.csv")

pca=PCA(n_components=100, random_state=34)
train_data_pca=pca.fit_transform(train_data.values[:,1:])
test_data_pca=pca.fit_transform(train_data.values)

# lm_clf = SVC()
# lm_clf.fit(train_data_pca,train_data.values[:, 0])
# ret = lm_clf.predict(test_data_pca)
# print("LALALALALAL")

# lm_clf = BaggingClassifier(
#     LogisticRegression(), max_features=1.0, max_samples=0.2)

knn_clf = BaggingClassifier(
    KNeighborsClassifier(n_jobs=-1), n_estimators=3, max_features=1.0, max_samples=0.3, n_jobs=-1)

# nb_clf = BaggingClassifier(
#     GaussianNB(), n_estimators=3, max_features=1.0, max_samples=0.3, n_jobs=-1)

# svc_clf  = BaggingClassifier(
#     SVC(), n_estimators=3, max_features=1.0, max_samples=0.3, n_jobs=-1)

rf_clf = RandomForestClassifier(n_estimators=110, max_depth=5,
                                                  min_samples_split=2, min_samples_leaf=1, random_state=34, n_jobs=-1)

st_clf = StackingClassifier(classifiers=[knn_clf, rf_clf], meta_classifier=LogisticRegression(), use_probas=True, average_probas=False)


print('3-fold cross validation:\n')

for clf, label in zip([knn_clf, rf_clf, st_clf],
                      [
                       'KNN',
                       'Random Forest',
                       'StackingClassifier']):

    scores = model_selection.cross_val_score(
        clf, train_data_pca, train_data.values[:, 0], cv=3, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

# ret = bagging_clf.predict(test_data.values)

# result = pd.DataFrame(data=test_data.index, index=None, columns=['ImageId'])
# result['Label'] = ret
# result.ImageId = result.ImageId + 1

# result.to_csv('simple-stacking.csv', index=False)
