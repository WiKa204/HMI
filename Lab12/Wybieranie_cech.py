import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from scipy.cluster.hierarchy import dendrogram, linkage, distance, fcluster


path ='datasets'

data_train= pd.read_hdf(f'{path}/train_03_2018-06-14_2_All')
data_test= pd.read_hdf(f'{path}/test_03_2018-06-14_2_All')
columns = list(data_train.filter(regex='input').columns)

X_train = data_test[columns]
y_train = data_test['output_0']
X_test = data_test[columns]
y_test = data_test['output_0']


clf = RandomForestClassifier(random_state=100)
clf_full = clf.fit(X_train,y_train)
preds = clf.predict(X_test)

print(precision_score(preds, y_test, average='macro'))
print(confusion_matrix(y_test, preds))

top_n=20 #number of top features to show
tree_feature_importances = clf.feature_importances_
sorted_idx = tree_feature_importances.argsort()
sorted_idx = sorted_idx[-top_n:]
y_ticks = np.arange(0, len(sorted_idx))
cols_ord =  [columns[i] for i in sorted_idx]
fig, ax = plt.subplots()
ax.barh(y_ticks, tree_feature_importances[sorted_idx])
ax.set_yticks(y_ticks)
ax.set_yticklabels(cols_ord)
ax.set_title("Random Forest Feature Importances")
fig.tight_layout()
plt.show()

result = permutation_importance(clf, X_test, y_test, n_repeats=10,
                                random_state=0, n_jobs=2)
sorted_idx = result.importances_mean.argsort()[-30:]

fig, ax = plt.subplots()
ax.boxplot(result.importances[sorted_idx].T, vert=False, labels=X_test.columns[sorted_idx])
ax.set_title("Permutation Importances (test set)")
fig.tight_layout()
plt.show()

df = data_train.filter(regex='input.*12$')
corr = df.corr().values

pdist = distance.pdist(np.abs(corr))
Z = linkage(pdist, method='ward')

labelList = list(df.columns)
plt.figure(figsize=(15, 12))
dendrogram(
            Z,
            orientation='right',
            labels=labelList,
            distance_sort='descending',
            show_leaf_counts=False
          )

plt.show()
th=2.0
plt.axvline(x=th, c='grey', lw=1, linestyle='dashed')

idx = fcluster(Z, th, 'distance')
v = [[id, v.split('_')[2]] for id, v in zip(idx, df.columns)]
selected_features = pd.DataFrame(v, columns=['cluster', 'feature']).groupby('cluster').first()

#generacja cech dla wszystkich kanałów
columns = list([])
for feature in selected_features['feature']:
    col_f = data_train.filter(regex=f'input_\d+_{feature}_').columns
    columns.extend(col_f)