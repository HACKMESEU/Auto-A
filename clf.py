import numpy as np
import numpy as np
import preprocess as pp
# Use leave-one-out method to process cross validation
from sklearn.model_selection import LeaveOneOut, LeavePOut
from sklearn.model_selection import cross_val_predict

# Import the origin data from excel files
xx = 'C:\\Users\\Sun Weihang\\Desktop\\data1121\\星形\\20171206150826para.xlsx'
sm = 'C:\\Users\\Sun Weihang\\Desktop\\data1121\\髓母\\20171204130028para.xlsx'
sg = 'C:\\Users\\Sun Weihang\\Desktop\\data1121\\室管膜瘤\\20171206160229para.xlsx'

xx = 'C:\\Users\\Sun Weihang\\Desktop\\data1121\\星形\\20171206192202para.xlsx'
sm = 'C:\\Users\\Sun Weihang\\Desktop\\data1121\\髓母\\20171206192407para.xlsx'
#sg = 'C:\\Users\\Sun Weihang\\Desktop\\data1121\\室管膜瘤\\20171206194005para.xlsx'
sg = 'C:\\Users\\Sun Weihang\\Desktop\\data1121\\室管膜瘤\\20171208154848para.xlsx'

XX = pp.loadExcel(xx)
XX_y = 0*np.ones(XX.shape[0])
SM = pp.loadExcel(sm)
SM_y = 1*np.ones(SM.shape[0])
SG = pp.loadExcel(sg)
SG_y = 2*np.ones(SG.shape[0])
X_data = np.row_stack((XX,SM,SG))
X_data = np.delete(X_data,(16,33,50),axis=1)
#X_data = X_data[:23,:]
X_origin = X_data
#X_data = X_data[:,(9,10)]
Y_data = np.append(np.append(XX_y,SM_y),SG_y)
#Y_data = Y_data[:23]

# Feature selection using random forest
from sklearn.ensemble import ExtraTreesClassifier
# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=40,
                              random_state=0)
import numpy as np
import matplotlib.pyplot as plt
X = X_data
y =Y_data
forest.fit(X_data, Y_data)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
# Print the feature ranking
print("Feature ranking:")
for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
#plt.bar(range(X.shape[1]), importances[indices],
#       color="r", yerr=std[indices], align="center")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()


# Training process.
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm


def crossvalidation( X_origin, Y_data,models):

    C = 1
    loo = LeaveOneOut()
    lpo = LeavePOut(p=2)
    for clf in models:
        #scores = cross_validate(clf, X_origin, Y_data, cv=loo, return_train_score=True)
        predictions = cross_val_predict(clf, X_origin, Y_data, cv=loo)

        print(clf)
        # print(Y_data)
        # print(predictions)
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(Y_data, predictions).ravel()
        Sen0 = cm[0] / (cm[0:3].sum())
        Spe0 = (cm[4] + cm[5] + cm[7] + cm[8]) / (cm[3:9].sum())
        Acc0 = (cm[0] + cm[4] + cm[5] + cm[7] + cm[8]) / (cm.sum())
        print('class0:', 'Sensitivity:', Sen0)
        print('class0:', 'Specificity:', Spe0)
        print('class0:', 'Accuracy:', Acc0)
        Sen1 = cm[4] / (cm[3:6].sum())
        Spe1 = (cm[0] + cm[2] + cm[6] + cm[8]) / (cm[0:3].sum() + cm[6:9].sum())
        Acc1 = (cm[4] + cm[0] + cm[2] + cm[6] + cm[8]) / (cm.sum())
        print('class1:', 'Sensitivity:', Sen1)
        print('class1:', 'Specificity:', Spe1)
        print('class1:', 'Accuracy:', Acc1)
        Sen2 = cm[8] / (cm[6:9].sum())
        Spe2 = (cm[0] + cm[1] + cm[3] + cm[4]) / (cm[0:3].sum() + cm[3:6].sum())
        Acc2 = (cm[8] + cm[0] + cm[1] + cm[3] + cm[4]) / (cm.sum())
        print('class2:', 'Sensitivity:', Sen2)
        print('class2:', 'Specificity:', Spe2)
        print('class2:', 'Accuracy:', Acc2)

models = (svm.SVC(kernel='rbf', C=1000, gamma=0.1, decision_function_shape='ovo'),
              # svm.SVC(kernel='linear', C=C),
              # MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1),
              forest
              )


#----------- All features
crossvalidation(X_origin,Y_data,models)
from sklearn.tree import export_graphviz

i_tree = 0
import graphviz

for tree_in_forest in forest.estimators_:
    if(i_tree == 1):
        #my_file = StringIO()
        my_file = export_graphviz(tree_in_forest,impurity=False, out_file = None)
        graph = graphviz.Source(my_file)
        graph.view()
        #png_str = pydotplus.graph_from_dot_data(my_file)

        #Image(graph.create_png())
        #plt.figure()
        #plt.imshow(graph)
        #plt.show()
        #graph.view()


    i_tree = i_tree + 1


#----------- After features selection
# #X_origin = X_data[:,(8,9,7)]
X_origin = X_data[:,(8,7,22)]
crossvalidation(X_origin,Y_data,models)

#----------- feature reduction using PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
#X_data = X_data[:]
X_r= pca.fit(X_data).transform(X_data)
print(pca.explained_variance_ratio_)
X_r  = X_r / np.linalg.norm(X_r)
target_names = ['0','1','2']
plt.figure()
colors = ['navy', 'turquoise', 'darkorange']
lw = 2
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of  dataset')
plt.show()
X_origin = X_r
X_origin  = X_origin / np.linalg.norm(X_origin)
crossvalidation(X_origin,Y_data,models)





#X_data =X_r
def make_meshgrid(x, y, h=0.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 100, x.max() + 100
    y_min, y_max = y.min() - 100, y.max() + 100
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


# import some data to play with
#iris = datasets.load_iris()
# Take the first two features. We could avoid this by using a two-dim dataset
X = X_data[:,(8,9)]
y = Y_data


 #we create an instance of SVM and fit out data. We do not scale our
 #data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter



models = (svm.SVC(kernel='rbf',C=1000,gamma=0.1,decision_function_shape='ovo'),
          #svm.SVC(kernel='linear', C=C),
          #MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1),
          forest
          )
models = (clf.fit(X, y) for clf in models)

# title for the plots
titles = ('SVC with rbf kernel',
          'Random Forest',

          'Random Forest')

# Set-up 2x2 grid for plotting.
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1,5)

for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Feature 0')
    ax.set_ylabel('Feature 1')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()




# Using leave-one-out method to cross validation
#X_origin = X_origin[:,(8,9,7)]
#X_origin = X_r
#loo = LeaveOneOut()
#lpo = LeavePOut(p=2)
#models = (svm.SVC(kernel='rbf',C=1.0,gamma=0.5),
#          svm.SVC(kernel='linear', C=C),
#          forest)
#for clf in models:
#    scores = cross_validate(clf, X_origin, Y_data, cv=loo, return_train_score=True)
#    print(Y_data)
#    print(scores['test_score'])
#    print(clf,np.mean(scores['test_score']))







#from sklearn.feature_selection import VarianceThreshold
#sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
#X = sel.fit_transform(X_data)
#from sklearn.feature_selection import SelectKBest
#from sklearn.feature_selection import chi2
#X_new = SelectKBest(chi2, k=2).fit_transform(X, Y_data)


#from sklearn.decomposition import PCA
#pca = PCA(n_components = 2 )
#X_data = pca.fit(X_data).transform(X_data)
#from sklearn import datasets
#iris = datasets.load_iris()
#X_data = iris.data
#Y_data = iris.target
# Choose
#clf = svm.SVC(decision_function_shape='ovo').fit(X_data,Y_data)
#train_accuracy = clf.score(X_data,Y_data)
#print(train_accuracy)



#clf = svm.SVC(decision_function_shape='ovo',kernel='linear')



