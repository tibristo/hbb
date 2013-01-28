from numpy import *
from root_numpy import *


def shuffle_in_unison(a, b, c):
    assert len(a) == len(b)
    shuffled_a = empty(a.shape, dtype=a.dtype)
    shuffled_b = empty(b.shape, dtype=b.dtype)
    shuffled_c = empty(c.shape, dtype=c.dtype)
    permutation = random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
        shuffled_c[new_index] = c[old_index]
    return shuffled_a, shuffled_b, shuffled_c

def cutTree(arr, training, pos):
    lenarr = len(arr)
    if pos > lenarr:
        pos = lenarr-1
    if training == True:
        print 'true'
        return arr[:pos]
    else:
        print 'false'
        return arr[lenarr-pos:]

def cutCols(arr, varIdx, rows, cols):
    rowcount = 0
    outarr = ones((int(rows),int(cols)))
    for row in arr:
        colcount = 0
        for col in varIdx:
            outarr[rowcount][colcount] = row[col]
            colcount = colcount + 1
        rowcount = rowcount + 1
    return outarr

def onesInt(length):
    arr = []
    for i in xrange(0,length):
        arr.append(1)
    return arr

def zerosInt(length):
    arr = []
    for i in xrange(0,length):
        arr.append(0)
    return arr

def setWeights(length, weight):
    weights = []
    for i in xrange(0,length):
        weights.append(weight)
    return weights


sig = root2array('BonnTMVATrainingSample_ZH125.root','TrainTree')
random.shuffle(sig)
bkg = root2array('BonnTMVATrainingSample_ttbar.root','TrainTree')
random.shuffle(bkg)


#cut in half for training and testing, remove unwanted variables not for training
sigtemp = cutTree(sig,True,len(sig)/2)
print len(sigtemp)
bkgtemp = cutTree(bkg,True,len(bkg)/2)
print len(bkgtemp)
#keep indices of variables we want
varIdx = []
variableNames = ['m_ll','m_Bb','MET','dPhi_VH', 'ptImbalanceSignificance', 'pt_V', 'pt_Bb', 'dR_Bb', 'acop_Bb', 'dEta_Bb', 'mv1_jet0', 'mv1_jet1']
foundVariables = []


xcount = 0

for x in sig.dtype.names:
    if x in variableNames:
        varIdx.append(xcount)
        foundVariables.append(x)
    xcount = xcount + 1

print sig.dtype.names
print varIdx
print foundVariables
sigTrain = cutCols(sigtemp, varIdx, len(sigtemp), len(variableNames))
bkgTrain = cutCols(bkgtemp, varIdx, len(bkgtemp), len(variableNames))

#add the training trees together, keeping track of which entries are signal and background
xt = vstack((sigTrain, bkgTrain))
y11 = onesInt(len(sigTrain))
y21 = zerosInt(len(bkgTrain))
yt = hstack((y11, y21))
yt = transpose(yt)
bkgWeight = len(sigTrain)/len(bkgTrain)
weightsBkg = setWeights(len(bkgTrain),bkgWeight)
weightsSig = setWeights(len(sigTrain),1)
weightst = hstack((weightsSig,weightsBkg))
weightst = transpose(weightst)

x,y,weights = shuffle_in_unison(xt,yt,weightst)
print x
print y

print 'starting training on GradientBoostingClassifier'

from sklearn.ensemble import GradientBoostingClassifier

# parameters for boosting:
# GradientBoostingClassifier(loss='deviance', learning_rate=0.10000000000000001, n_estimators=100, subsample=1.0, min_samples_split=2, min_samples_leaf=1, max_depth=3, init=None, random_state=None, max_features=None, verbose=0)

gb = GradientBoostingClassifier().fit(x,y)








#Test the fit on the other half of the data
sigtemp1 = cutTree(sig,False,len(sig)/2)
bkgtemp1 = cutTree(bkg,False,len(bkg)/2)

sigTest = cutCols(sigtemp1, varIdx, len(sigtemp1), len(variableNames))
bkgTest = cutCols(bkgtemp1, varIdx, len(bkgtemp1), len(variableNames))

x1 = vstack((sigTest, bkgTest))
y1 = hstack((onesInt(len(sigTest)), zerosInt(len(bkgTest))))
y1 = transpose(y1)
print 'starting testing'

score = gb.score(x1,y1)

print score






'''
print 'starting training on ExtraTreesClassifier'
#class sklearn.ensemble.ExtraTreesClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_density=0.1, max_features='auto', bootstrap=False, compute_importances=False, oob_score=False, n_jobs=1, random_state=None, verbose=0)
from sklearn.ensemble import ExtraTreesClassifier

# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=400,
                              compute_importances=True,
                              random_state=0)

forest.fit(x, y)
importances = forest.feature_importances_
std = std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = argsort(importances)[::-1]

# Print the feature ranking
print "Feature ranking:"

for f in xrange(12):
    print "%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]])
'''

'''
# Plot the feature importances of the forest
import pylab as pl
pl.figure()
pl.title("Feature importances")
pl.bar(xrange(12), importances[indices],
       color="r", yerr=std[indices], align="center")
pl.xticks(xrange(12), indices)
pl.xlim([-1, 12])
pl.show()
'''


print 'starting training on randomforestclassifier'

#class sklearn.ensemble.RandomForestClassifier(n_estimators=12, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_density=0.1, max_features='auto', bootstrap=True, compute_importances=False, oob_score=False, n_jobs=1, random_state=None, verbose=0)
'''
from sklearn.ensemble import RandomForestClassifier

# Build a forest and compute the feature importances
forestrand = RandomForestClassifier(n_estimators=400,
                              compute_importances=True,
                              random_state=0)

forestrand.fit(x, y)
importancesrand = forestrand.feature_importances_
std = std([tree.feature_importances_ for tree in forestrand.estimators_],
             axis=0)
indicesrand = argsort(importancesrand)[::-1]

# Print the feature ranking
print "Feature ranking:"

for f in xrange(12):
    print "%d. feature %d (%f)" % (f + 1, indicesrand[f], importancesrand[indicesrand[f]])

# Plot the feature importances of the forest
import pylab as pl
pl.figure()
pl.title("Feature importances Random")
pl.bar(xrange(12), importancesrand[indicesrand],
       color="r", yerr=std[indicesrand], align="center")
pl.xticks(xrange(12), indicesrand)
pl.xlim([-1, 12])
pl.show()
'''








print 'starting training on DecisionTreeClassifier'
#class sklearn.tree.DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_density=0.10000000000000001, max_features=None, compute_importances=False, random_state=None)


from sklearn.tree import DecisionTreeClassifier
'''
# Build a forest and compute the feature importances
dectree = DecisionTreeClassifier(compute_importances=True)

dectree.fit(x, y)
importancesdectree = dectree.feature_importances_
print importancesdectree
print dectree.score(x1,y1)
'''
'''
std = std([tree.feature_importances_ for tree in dectree.estimators_],
             axis=0)
indicesdectree = argsort(importancesdectree)[::-1]

# Print the feature ranking
print "Feature ranking:"

for f in xrange(12):
    print "%d. feature %d (%f)" % (f + 1, indicesdectree[f], importancesdectree[indicesdectree[f]])


# Plot the feature importances of the forest
import pylab as pl
pl.figure()
pl.title("Feature importances Dectreeom")
pl.bar(xrange(12), importancesdectree[indicesdectree],
       color="r", yerr=std[indicesdectree], align="center")
pl.xticks(xrange(12), indicesdectree)
pl.xlim([-1, 12])
pl.show()
'''




print 'starting training on AdaBoostClassifier'
#class sklearn.tree.DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_density=0.10000000000000001, max_features=None, compute_importances=False, random_state=None)


from sklearn.ensemble import AdaBoostClassifier

# Build a forest and compute the feature importances
ada = AdaBoostClassifier(DecisionTreeClassifier(compute_importances=True,max_depth=4,min_samples_split=20,min_samples_leaf=100),n_estimators=400, learning_rate=0.5, algorithm="SAMME",compute_importances=True)

ada.fit(x, y)
importancesada = ada.feature_importances_
print importancesada
#print ada.score(x1,y1)

std = std([tree.feature_importances_ for tree in ada.estimators_],
             axis=0)
indicesada = argsort(importancesada)[::-1]
variableNamesSorted = []
for i in indicesada:
    variableNamesSorted.append(foundVariables[i])

# Print the feature ranking
print "Feature ranking:"

for f in xrange(12):
    print "%d. feature %d (%f)" % (f + 1, indicesada[f], importancesada[indicesada[f]]) + " " +variableNamesSorted[f]


# Plot the feature importances of the forest

import pylab as pl

pl.figure()
pl.title("Feature importances Ada")
pl.bar(xrange(12), importancesada[indicesada],
       color="r", yerr=std[indicesada], align="center")
pl.xticks(xrange(12), variableNamesSorted)#indicesada)
pl.xlim([-1, 12])
pl.show()

plot_colors = "br"
plot_step = 1000.0
class_names = "AB"


pl.figure(figsize=(15, 5))
'''
# Plot the decision boundaries
pl.subplot(131)
x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
print 'xmin ' + str(x_min)
print 'xmax ' + str(x_max)
print 'ymin ' + str(y_min)
print 'ymax ' + str(y_max)
xx, yy = meshgrid(arange(x_min, x_max, plot_step),
                     arange(y_min, y_max, plot_step))

Z = ada.predict(c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
cs = pl.contourf(xx, yy, Z, cmap=pl.cm.Paired)
pl.axis("tight")
'''




# Plot the training points
for i, n, c in zip(xrange(2), class_names, plot_colors):
    idx = where(y == i)
    pl.scatter(x[idx, 0], x[idx, 1],
               c=c, cmap=pl.cm.Paired,
               label="Class %s" % n)
pl.axis("tight")
pl.legend(loc='upper right')
pl.xlabel("Decision Boundary")





# Plot the class probabilities
class_proba = ada.predict_proba(x)[:, -1]
pl.subplot(132)
for i, n, c in zip(xrange(2), class_names, plot_colors):
    pl.hist(class_proba[y == i],
            bins=20,
            range=(0, 1),
            facecolor=c,
            label='Class %s' % n)
pl.legend(loc='upper center')
pl.ylabel('Samples')
pl.xlabel('Class Probability')

# Plot the two-class decision scores
twoclass_output = ada.decision_function(x)
pl.subplot(133)
for i, n, c in zip(xrange(2), class_names, plot_colors):
    pl.hist(twoclass_output[y == i],
            bins=20,
            range=(-1, 1),
            facecolor=c,
            label='Class %s' % n)
pl.legend(loc='upper right')
pl.ylabel('Samples')
pl.xlabel('Two-class Decision Scores')

pl.subplots_adjust(wspace=0.25)
pl.show()
