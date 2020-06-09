#example
from sklearn.model_selection import GroupKFold
X = [0,1,2,3,4,5,6,7,8,9,10]
y = ["a", "b", "b", "b", "c", "c", "c", "d", "d", "e","e"]
groups = [1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3]
print("Criterion: 2<=n_splits<=n_groups. Also, there is no randomness hence random_state=None\n")
gkf = GroupKFold(n_splits=2)
for i,j in enumerate(gkf.split(X, y, groups=groups)):
    train, val = j
    print("split -",i+1)
    print("%s %s" % (train, val))
    print("train size:%s, val size:%s" % (len(train),len(val)))
    print("-"*50)
print("\n-The train & val sizes may differ in each split")
