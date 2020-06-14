%%time
from sklearn.datasets import make_regression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
model=MultiOutputRegressor(GradientBoostingRegressor(random_state=0))
model.fit(X_train,y_train)
