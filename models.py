
# CLASSIFIERS ----------------------------------

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Names.
cnames = ["Nearest_Neighbors", "SVM_Linear", "SVM_RBF", "Gaussian_NB", "Bernoulli_NB", "QuadraticDA",
           "Stochastic_GDC", "Decision_Tree", "Random_Forest", "NN_MLP", "Ada_Boost", "XGBC_Boost", "Cat_Boost"]

# Models.
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.010),
    SVC(gamma=2, C=1),
    GaussianNB(),
    BernoulliNB(),
    QuadraticDiscriminantAnalysis(),
    SGDClassifier(loss="hinge", penalty="elasticnet", max_iter=5),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    XGBClassifier(),
    CatBoostClassifier(logging_level='Silent')]

# REGRESORS ----------------------------------

from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, Ridge, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

# Regresion
rnames = [
    "Linear", "Lasso", "Elastic Net", "Ridge", "Huber", "Random Forest",
    "Gradient Boosting", "Support Vector", "KNeighbors", "XGBoost"]

# Models.
regressors = [
    LinearRegression(),
    Lasso(alpha=0.1),
    ElasticNet(random_state=0),
    Ridge(alpha=1.0),
    HuberRegressor(),
    RandomForestRegressor(random_state=0),
    GradientBoostingRegressor(random_state=0),
    SVR(),
    KNeighborsRegressor(n_neighbors=2),
    XGBRegressor(objective="reg:squarederror", random_state=0)
]