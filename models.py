from sklearn import linear_model
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVR
from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
 from sklearn.gaussian_process import GaussianProcessRegressor


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

# Regresion
c_names = ["Lasso", "Elastic-Net", "Log. Regr", "Linear SVR", "SGD Regressor", "KNeighbors Regr.", "Gaussian Regr",

           ]

# Models.
classifiers = [
    linear_model.Lasso(alpha=0.1),
    ElasticNet(random_state=0),
    LogisticRegression(random_state=0),
    LinearSVR(dual='auto', random_state=0, tol=1e-05),
    SGDRegressor(max_iter=1000, tol=1e-3),
    KNeighborsRegressor(n_neighbors=2),
    GaussianProcessRegressor(kernel=kernel, random_state=0)

]

# Classifiers
# Names.
c_names = ["Nearest Neighbors", "SVM Linear", "SVM RBF", "Gaussian NB", "Bernoulli NB", "QuadraticDA",
           "Stochastic GDC", "Decision Tree", "Random Forest", "NN MLP", "Ada Boost", "XGBC Boost", "Cat Boost"]

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
