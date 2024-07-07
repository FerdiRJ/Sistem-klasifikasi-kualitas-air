import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
import random
import math
import time

# Load dataset
dt = pd.read_csv("preprosess.csv")
x = dt.iloc[:, :-1]
y = dt['probabilitas']

xtrain,xtest,ytrain,ytest = train_test_split(x,y,train_size=0.75,random_state=1)

smote = SMOTE(random_state=1)
xtrain_resampled, ytrain_resampled = smote.fit_resample(xtrain, ytrain)

# Function to calculate accuracy using KNN with given parameters
def evaluate_knn(n_neighbors):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(xtrain_resampled, ytrain_resampled)
    y_pred = model.predict(xtest)
    return accuracy_score(ytest, y_pred)

# Custom transformer for simulated annealing
class SimulatedAnnealingKNN(BaseEstimator, TransformerMixin):
    def __init__(self, iterations=50):
        self.iterations = iterations
        self.best_parameters_ = None
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train, self.y_train = X, y
        self.best_parameters_, _ = self.parallel_simulated_annealing_knn(X, y)
        return self

    def transform(self, X):
        return X

    def predict(self, X):
        if self.best_parameters_ is None:
            raise ValueError("fit method must be called before predict")
        
        n_neighbors = self.best_parameters_[0]
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(self.X_train, self.y_train)
        return knn.predict(X)

    def parallel_simulated_annealing_knn(self, X, y):
        results = Parallel(n_jobs=-1)(
            delayed(self.simulated_annealing_knn)(X, y) for _ in range(self.iterations)
        )
        return max(results, key=lambda x: x[1])

    def simulated_annealing_knn(self, X, y):
        # Initial parameters
        current_n_neighbors = random.randint(1, 200)

        current_accuracy = evaluate_knn(current_n_neighbors)
        best_accuracy = current_accuracy
        best_parameters = (current_n_neighbors,)

        # Parameters for simulated annealing
        initial_temperature = 100
        final_temperature = 0.1
        cooling_rate = 0.95
        iterations = 50

        current_temperature = initial_temperature

        for i in range(iterations):
            new_n_neighbors = max(1, min(1000, current_n_neighbors + random.randint(-5, 5)))

            new_accuracy = evaluate_knn(new_n_neighbors)

            # Acceptance probability
            if new_accuracy > current_accuracy or random.uniform(0, 1) < math.exp((new_accuracy - current_accuracy) / current_temperature):
                current_n_neighbors = new_n_neighbors
                current_accuracy = new_accuracy

                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    best_parameters = (current_n_neighbors,)

            # Cooling
            current_temperature *= cooling_rate
            if current_temperature < final_temperature:
                break
        return best_parameters, best_accuracy