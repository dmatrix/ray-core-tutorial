import time
from typing import Tuple
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import xgboost as xgb
import ray

# states to inspect 
STATES = ["INITIALIZED", "RUNNING", "DONE"]

RANDOM_FOREST_CONFIGS = {"n_estimators": 150,
                         "name": "random_forest",
                         "type": "lr"}
DECISION_TREE_CONFIGS = {"max_depth": 15,
                         "name": "decision_tree",
                         "type": "lr"}
XGBOOST_CONFIGS =  {"max_depth": 10,
                    "n_estimators": 150,
                    "name": "xgboost",
                    "type": "lr"}

@ray.remote
class RFRActor:
    """
    An actor model to train and score the calfornia house data using Random Forest Regressor
    """
    def __init__(self, **kwargs):
        self.kwargs= kwargs
        self.name = kwargs["name"]
        self.estimators = kwargs["n_estimators"]
        self.state = STATES[0]
        self.X, self.y = None, None
        self.X_train, self.X_test = None, None
        self.y_train, self.y_test = None, None
        self.model = None

    def _prepare_data_and_model(self) -> None:
        self.X, self.y = fetch_california_housing(return_X_y=True, as_frame=True)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.model = RandomForestRegressor(n_estimators=self.estimators, random_state=42)

    def train_and_evaluate_model(self) -> Tuple[int, str, float,float]:
        """
        Train the model and evaluate and report MSE
        """
        self._prepare_data_and_model()

        print(f"Start training model {self.name} with estimators: {self.estimators} ...")

        start_time = time.time()
        self.model.fit(self.X_train, self.y_train)
        self.state = STATES[1]
        y_pred = self.model.predict(self.X_test)
        score = mean_squared_error(self.y_test, y_pred)
        self.state = STATES[2]

        end_time = time.time()
        print(f"End training model {self.name} with estimators: {self.estimators} took: {end_time - start_time:.2f} seconds")

        return  self.get_state(), self.estimators, round(score, 4), round(end_time - start_time, 2)

    def get_state(self) -> str:
        return self.state

@ray.remote
class DTActor:
    """
    An actor model to train and score the calfornia house data using Decision Tree Regressor
    """
    def __init__(self, **kwargs):
        self.kwargs= kwargs
        self.name = kwargs["name"]
        self.max_depth = kwargs["max_depth"]
        self.state = STATES[0]
        self.X, self.y = None, None
        self.X_train, self.X_test = None, None
        self.y_train, self.y_test = None, None
        self.model = None

    def _prepare_data_and_model(self) -> None:
        self.X, self.y = fetch_california_housing(return_X_y=True, as_frame=True)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.model = DecisionTreeRegressor(max_depth=self.max_depth, random_state=42)

    def train_and_evaluate_model(self) -> Tuple[int, str, float,float]:
        """
        Train the model and evaluate and report MSE
        """
        self._prepare_data_and_model()
        print(f"Start training model {self.name} with max depth: { self.max_depth } ...")

        start_time = time.time()
        self.model.fit(self.X_train, self.y_train)
        self.state = STATES[1]
        y_pred = self.model.predict(self.X_test)
        score = mean_squared_error(self.y_test, y_pred)
        self.state = STATES[2]

        end_time = time.time()
        print(f"End training model {self.name} with max_depth tree: {self.max_depth} took: {end_time - start_time:.2f} seconds")

        return  self.get_state(), self.max_depth, round(score, 4), round(end_time - start_time, 2)

    def get_state(self) -> str:
        return self.state

@ray.remote
class XGBoostActor:
    """
    An actor model to train and score the calfornia house data using XGBoost Regressor
    """
    def __init__(self, **kwargs):
        self.kwargs= kwargs
        self.name = kwargs["name"]
        self.max_depth = kwargs["max_depth"]
        self.estimators = kwargs["n_estimators"]
        self.state = STATES[0]
        self.X, self.y = None, None
        self.X_train, self.X_test = None, None
        self.y_train, self.y_test = None, None
        self.model = None

    def _prepare_data_and_model(self) -> None:
        self.X, self.y = fetch_california_housing(return_X_y=True, as_frame=True)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.model = xgb.XGBRegressor(objective='reg:squarederror',
                                      colsample_bytree=1,
                                      eta=0.3,
                                      learning_rate = 0.1,
                                      max_depth=self.max_depth,
                                      n_estimators=self.estimators,
                                      random_state=42)
    
    def train_and_evaluate_model(self) -> Tuple[int, str, float,float]:
        """
        Train the model and evaluate and report MSE
        """
        self._prepare_data_and_model()
        print(f"Start training model {self.name} with estimators: {self.estimators} and max depth: { self.max_depth } ...")
        start_time = time.time()
        self.model.fit(self.X_train, self.y_train)
        self.state = STATES[1]
        y_pred = self.model.predict(self.X_test)
        score = mean_squared_error(self.y_test, y_pred)
        self.state = STATES[2]

        end_time = time.time()
        print(f"End training model {self.name} with estimators: {self.estimators} and max depth: { self.max_depth } and took: {end_time - start_time:.2f})")

        return  self.get_state(), self.max_depth, round(score, 4), round(end_time - start_time, 2)

    def get_state(self) -> str:
        return self.state