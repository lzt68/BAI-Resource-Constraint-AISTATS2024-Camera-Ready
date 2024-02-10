import numpy as np
from random import Random
from typing import Tuple
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.base import clone


class Env_Classifier(object):
    def __init__(self, dataset: callable, Match_Index_to_Model: dict, K: int = 2, C: np.ndarray = np.array([60.0]), L: int = 1, random_seed=12345) -> None:
        """In this environment, the arm is the classification machine learning model,
        the reward is the prediction accuracy, the consumption is the running time of each model

        Args:
            dataset (callable): callable object that contains data and target
            Match_Index_to_Model (dict): The dictionary that matchs arm index (1,2,...,K) to machine learning model
            K (int, optional): The total number of arms. Defaults to 2.
            C (int, optional): Initial Resource. Defaults to np.array([60.]), which means time
            L (int, optional): Number of resource. Defaults to 1.
            random_seed (int, optional): Random seed. Defaults to 12345.
        """
        assert C.shape[0] == L, "number of initial resources doesn't match"
        assert len(Match_Index_to_Model) == K, "number of arms doesn't match"
        assert np.all(C > 0), "initial resource should be greater than 0"

        self.K = K
        self.C = C
        self.L = L
        self.Match_Index_to_Model = Match_Index_to_Model
        self.dataset = dataset

        self.consumption = np.zeros(L)
        self.stop = False  # when the consumption > C-1, the algorithm stops
        self.random_seed = random_seed
        self.random_generator = Random()
        self.random_generator.seed(random_seed)

    def response(self, arm: int) -> Tuple[np.ndarray, float]:
        """Respond to the selected arm of agent

        Args:
            arm (int): arm index range from 1 to K

        Returns:
            consumption (np.ndarray): The consumption of each resources
            reward (float): The realized reward
        """
        if not self.stop:
            # spli the training and testing dataset with different random seed
            new_random_state = self.random_generator.randint(0, 2**31 - 1)
            X_train, X_test, Y_train, Y_test = train_test_split(self.dataset["data"], self.dataset["target"], test_size=0.3, random_state=new_random_state)

            # trian a machine learning model
            model = clone(self.Match_Index_to_Model[arm])
            t1 = time.time()
            model.fit(X_train, Y_train)
            y_test_predict = model.predict(X_test)
            t2 = time.time()

            # calculate the performance
            consumption = t2 - t1
            reward = accuracy_score(Y_test, y_test_predict)
            self.consumption += consumption
            if np.any(self.consumption >= self.C - 1):
                self.stop = True
            return consumption, reward
        else:
            return None

    def if_stop(self):
        return self.stop


class Env_Classifier_CrossEntropy(object):
    def __init__(self, dataset: callable, Match_Index_to_Model: dict, K: int = 2, C: np.ndarray = np.array([60.0]), L: int = 1, random_seed=12345) -> None:
        """In this environment, the arm is the classification machine learning model,
        the reward is the negative value of cross entropy,
        the consumption is the running time of each model

        Args:
            dataset (callable): callable object that contains data and target
            Match_Index_to_Model (dict): The dictionary that matchs arm index (1,2,...,K) to machine learning model
            K (int, optional): The total number of arms. Defaults to 2.
            C (int, optional): Initial Resource. Defaults to np.array([60.]), which means time
            L (int, optional): Number of resource. Defaults to 1.
            random_seed (int, optional): Random seed. Defaults to 12345.
        """
        assert C.shape[0] == L, "number of initial resources doesn't match"
        assert len(Match_Index_to_Model) == K, "number of arms doesn't match"
        assert np.all(C > 0), "initial resource should be greater than 0"

        self.K = K
        self.C = C
        self.L = L
        self.Match_Index_to_Model = Match_Index_to_Model
        self.dataset = dataset

        self.consumption = np.zeros(L)
        self.stop = False  # when the consumption > C-1, the algorithm stops
        self.random_seed = random_seed
        self.random_generator = Random()
        self.random_generator.seed(random_seed)

    def response(self, arm: int) -> Tuple[np.ndarray, float]:
        """Respond to the selected arm of agent

        Args:
            arm (int): arm index range from 1 to K

        Returns:
            consumption (np.ndarray): The consumption of each resources
            reward (float): The realized reward
        """
        if not self.stop:
            # spli the training and testing dataset with different random seed
            new_random_state = self.random_generator.randint(0, 2**31 - 1)
            X_train, X_test, Y_train, Y_test = train_test_split(self.dataset["data"], self.dataset["target"], test_size=0.3, random_state=new_random_state)

            # trian a machine learning model
            model = clone(self.Match_Index_to_Model[arm])
            t1 = time.time()
            model.fit(X_train, Y_train)
            y_test_predict_proba = model.predict_proba(X_test)
            t2 = time.time()

            # calculate the performance
            consumption = t2 - t1
            reward = -log_loss(Y_test, y_test_predict_proba)
            self.consumption += consumption
            if np.any(self.consumption >= self.C - 1):
                self.stop = True
            return consumption, reward
        else:
            return None

    def if_stop(self):
        return self.stop


class Env_Classifier_CrossEntropy_divide3plus1(object):
    def __init__(self, dataset: callable, Match_Index_to_Model: dict, K: int = 2, C: np.ndarray = np.array([60.0]), L: int = 1, random_seed=12345) -> None:
        """In this environment, the arm is the classification machine learning model,
        the reward is the negative value of cross entropy,
        the consumption is the running time of each model

        Args:
            dataset (callable): callable object that contains data and target
            Match_Index_to_Model (dict): The dictionary that matchs arm index (1,2,...,K) to machine learning model
            K (int, optional): The total number of arms. Defaults to 2.
            C (int, optional): Initial Resource. Defaults to np.array([60.]), which means time
            L (int, optional): Number of resource. Defaults to 1.
            random_seed (int, optional): Random seed. Defaults to 12345.
        """
        assert C.shape[0] == L, "number of initial resources doesn't match"
        assert len(Match_Index_to_Model) == K, "number of arms doesn't match"
        assert np.all(C > 0), "initial resource should be greater than 0"

        self.K = K
        self.C = C
        self.L = L
        self.Match_Index_to_Model = Match_Index_to_Model
        self.dataset = dataset

        self.consumption = np.zeros(L)
        self.stop = False  # when the consumption > C-1, the algorithm stops
        self.random_seed = random_seed
        self.random_generator = Random()
        self.random_generator.seed(random_seed)

    def response(self, arm: int) -> Tuple[np.ndarray, float]:
        """Respond to the selected arm of agent

        Args:
            arm (int): arm index range from 1 to K

        Returns:
            consumption (np.ndarray): The consumption of each resources
            reward (float): The realized reward
        """
        if not self.stop:
            # spli the training and testing dataset with different random seed
            new_random_state = self.random_generator.randint(0, 2**31 - 1)
            X_train, X_test, Y_train, Y_test = train_test_split(self.dataset["data"], self.dataset["target"], test_size=0.3, random_state=new_random_state)

            # trian a machine learning model
            model = clone(self.Match_Index_to_Model[arm])
            t1 = time.time()
            model.fit(X_train, Y_train)
            y_test_predict_proba = model.predict_proba(X_test)
            t2 = time.time()

            # calculate the performance
            consumption = t2 - t1
            reward = -log_loss(Y_test, y_test_predict_proba)
            reward = reward / 3 + 1  # we map the reward to the (0, 1)
            self.consumption += consumption
            if np.any(self.consumption >= self.C - 1):
                self.stop = True
            return consumption, reward
        else:
            return None

    def if_stop(self):
        return self.stop
