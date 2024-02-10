# The source file is to generate different synthesis experiment setting
from __future__ import annotations
from typing import Union
import numpy as np
from tqdm import tqdm
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.base import clone
import pickle


def GetBestArm(model_list, n_ground_truth, dataset):
    Match_Index_to_Model = dict()
    for ii, model in enumerate(model_list):
        Match_Index_to_Model[ii + 1] = model
    cross_entropy_ = np.zeros((len(Match_Index_to_Model), n_ground_truth))
    running_time_ = np.zeros((len(Match_Index_to_Model), n_ground_truth))
    for arm_index in range(1, len(Match_Index_to_Model) + 1):
        for exp_index in tqdm(range(n_ground_truth)):
            # split the dataset with different random seed
            new_random_state = np.random.randint(0, 2**31 - 1)
            X_train, X_test, Y_train, Y_test = train_test_split(dataset["data"], dataset["target"], test_size=0.3, random_state=new_random_state)

            t1 = time.time()
            model = clone(Match_Index_to_Model[arm_index])
            model.fit(X_train, Y_train)
            y_test_predict_proba = model.predict_proba(X_test)
            t2 = time.time()

            cross_entropy_[arm_index - 1, exp_index] = -log_loss(Y_test, y_test_predict_proba)
            running_time_[arm_index - 1, exp_index] = t2 - t1
    cross_entropy_mean_ = np.mean(cross_entropy_, axis=1)
    running_time_mean_ = np.mean(running_time_, axis=1)
    best_arm = np.argmax(cross_entropy_mean_) + 1
    print(f"best arm is {best_arm}, best model is {model_list[best_arm-1].__str__()}")
    for ii, model in enumerate(model_list):
        print(f"{model.__str__()}, entropy {-cross_entropy_mean_[ii]}, running time {running_time_mean_[ii]}")
    return best_arm, Match_Index_to_Model, cross_entropy_, running_time_


def Experiment_Classiflier(
    model_list: list,
    best_arm: Union[None, int],
    dataset: dict,
    env_class,
    agent_class_list,
    # agent_para: dict,
    n_experiment: int,
    K: int,
    C: np.ndarray,
    L: int,
    random_seed: int = 0,
    disable_tqdm: bool = False,
    shuffle: bool = True,
    n_ground_truth: int = 1000,  # the repeated times of algorithm to get the ground truth best arm
):
    assert len(model_list) == K, "Number of arms doesn't match"
    assert len(C) == L, "number of resources doesn't match"
    if_bset_arm_none = best_arm is None
    # get the ground truth best arm
    if if_bset_arm_none:
        best_arm, Match_Index_to_Model, cross_entropy_, running_time_ = GetBestArm(model_list=model_list, n_ground_truth=n_ground_truth, dataset=dataset)
    else:
        print(f"Accepted best arm is {best_arm}")
        Match_Index_to_Model = dict()
        for ii, model in enumerate(model_list):
            Match_Index_to_Model[ii + 1] = model

    # conduct the experiment
    for agent_class in agent_class_list:
        best_arm_ = np.zeros(n_experiment)
        predict_arm_ = np.zeros(n_experiment)
        stopping_times_ = np.zeros(n_experiment)
        for exp_index in tqdm(range(n_experiment), disable=disable_tqdm):
            # different random_seed will generate different permutation
            np.random.seed(random_seed + exp_index)

            # shffule the arm
            if shuffle:
                index = np.arange(len(Match_Index_to_Model))
                key_list = np.array(list(Match_Index_to_Model.keys()))
                np.random.shuffle(index)
                key_list = key_list[index]
                new_Match_Index_to_Model = dict()
                for ii in range(1, len(key_list) + 1):
                    new_Match_Index_to_Model[ii] = Match_Index_to_Model[key_list[ii - 1]]
                for new_index in range(1, len(Match_Index_to_Model) + 1):
                    if key_list[new_index - 1] == best_arm:
                        new_best_arm = new_index
                        break
                best_arm_[exp_index] = new_best_arm
            else:
                best_arm_[exp_index] = best_arm

            # conduct the experiment
            if shuffle:
                env = env_class(dataset=dataset, Match_Index_to_Model=new_Match_Index_to_Model, K=K, L=L, C=C, random_seed=random_seed + exp_index)
            else:
                env = env_class(dataset=dataset, Match_Index_to_Model=Match_Index_to_Model, K=K, L=L, C=C, random_seed=random_seed + exp_index)
            # agent_para["K"] = K
            # agent_para["C"] = C
            # agent_para["L"] = L
            agent = agent_class(K=K, C=C, L=L)
            while not env.if_stop():
                arm = agent.action()
                d, r = env.response(arm)
                agent.observe(demand=d, reward=r)
            predict_arm_[exp_index] = agent.predict()
            stopping_times_[exp_index] = agent.t

        # calculate the return value
        success_rate = np.mean(predict_arm_ == best_arm_)
        std_success_rate = np.sqrt(success_rate * (1 - success_rate)) / np.sqrt(n_experiment)
        stopping_times = np.mean(stopping_times_)

        filename = f"{agent_class.__name__}.pickle"
        with open(filename, "wb") as f:
            pickle.dump(success_rate, f)
            pickle.dump(std_success_rate, f)
            pickle.dump(stopping_times_, f)
            pickle.dump(predict_arm_, f)
            pickle.dump(best_arm_, f)
            if if_bset_arm_none:
                pickle.dump(cross_entropy_, f)
                pickle.dump(running_time_, f)
        print(f"model {agent_class.__name__}, success rate is {success_rate}, std is {std_success_rate}")

    # return success_rate, std_success_rate, stopping_times_, predict_arm_, best_arm_, cross_entropy_, running_time_


def Experiment_Classiflier_Single(
    model_list: list,
    dataset: dict,
    env_class,
    agent_class,
    agent_para: dict,
    n_experiment: int,
    K: int,
    C: np.ndarray,
    L: int,
    random_seed: int = 0,
    disable_tqdm: bool = False,
    shuffle: bool = True,
    n_ground_truth: int = 1000,  # the repeated times of algorithm to get the ground truth best arm
):
    assert len(model_list) == K, "Number of arms doesn't match"
    assert len(C) == L, "number of resources doesn't match"

    # get the ground truth best arm
    Match_Index_to_Model = dict()
    for ii, model in enumerate(model_list):
        Match_Index_to_Model[ii + 1] = model
    cross_entropy_ = np.zeros((len(Match_Index_to_Model), n_ground_truth))
    running_time_ = np.zeros((len(Match_Index_to_Model), n_ground_truth))
    for arm_index in range(1, len(Match_Index_to_Model) + 1):
        for exp_index in tqdm(range(n_ground_truth)):
            # split the dataset with different random seed
            new_random_state = np.random.randint(0, 2**31 - 1)
            X_train, X_test, Y_train, Y_test = train_test_split(dataset["data"], dataset["target"], test_size=0.3, random_state=new_random_state)

            t1 = time.time()
            model = Match_Index_to_Model[arm_index]
            model.fit(X_train, Y_train)
            t2 = time.time()

            y_test_predict_proba = model.predict_proba(X_test)
            cross_entropy_[arm_index - 1, exp_index] = -log_loss(Y_test, y_test_predict_proba)
            running_time_[arm_index - 1, exp_index] = t2 - t1
    cross_entropy_mean_ = np.mean(cross_entropy_, axis=1)
    # running_time_mean_ = np.mean(running_time_, axis=1)
    best_arm = np.argmax(cross_entropy_mean_) + 1

    # conduct the experiment
    best_arm_ = np.zeros(n_experiment)
    predict_arm_ = np.zeros(n_experiment)
    stopping_times_ = np.zeros(n_experiment)
    for exp_index in tqdm(range(n_experiment), disable=disable_tqdm):
        # different random_seed will generate different permutation
        np.random.seed(random_seed + exp_index)

        # shffule the arm
        if shuffle:
            index = np.arange(len(Match_Index_to_Model))
            key_list = np.array(list(Match_Index_to_Model.keys()))
            np.random.shuffle(index)
            key_list = key_list[index]
            new_Match_Index_to_Model = dict()
            for ii in range(1, len(key_list) + 1):
                new_Match_Index_to_Model[ii] = Match_Index_to_Model[key_list[ii - 1]]
            for new_index in range(1, len(Match_Index_to_Model) + 1):
                if key_list[new_index - 1] == best_arm:
                    new_best_arm = new_index
                    break
            best_arm_[exp_index] = new_best_arm
        else:
            best_arm_[exp_index] = best_arm

        # conduct the experiment
        if shuffle:
            env = env_class(dataset=dataset, Match_Index_to_Model=new_Match_Index_to_Model, K=K, L=L, C=C, random_seed=random_seed + exp_index)
        else:
            env = env_class(dataset=dataset, Match_Index_to_Model=Match_Index_to_Model, K=K, L=L, C=C, random_seed=random_seed + exp_index)
        agent_para["K"] = K
        agent_para["C"] = C
        agent_para["L"] = L
        agent = agent_class(**agent_para)
        while not env.if_stop():
            arm = agent.action()
            d, r = env.response(arm)
            agent.observe(demand=d, reward=r)
        predict_arm_[exp_index] = agent.predict()
        stopping_times_[exp_index] = agent.t

    # calculate the return value
    success_rate = np.mean(predict_arm_ == best_arm_)
    std_success_rate = np.sqrt(success_rate * (1 - success_rate)) / np.sqrt(n_experiment)
    stopping_times_ = np.mean(stopping_times_)

    return success_rate, std_success_rate, stopping_times_, predict_arm_, best_arm_, cross_entropy_, running_time_
