# source file of agent

import numpy as np
from random import Random


class Uniform_Agent:  # use round robin to pull each arm
    def __init__(self, K=2, C=10, L=1) -> None:
        """Construct an instance of uniformly pulling policy

        Args:
            K (int, optional): Total number of arms. Defaults to 2.
            C (int, optional): Available initial resource. Defaults to 10.
            L (int, optional): Number of resources. Defaults to 1.
        """
        self.K = K
        self.C = C
        self.L = L
        self.t = 0  # index of epoch
        self.arm_ = list()  # record the action in each epoch
        self.pulling_times_ = np.zeros(K)
        self.total_reward_ = np.zeros(K)
        self.mean_reward_ = np.ones(K) * (-99)

    def action(self):
        # return the pulling arm in this epoch
        arm = self.t % self.K + 1
        self.arm_.append(arm)
        return arm

    def observe(self, demand, reward):
        # recorded the observed demand and reward
        self.t = self.t + 1
        self.pulling_times_[self.arm_[-1] - 1] += 1
        self.total_reward_[self.arm_[-1] - 1] += reward
        self.mean_reward_[self.arm_[-1] - 1] = (
            self.total_reward_[self.arm_[-1] - 1] / self.pulling_times_[self.arm_[-1] - 1]
        )

    def predict(self):
        # output the predicted best arm, we need to make sure the pulling times of each arm is the same
        # pulling_times = self.t // self.K
        # mean_reward = np.array([np.mean(self.reward_[ii][:pulling_times]) for ii in range(1, self.K + 1)])
        best_arm = np.argmax(self.mean_reward_) + 1
        return best_arm


class UCB_Agent:
    def __init__(self, K=2, C=np.array([10]), L=1, a=2) -> None:
        """Construct an instance of Upper Confidence Bound Policy

        Args:
            K (int, optional): Total number of arms. Defaults to 2.
            C (np.ndarray, optional): Available initial resource. Defaults to np.array([10]).
            a (int, optional): The exploration factor. Defaults to 2.
        """
        assert len(C) == L, f"number of resources doesn't match, C {C}, L {L}"

        self.K = K
        self.C = C
        self.L = L
        self.a = a
        self.t = 1
        self.pulling_times_ = np.zeros(K)
        self.arm_ = list()  # record the action in each epoch
        self.total_consumption = 0

        self.demand_ = np.zeros((L, K))  # total consumption
        self.reward_ = np.zeros(K)  # total reward
        self.mean_reward_ = np.zeros(K)
        self.confidence_ = np.ones(K) * 9999

        self.J = 1  # recommended best arm
        self.J_ = list()

    def action(self):
        # return the pulling arm in this epoch
        upper_bound_ = self.mean_reward_ + self.confidence_
        arm = np.argmax(upper_bound_) + 1
        self.arm_.append(arm)
        return arm

    def observe(self, demand, reward):
        # record the arms in this round
        arm_index = self.arm_[-1] - 1

        # update the history of this arm
        self.reward_[arm_index] += reward
        self.demand_[:, arm_index] += demand
        self.total_consumption += demand
        self.pulling_times_[arm_index] += 1
        self.mean_reward_[arm_index] = self.reward_[arm_index] / self.pulling_times_[arm_index]
        self.t += 1

        # generate a new arm
        if self.t <= self.K + 1:
            # some of the pulling times equal to zero
            index = self.pulling_times_ > 0
            self.confidence_[index] = np.sqrt(self.a * np.log(self.t) / self.pulling_times_[index])
        else:
            self.confidence_ = np.sqrt(self.a * np.log(self.t) / self.pulling_times_)
        self.J = np.argmax(self.mean_reward_) + 1
        self.J_.append(self.J)

    def predict(self):
        # output the predicted best arm
        arm = self.J
        return arm


class SequentialHalvingRR_Recycle_FailureFlag_History_Agent:
    # use round robin to pull remaining arms, and eliminate half of the remaining arms
    # And we try not to throw out unused resource in each round
    def __init__(self, K=2, C=np.array([10]), L=1) -> None:
        """In this version, if there are some arms never getting pulled but we need to conduct the elimination,
        the algorithm will end with failure and predict -1, but the algorithm will not throw an exception
        And we will not abandon pulling history

        Args:
            K (int, optional): Total number of arms. Defaults to 2.
            C (np.ndarray, optional): Available initial resource. Defaults to np.array([10]).
            L (int, optional): The number of resources. Defaults to 1.
        """
        assert len(C) == L, "number of resources doesn't match"
        self.failure_flag = False

        self.K = K
        self.C = C
        self.L = L
        self.t = 0  # index of round
        self.t_q = 0  # index of round in each phase
        self.q = 0  # index of phase
        self.arm_ = list()  # record the action in each epoch

        self.demand_ = dict()  # record the consumption of arms in each epoch
        self.reward_ = dict()  # record the observed reward of arms in each epoch
        self.consumption = np.zeros(L)  # record the total consumption in each phase
        for arm_index in range(1, K + 1):
            # for each arm, create a list
            # when we enter a new epoch, we clear the existing memory
            self.demand_[arm_index] = list()
            self.reward_[arm_index] = list()
        self.pulling_times_ = np.zeros(K)

        self.total_demand_ = dict()
        self.total_reward_ = dict()
        for arm_index in range(1, K + 1):
            # for each arm, create a list
            # but we will not clear the memory
            self.total_demand_[arm_index] = list()
            self.total_reward_[arm_index] = list()
        self.total_consumption = np.zeros(L)  # the total consumption of all the phase

        self.survive_arms = list(range(1, K + 1))
        # self.ration_q_ = np.zeros(int(np.ceil(C / np.ceil(np.log2(K)))))
        self.ration_q_ = np.zeros((L, int(np.ceil(np.log2(K)))))
        self.ration_q_[:, 0] = C / np.ceil(np.log2(K))

    def action(self):
        # return the pulling arm in this epoch
        if not self.failure_flag:
            # the algorithm doesn't meet and
            index = self.t_q % len(self.survive_arms)
            arm = self.survive_arms[index]
            self.arm_.append(arm)
            return arm
        else:
            arm = 1
            self.arm_.append(arm)
            return arm

    def observe(self, demand, reward):
        if self.failure_flag:
            # exception occurs
            return

        # record the arms in this phase
        self.reward_[self.arm_[-1]].append(reward)
        self.demand_[self.arm_[-1]].append(demand)
        self.pulling_times_[self.arm_[-1] - 1] += 1

        # record the arms in this overall array
        self.total_reward_[self.arm_[-1]].append(reward)
        self.total_demand_[self.arm_[-1]].append(demand)

        # update the consumption
        self.consumption = self.consumption + demand
        self.total_consumption = self.total_consumption + demand

        # update the index of rounds
        self.t = self.t + 1
        self.t_q = self.t_q + 1

        if len(self.survive_arms) == 1:
            return

        # check whether conduct the elimination
        if np.any(self.consumption >= self.ration_q_[:, self.q] - 1):
            self.eliminate()

    def predict(self):
        if self.failure_flag:
            # if we had met error
            return -1

        # output the predicted best arm
        assert len(self.survive_arms) <= 2
        # numeric error might lead to len(self.survive_arms)=2
        # which means in the function self.observation, self.consumption < self.ration_q_[self.q] - 1 might still hold
        # but the gap is roughly 1e-8
        # Then we need to conduct the final elimination here

        if len(self.survive_arms) == 2:
            self.eliminate()
        best_arm = self.survive_arms[0]
        return best_arm

    def eliminate(self):
        if self.failure_flag:
            return

        if len(self.survive_arms) == 1:
            return

        if self.q == 0:
            for ii in self.survive_arms:
                if len(self.reward_[ii]) == 0:
                    self.failure_flag = True
                    return

        # pulling_times = self.t_q // len(self.survive_arms)
        pulling_times = np.min([self.pulling_times_[ii - 1] for ii in self.survive_arms]).astype(np.int32)
        mean_reward = np.array([np.mean(self.reward_[ii][:pulling_times]) for ii in self.survive_arms])
        random_values = np.random.uniform(low=0.0, high=1.0, size=len(self.survive_arms))

        # sort the mean reward with descending order
        # sort_order = np.argsort(mean_reward)[::-1]
        sort_order = np.lexsort((random_values, mean_reward))[::-1]  # avoid preference of larger index

        # clear the memory
        self.t_q = 0
        # for arm_index in self.survive_arms:
        #     self.demand_[arm_index] = list()
        #     self.reward_[arm_index] = list()

        # eliminate half of the arms
        self.survive_arms = np.array(
            [self.survive_arms[ii] for ii in sort_order[: int(np.ceil(len(self.survive_arms) / 2))]]
        )
        self.survive_arms = np.sort(self.survive_arms)

        if len(self.survive_arms) == 1:
            return

        self.ration_q_[:, self.q + 1] = self.C / np.ceil(np.log2(self.K)) + self.ration_q_[:, self.q] - self.consumption
        self.q = self.q + 1
        self.consumption = 0


class SequentialHalving_FixedBudget_Agent:
    def __init__(self, K=2, budget=10) -> None:
        """The classic Fixed Budget SH algorithm, in this case, the consumption is always 1

        Args:
            K (int, optional): Total number of arms. Defaults to 2.
            Budget (int, optional): Available budget Defaults to 10.
        """
        assert np.ceil(np.log2(K)) * K <= budget, "budget is not enough"

        self.K = K
        self.budget = budget
        self.t = 0  # index of round
        self.t_q = 0  # index of round in each phase
        self.q = 0  # index of phase
        self.arm_ = list()  # record the action in each epoch

        self.reward_ = dict()  # record the observed reward of arms in each epoch
        # self.mean_reward_ = np.zeros(K)  # record the mean reward
        for arm_index in range(1, K + 1):
            # for each arm, create a list
            # when we enter a new epoch, we clear the existing memory
            self.reward_[arm_index] = list()
        self.pulling_times_ = np.zeros(K)

        self.total_reward_ = dict()
        for arm_index in range(1, K + 1):
            # for each arm, create a list
            # but we will not clear the memory
            self.total_reward_[arm_index] = list()

        self.survive_arms = list(range(1, K + 1))

        self.pulling_list = []
        for kk in range(1, K + 1):
            self.pulling_list = self.pulling_list + [kk] * int(
                np.floor(self.budget / np.ceil(np.log2(self.K)) / self.K)
            )

        self.complete = False  # mark whether the algorithm complete or not

    def action(self):
        # return the pulling arm in this epoch
        assert len(self.pulling_list) > 0
        arm = self.pulling_list[0]
        self.pulling_list.pop(0)
        self.arm_.append(arm)
        return arm

    def observe(self, reward):
        # record the arms in this phase
        self.reward_[self.arm_[-1]].append(reward)
        self.pulling_times_[self.arm_[-1] - 1] += 1

        # record the arms in this overall array
        self.total_reward_[self.arm_[-1]].append(reward)

        # update the index of rounds
        self.t = self.t + 1
        self.t_q = self.t_q + 1

        if len(self.survive_arms) == 1:
            self.complete = True
            return

        # check whether conduct the elimination
        if len(self.pulling_list) == 0:
            # we need to make sure all the arm share the same pulling times
            pulling_times = self.t_q // len(self.survive_arms)
            mean_reward = np.array([np.mean(self.reward_[ii][:pulling_times]) for ii in self.survive_arms])
            random_values = np.random.uniform(low=0.0, high=1.0, size=len(self.survive_arms))

            # sort the mean reward with descending order
            # sort_order = np.argsort(mean_reward)[::-1]
            sort_order = np.lexsort((random_values, mean_reward))[::-1]  # avoid preference of larger index
            self.survive_arms = np.array(
                [self.survive_arms[ii] for ii in sort_order[: int(np.ceil(len(self.survive_arms) / 2))]]
            )
            self.survive_arms = np.sort(self.survive_arms)

            # generate pulling list
            self.pulling_list = []
            for arm in self.survive_arms:
                self.pulling_list = self.pulling_list + [arm] * int(
                    np.floor(self.budget / np.ceil(np.log2(self.K)) / len(self.survive_arms))
                )

            # clear the memory
            self.t_q = 0
            for arm_index in range(1, self.K + 1):
                self.reward_[arm_index] = list()
            self.consumption = 0
            self.q = self.q + 1

    def predict(self):
        # output the predicted best arm
        assert len(self.survive_arms) == 1
        best_arm = self.survive_arms[0]
        return best_arm

    def if_complete(self):
        return self.complete


class DoublingSequentialHalving_Agent:
    # Apply doubling trick on Fixed budget sequential halving, to make it an anytime algorithm
    # when the current SH oracle doesn't terminate, always return the recommended arm in the last SH oracle
    def __init__(self, K=2, C=np.array([10]), L=1) -> None:
        """Construct an instance of Sequential Halving policy

        Args:
            K (int, optional): Total number of arms. Defaults to 2.
            C (np.ndarray, optional): Available initial resource. Defaults to np.array([10]).
            L (int, optional): The number of resources. Defaults to 1.
        """
        self.K = K
        self.C = C
        self.L = L
        self.t = 0  # index of round
        self.arm_ = list()  # record the action in each epoch

        self.pulling_times_ = np.zeros(K)

        self.total_demand_ = dict()
        self.total_reward_ = dict()
        for arm_index in range(1, K + 1):
            # for each arm, create a list
            # but we will not clear the memory
            self.total_demand_[arm_index] = list()
            self.total_reward_[arm_index] = list()
        self.total_consumption = np.zeros(L)  # the total consumption of all the phase

        # setup the SH oracle
        self.budget = K * np.ceil(np.log2(K))
        self.SH_oracle = SequentialHalving_FixedBudget_Agent(K=K, budget=self.budget)

        # predicted arm
        self.J = 1  # with out any information, we predict arm 1 as the best arm
        self.J_ = [1]  # history of predicted arm

    def action(self):
        # return the pulling arm in this epoch
        arm = self.SH_oracle.action()
        self.arm_.append(arm)
        return arm

    def observe(self, demand, reward):
        # record the arms in this phase
        self.pulling_times_[self.arm_[-1] - 1] += 1

        # record the arms in this overall array
        self.total_reward_[self.arm_[-1]].append(reward)
        self.total_demand_[self.arm_[-1]].append(demand)

        # update the consumption
        self.total_consumption = self.total_consumption + demand

        # update the index of rounds
        self.t = self.t + 1

        # update the history of SH oracle
        self.SH_oracle.observe(reward=reward)
        if self.SH_oracle.if_complete():
            self.J = self.SH_oracle.predict()
            self.J_.append(self.J)
            self.budget *= 2
            self.SH_oracle = SequentialHalving_FixedBudget_Agent(K=self.K, budget=self.budget)

    def predict(self):
        # output the predicted best arm
        best_arm = self.J
        return best_arm


class AT_LUCB_Agent:
    def __init__(self, K=2, C=np.array([10]), L=1, delta_1=0.5, alpha=0.99, epsilon=0.0, m=1) -> None:
        """Construct an instance of Anytime Lower and Upper Confidence Bound
        The algorithm came from June&Nowak2016, top m identification problem

        Args:
            K (int, optional): Total number of arms. Defaults to 2.
            C (int, optional): Available initial resource. Defaults to np.array([10]).
            L (int, optional): The number of resources. Defaults to 1.
            delta_1 (float, optional): Confidence Level. Defaults to 0.5, 1/200 <= delta_1 <= n
            alpha (float, optional): Discount Factor. Defaults to 0.99., 1/50 <= alpha < 1
            epsilon (float, optional): Tolerance of error. Defaults to 0.0.
            m (int, optional): we aim to find top m arms. Default value is 1
        """
        assert len(C) == L, "number of resources doesn't match"

        self.K = K
        self.C = C
        self.L = L
        self.delta_1 = delta_1
        self.alpha = alpha
        self.epsilon = epsilon
        self.m = m

        self.t = 0  # index of round
        self.t_for_delta = 1  # in the algorithm, t_for_delta increases only after 2 pulls
        self.total_consumption = np.zeros(L)  # record the overall consumption in each phase
        self.S = 1  # S(0)
        self.S_ = list()  # record the chaning history of S
        self.J = np.arange(1, self.m + 1)
        self.J_ = list()  # record the chaning history of J

        self.pulling_times_ = np.zeros(K)
        self.demand_ = np.zeros((L, K))  # total consumption
        self.reward_ = np.zeros(K)  # total reward
        self.mean_reward_ = np.zeros(K)
        self.arm_ = list()  # record the action in each epoch

        self.pulling_list = list(np.arange(1, K + 1))
        # Each time, this algorithm will generate two arms to pull

    def action(self):
        # return the pulling arm in this epoch
        assert len(self.pulling_list) > 0, "failed to generate pulling arms"
        arm = self.pulling_list.pop(0)
        self.arm_.append(arm)
        return arm

    def observe(self, demand, reward):
        # record the arms in this round
        arm_index = self.arm_[-1] - 1

        # record the arms in this round
        self.reward_[arm_index] += reward
        self.demand_[:, arm_index] += demand
        self.total_consumption += demand
        self.pulling_times_[arm_index] += 1
        self.mean_reward_[arm_index] = self.reward_[arm_index] / self.pulling_times_[arm_index]

        # update the index of rounds
        self.t = self.t + 1

        # if self.pulling_list is empty, we need to regenerate arms
        if len(self.pulling_list) == 0:
            delta_s_t_1 = self.delta_1 * self.alpha ** (self.S - 1)
            if self.Term(self.t_for_delta, delta_s_t_1, self.epsilon):
                ## update S(t) and new pulling arms
                self.S, ht_star_delta, lt_star_delta, self.J = self.UpdateS(self.S)
                self.pulling_list.append(ht_star_delta)
                self.pulling_list.append(lt_star_delta)
                self.t_for_delta += 1
            else:
                if self.S == 1:
                    self.J = np.argpartition(self.mean_reward_, -self.m)[-self.m :] + 1

                ## generate new pulling arms
                ht_star_delta, lt_star_delta = self.Get_LUCB_l_h_star(
                    self.t_for_delta, self.delta_1 * self.alpha ** (self.S - 1)
                )
                self.pulling_list.append(ht_star_delta)
                self.pulling_list.append(lt_star_delta)
                self.t_for_delta += 1

            self.S_.append(self.S)  # record the history
            self.J_.append(self.J)

    def predict(self, m=1):
        # output the predicted best arm
        if m == 1:
            arm = np.argmax(self.mean_reward_) + 1
        else:
            arm = self.J
        return arm

    def UpdateS(self, S):
        # This fcuntion is used to accelerate the step to update S

        # calculate set High^t
        hight = np.argpartition(self.mean_reward_, -self.m)[-self.m :]
        mean_reward_for_U = np.copy(self.mean_reward_)
        mean_reward_for_L = np.copy(self.mean_reward_[hight])
        mean_reward_for_U[hight] = -np.inf

        # update S
        tempS = S + 1
        while True:
            delta = self.delta_1 * self.alpha ** (tempS - 1)
            confidence = np.sqrt(
                1 / self.pulling_times_ / 2 * (np.log(5 * self.K / 4 / delta) + 4 * np.log(self.t_for_delta))
            )
            gap = np.max(mean_reward_for_U + confidence) - np.min(mean_reward_for_L - confidence[hight])
            if gap < self.epsilon:
                break
            tempS += 1

        # calculate ht_star_delta and lt_star_delta
        delta = self.delta_1 * self.alpha ** (tempS - 1)
        confidence = np.sqrt(
            1 / self.pulling_times_ / 2 * (np.log(5 * self.K / 4 / delta) + 4 * np.log(self.t_for_delta))
        )
        U = self.mean_reward_ + confidence
        L = self.mean_reward_[hight] - confidence[hight]
        U[hight] = -np.inf
        lt_star_delta = np.argmax(U) + 1
        ht_star_delta = hight[np.argmin(L)] + 1

        return tempS, ht_star_delta, lt_star_delta, hight + 1

    def Get_LUCB_l_h_star(self, t, delta):
        # auxiliary function, help to calculate $U^t_a(\delta), L^t_a(\delta), h^t_*(\delta), l^t_*(\delta)$

        # calculate set High^t
        hight = np.argpartition(self.mean_reward_, -self.m)[-self.m :]

        delta = self.delta_1 * self.alpha ** (self.S - 1)
        confidence = np.sqrt(1 / self.pulling_times_ / 2 * (np.log(5 * self.K / 4 / delta) + 4 * np.log(t)))
        U = self.mean_reward_ + confidence
        L = self.mean_reward_[hight] - confidence[hight]
        U[hight] = -np.inf
        lt_star_delta = np.argmax(U) + 1
        ht_star_delta = hight[np.argmin(L)] + 1

        return ht_star_delta, lt_star_delta

    def Term(self, t, delta, epsilon):
        # auxiliary function
        # # use it to judge whether $U^t_{l^t_*(\delta)}(\delta)-L^t_{h^t_*(\delta)}(\delta)<epsilon$
        # ut_a_delta, lt_a_delta, ht_star_delta, lt_star_delta = self.Get_LUCB_l_h_star(t, delta)

        hight = np.argpartition(self.mean_reward_, -self.m)[-self.m :]
        mean_reward_for_U = np.copy(self.mean_reward_)
        mean_reward_for_L = np.copy(self.mean_reward_[hight])
        mean_reward_for_U[hight] = -np.inf

        delta = self.delta_1 * self.alpha ** (self.S - 1)
        confidence = np.sqrt(1 / self.pulling_times_ / 2 * (np.log(5 * self.K / 4 / delta) + 4 * np.log(t)))
        gap = np.max(mean_reward_for_U + confidence) - np.min(mean_reward_for_L - confidence[hight])

        return gap < epsilon
