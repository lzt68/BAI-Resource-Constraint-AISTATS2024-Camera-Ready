# The source file is to generate different synthesis experiment setting
# Calculate the H_1, H_2 and modified H_1, H_2
from __future__ import annotations
import numpy as np
from tqdm import tqdm


def One_suboptimal_MultiR_HL(
    K: int, L: int = 2, rlow: float = 0.8, rhigh: float = 0.9, dlow: float = 0.1, dhigh: float = 0.9
) -> tuple[np.ndarray, np.ndarray]:
    """Arms with higher reward consume less resource, but the mean reward and consumption only have two values

    Args:
        K (int): number of arms
        L (int, optional): number of resources. Defaults to 2.
        rlow (float, optional): smaller value of mean reward. Defaults to 0.8.
        rhigh (float, optional): bigger value of mean reward. Defaults to 0.9.
        dlow (float, optional): smaller value of mean consumption. Defaults to 0.1.
        dhigh (float, optional): bigger value of mean consumption. Defaults to 0.9.

    Returns:
        tuple[np.ndarray, np.ndarray]: ndarray for reward, demand
    """
    reward = np.ones(K) * rlow
    reward[0] = rhigh

    demand = np.zeros((L, K))
    demand[:, 0 : K // 2] = dlow
    demand[:, K // 2 :] = dhigh
    return reward, demand


def One_suboptimal_MultiR_Mixture(
    K: int, L: int = 2, rlow: float = 0.8, rhigh: float = 0.9, dlow: float = 0.1, dhigh: float = 0.9
) -> tuple[np.ndarray, np.ndarray]:
    """Arms with higher reward consume less resource, but the mean reward and consumption only have two values

    Args:
        K (int): number of arms
        L (int, optional): number of resources. Defaults to 2.
        rlow (float, optional): smaller value of mean reward. Defaults to 0.8.
        rhigh (float, optional): bigger value of mean reward. Defaults to 0.9.
        dlow (float, optional): smaller value of mean consumption. Defaults to 0.1.
        dhigh (float, optional): bigger value of mean consumption. Defaults to 0.9.

    Returns:
        tuple[np.ndarray, np.ndarray]: ndarray for reward, demand
    """
    reward = np.ones(K) * rlow
    reward[0] = rhigh

    demand = np.zeros((L, K))
    demand[0 : L // 2, 0 : K // 2] = dlow
    demand[0 : L // 2, K // 2 :] = dhigh
    demand[L // 2 :, 0 : K // 2] = dhigh
    demand[L // 2 :, K // 2 :] = dlow
    return reward, demand


def One_suboptimal_MultiR_HH(
    K: int, L: int = 2, rlow: float = 0.8, rhigh: float = 0.9, dlow: float = 0.1, dhigh: float = 0.9
) -> tuple[np.ndarray, np.ndarray]:
    """Arms with higher reward consume more resource, but the mean reward and consumption only have two values

    Args:
        K (int): number of arms
        L (int, optional): number of resources. Defaults to 2.
        rlow (float, optional): smaller value of mean reward. Defaults to 0.8.
        rhigh (float, optional): bigger value of mean reward. Defaults to 0.9.
        dlow (float, optional): smaller value of mean consumption. Defaults to 0.1.
        dhigh (float, optional): bigger value of mean consumption. Defaults to 0.9.

    Returns:
        tuple[np.ndarray, np.ndarray]: ndarray for reward, demand
    """
    reward = np.ones(K) * rlow
    reward[0] = rhigh
    demand = np.zeros((L, K))
    demand[:, 0 : K // 2] = dhigh
    demand[:, K // 2 :] = dlow
    return reward, demand


def Poly_MultiR_HL(K: int, L: int = 2, dlow: float = 0.1, dhigh: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
    """The reward is $r_1=0.9$, $r_i=0.9(1-\sqrt{\frac{i}{n}}), i\ge 2$

    Args:
        K (int): number of arms
        L (int, optional): number of resources. Defaults to 2.
        rlow (float, optional): smaller value of mean reward. Defaults to 0.1.
        rhigh (float, optional): bigger value of mean reward. Defaults to 0.9.
        dlow (float, optional): smaller value of mean consumption. Defaults to 0.1.
        dhigh (float, optional): bigger value of mean consumption. Defaults to 0.9.

    Returns:
        tuple[np.ndarray, np.ndarray]: reward, demand
    """
    reward = np.zeros(K)
    reward[0] = 0.9
    reward[1:] = 0.9 * (1 - np.sqrt(np.arange(2, K + 1) / K))

    demand = np.zeros((L, K))
    demand[:, 0 : K // 2] = dlow
    demand[:, K // 2 :] = dhigh

    return reward, demand


def Poly_MultiR_Mixture(
    K: int, L: int = 2, rlow: float = 0.1, rhigh: float = 0.9, dlow: float = 0.1, dhigh: float = 0.9
) -> tuple[np.ndarray, np.ndarray]:
    """The reward is $r_1=0.9$, $r_i=0.9(1-\sqrt{\frac{i}{n}}), i\ge 2$

    Args:
        K (int): number of arms
        L (int, optional): number of resources. Defaults to 2.
        rlow (float, optional): smaller value of mean reward. Defaults to 0.1.
        rhigh (float, optional): bigger value of mean reward. Defaults to 0.9.
        dlow (float, optional): smaller value of mean consumption. Defaults to 0.1.
        dhigh (float, optional): bigger value of mean consumption. Defaults to 0.9.

    Returns:
        tuple[np.ndarray, np.ndarray]: reward, demand
    """
    reward = np.zeros(K)
    reward[0] = 0.9
    reward[1:] = 0.9 * (1 - np.sqrt(np.arange(2, K + 1) / K))

    demand = np.zeros((L, K))
    demand[0 : L // 2, 0 : K // 2] = dlow
    demand[0 : L // 2, K // 2 :] = dhigh
    demand[L // 2 :, 0 : K // 2] = dhigh
    demand[L // 2 :, K // 2 :] = dlow

    return reward, demand


def Poly_MultiR_HH(
    K: int, L: int = 2, rlow: float = 0.1, rhigh: float = 0.9, dlow: float = 0.1, dhigh: float = 0.9
) -> tuple[np.ndarray, np.ndarray]:
    """The reward is $r_1=0.9$, $r_i=0.9(1-\sqrt{\frac{i}{n}}), i\ge 2$

    Args:
        K (int): number of arms
        L (int, optional): number of resources. Defaults to 2.
        rlow (float, optional): smaller value of mean reward. Defaults to 0.1.
        rhigh (float, optional): bigger value of mean reward. Defaults to 0.9.
        dlow (float, optional): smaller value of mean consumption. Defaults to 0.1.
        dhigh (float, optional): bigger value of mean consumption. Defaults to 0.9.

    Returns:
        tuple[np.ndarray, np.ndarray]: reward, demand
    """
    reward = np.zeros(K)
    reward[0] = 0.9
    reward[1:] = 0.9 * (1 - np.sqrt(np.arange(2, K + 1) / K))

    demand = np.zeros((L, K))
    demand[:, 0 : K // 2] = dhigh
    demand[:, K // 2 :] = dlow

    return reward, demand


def Geometry_MultiR_HL(
    K: int, L: int = 2, rlow: float = 0.1, rhigh: float = 0.9, dlow: float = 0.1, dhigh: float = 0.9
) -> tuple[np.ndarray, np.ndarray]:
    """The reward is a geometric sequence, arm with higher reward consume less

    Args:
        K (int): number of arms
        L (int, optional): number of resources. Defaults to 2.
        rlow (float, optional): smaller value of mean reward. Defaults to 0.1.
        rhigh (float, optional): bigger value of mean reward. Defaults to 0.9.
        dlow (float, optional): smaller value of mean consumption. Defaults to 0.1.
        dhigh (float, optional): bigger value of mean consumption. Defaults to 0.9.

    Returns:
        tuple[np.ndarray, np.ndarray]: _description_
    """
    reward = np.geomspace(rhigh, rlow, K)

    demand = np.zeros((L, K))
    demand[:, 0 : K // 2] = dlow
    demand[:, K // 2 :] = dhigh

    return reward, demand


def Geometry_MultiR_Mixture(
    K: int, L: int = 2, rlow: float = 0.1, rhigh: float = 0.9, dlow: float = 0.1, dhigh: float = 0.9
) -> tuple[np.ndarray, np.ndarray]:
    """The reward is a geometric sequence, arm with higher reward consume less

    Args:
        K (int): number of arms
        L (int, optional): number of resources. Defaults to 2.
        rlow (float, optional): smaller value of mean reward. Defaults to 0.1.
        rhigh (float, optional): bigger value of mean reward. Defaults to 0.9.
        dlow (float, optional): smaller value of mean consumption. Defaults to 0.1.
        dhigh (float, optional): bigger value of mean consumption. Defaults to 0.9.

    Returns:
        tuple[np.ndarray, np.ndarray]: _description_
    """
    reward = np.geomspace(rhigh, rlow, K)
    demand = np.zeros((L, K))
    demand[0 : L // 2, 0 : K // 2] = dlow
    demand[0 : L // 2, K // 2 :] = dhigh
    demand[L // 2 :, 0 : K // 2] = dhigh
    demand[L // 2 :, K // 2 :] = dlow

    return reward, demand


def Geometry_MultiR_HH(
    K: int, L: int = 2, rlow: float = 0.1, rhigh: float = 0.9, dlow: float = 0.1, dhigh: float = 0.9
) -> tuple[np.ndarray, np.ndarray]:
    """The reward is a geometric sequence, arm with higher reward consume more

    Args:
        K (int): number of arms
        L (int, optional): number of resources. Defaults to 2.
        rlow (float, optional): smaller value of mean reward. Defaults to 0.1.
        rhigh (float, optional): bigger value of mean reward. Defaults to 0.9.
        dlow (float, optional): smaller value of mean consumption. Defaults to 0.1.
        dhigh (float, optional): bigger value of mean consumption. Defaults to 0.9.

    Returns:
        tuple[np.ndarray, np.ndarray]: _description_
    """
    reward = np.geomspace(rhigh, rlow, K)
    demand = np.zeros((L, K))
    demand[:, 0 : K // 2] = dhigh
    demand[:, K // 2 :] = dlow

    return reward, demand


def Get_H1(reward: np.ndarray, demand: np.ndarray) -> float:
    # assume r_1>r_2>\cdots>r_K, $\{d_{(k)}\}_{k=1}^{K}$ is a permutation of $\{d_k\}_{k=1}^{K}$
    # $d_{(1)}\ge d_{(2)}\ge \cdots \ge d_{(K)}$
    # the classic defintion of $H_1=\frac{d_{(1)}}{(r_{(1)}-r_{(2)})^2}+\sum_{k=1}^K \frac{d_{(k)}}{(r_{(1)}-r_{(k)})^2}$
    reward = np.sort(reward)[::-1]
    demand = np.sort(demand)[::-1]
    gap = reward[0] - reward
    gap[0] = reward[0] - reward[1]
    H1 = np.sum(demand / gap**2)
    return H1


def Get_H2(reward: np.ndarray, demand: np.ndarray) -> float:
    # assume r_1>r_2>\cdots>r_K, $\{d_{(k)}\}_{k=1}^{K}$ is a permutation of $\{d_k\}_{k=1}^{K}$
    # $d_{(1)}\ge d_{(2)}\ge \cdots \ge d_{(K)}$
    # the classic defintion of $H_2=\max_{2\le k\le K}\frac{\sum_{i=1}^k d_{(i)}}{(r_1-r_k)^2}$
    reward = np.sort(reward)[::-1]
    demand = np.sort(demand)[::-1]
    cumsum_demand = np.cumsum(demand)
    gap = reward[0] - reward
    gap[0] = reward[0] - reward[1]
    ratio = cumsum_demand / gap**2
    H2 = np.max(ratio[1:])
    return H2


def Get_Modified_H2(reward: np.ndarray, demand: np.ndarray) -> float:
    # assume r_1>r_2>\cdots>r_K, $\{d_{(k)}\}_{k=1}^{K}$ is a permutation of $\{d_k\}_{k=1}^{K}$
    # $d_{(1)}\ge d_{(2)}\ge \cdots \ge d_{(K)}$
    # the classic defintion of $H_2=\max_{2\le k\le K}\frac{\sum_{i=1}^k d_{(i)}}{(r_1-r_k)^2}$
    # the modified defintion of $H_2=\max_{2\le k\le K}\frac{\sum_{i=1}^k f(d_{(i)})}{(r_1-r_k)^2}$
    reward = np.sort(reward)[::-1]
    demand = np.sort(demand)[::-1]
    f = lambda x: np.e**2 * x if x >= 1 / np.e**2 else 2 * 1 / np.log(1 / x)
    demand_f = np.array([f(x) for x in demand])

    cumsum_demand = np.cumsum(demand_f)
    gap = reward[0] - reward
    gap[0] = reward[0] - reward[1]
    ratio = cumsum_demand / gap**2
    modified_H2 = np.max(ratio[1:])
    return modified_H2


# -----------------------experiment oracle begin----------------------------------------
# each copy of this function will run n_experiment executions of experiments,
# while the function Experiment_Single will just run one single copy
def Experiment_MultiR(
    reward: np.ndarray,
    demand: np.ndarray,
    price: np.ndarray,
    r_or_p: bool,
    env_class,
    env_para: dict,
    agent_class,
    agent_para: dict,
    n_experiment: int,
    K: int,
    C: np.ndarray,
    L: int,
    random_seed: int = 0,
    disable_tqdm: bool = False,
    shuffle: bool = True,
):
    """Experiment oracle

    Args:
        reward (np.ndarray): the mean reward of each arm
        demand (np.ndarray): the mean consumption of each arm
        price (np.ndarray): the price of each arm
        r_or_p (bool): If true, means the env is Env_Correlated_Reward,
        env_class (_type_): the class of environment
        env_para (dict): extra parameter settings for the environment
        agent_class (_type_): the class of agent
        agent_para (dict): extra parameter settings for the agent
        n_experiment (int): execution times of independent experiment
        K (int) : arm numbers
        C (np.ndarray) : initial available resource
        L (int) : number of resources
        random_seed (int, optional) : random seed. default value is 0
        disable_tqdm (bool, optional): If true, mute the output of tqdm
        shuffle (bool, optional): If true, shffule the arm before the algorithm
    """
    assert len(C) == L, "number of resources doesn't match"

    best_arm_ = np.zeros(n_experiment)
    predict_arm_ = np.zeros(n_experiment)
    stopping_times_ = np.zeros(n_experiment)
    for exp_index in tqdm(range(n_experiment), disable=disable_tqdm):
        # different random_seed will generate different permutation
        np.random.seed(random_seed + exp_index)

        # permute the arm
        if shuffle:
            permuted_index = np.arange(K)
            np.random.shuffle(permuted_index)
            reward = reward[permuted_index]
            demand = demand[:, permuted_index]
            price = price[permuted_index]
        best_arm_[exp_index] = np.argmax(reward) + 1

        # set up the parameters of environments and agents, and define the env and agent
        if r_or_p:
            env_para["p_list"] = price
        else:
            env_para["r_list"] = reward
        env_para["d_list"] = demand
        env_para["K"] = K
        env_para["C"] = C
        env_para["L"] = L
        env_para["random_seed"] = random_seed + exp_index
        env = env_class(**env_para)

        agent_para["K"] = K
        agent_para["C"] = C
        agent_para["L"] = L
        agent = agent_class(**agent_para)

        # run the experiment
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

    return success_rate, std_success_rate, stopping_times_, predict_arm_, best_arm_


# -----------------------experiment oracle end----------------------------------------
