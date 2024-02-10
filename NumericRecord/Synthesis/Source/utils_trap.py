from __future__ import annotations
import numpy as np


def Balanced_Trap_HalfConsumption_Fix32_MultiR_HL(K: int, L: int = 2, rlow=0.1, rmid=0.8, rhigh=0.9, dlow: float = 0.1, dhigh: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
    """Arms with higher reward consume less resource, but the mean reward and consumption only have 3 values

    Args:
        K (int): number of arms.
        L (int, optional): number of resources. Defaults to 2.
        rlow (float, optional): smaller value of mean reward. Defaults to 0.1.
        rmid (float, optional): middle value of mean reward. Defaults to 0.8.
        rhigh (float, optional): bigger value of mean reward. Defaults to 0.9.
        dlow (float, optional): smaller value of mean consumption. Defaults to 0.1.
        dhigh (float, optional): bigger value of mean consumption. Defaults to 0.9.
    """
    fixed_competitor_num = 32
    assert K > fixed_competitor_num

    reward = np.zeros(K)
    reward[0] = rhigh
    reward[1:fixed_competitor_num] = rmid
    reward[fixed_competitor_num:] = rlow

    demand = np.zeros((L, K))
    demand[:, 0] = dlow
    demand[:, 1 : K // 2] = dlow
    demand[:, K // 2 :] = dhigh
    return reward, demand


def Balanced_Trap_HalfConsumption_Fix32_MultiR_Mixture(K: int, L: int = 2, rlow=0.1, rmid=0.8, rhigh=0.9, dlow: float = 0.1, dhigh: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
    """Arms with higher reward consume less resource, but the mean reward and consumption only have 3 values

    Args:
        K (int): number of arms.
        L (int, optional): number of resources. Defaults to 2.
        rlow (float, optional): smaller value of mean reward. Defaults to 0.1.
        rmid (float, optional): middle value of mean reward. Defaults to 0.8.
        rhigh (float, optional): bigger value of mean reward. Defaults to 0.9.
        dlow (float, optional): smaller value of mean consumption. Defaults to 0.1.
        dhigh (float, optional): bigger value of mean consumption. Defaults to 0.9.
    """
    fixed_competitor_num = 32
    assert K > fixed_competitor_num

    reward = np.zeros(K)
    reward[0] = rhigh
    reward[1:fixed_competitor_num] = rmid
    reward[fixed_competitor_num:] = rlow

    demand = np.zeros((L, K))
    demand[0 : L // 2, 0] = dlow
    demand[0 : L // 2, 1 : K // 2] = dlow
    demand[0 : L // 2, K // 2 :] = dhigh

    demand[L // 2 :, 0] = dhigh
    demand[L // 2 :, 1 : K // 2] = dhigh
    demand[L // 2 :, K // 2 :] = dlow
    return reward, demand


def Balanced_Trap_HalfConsumption_Fix32_MultiR_HH(K: int, L: int = 2, rlow=0.1, rmid=0.8, rhigh=0.9, dlow: float = 0.1, dhigh: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
    """Arms with higher reward consume more resource, but the mean reward and consumption only have 3 values

    Args:
        K (int): number of arms
        L (int, optional): number of resources. Defaults to 2.
        rlow (float, optional): smaller value of mean reward. Defaults to 0.1.
        rmid (float, optional): middle value of mean reward. Defaults to 0.8.
        rhigh (float, optional): bigger value of mean reward. Defaults to 0.9.
        dlow (float, optional): smaller value of mean consumption. Defaults to 0.1.
        dhigh (float, optional): bigger value of mean consumption. Defaults to 0.9.
    """
    fixed_competitor_num = 32
    assert K > fixed_competitor_num

    reward = np.zeros(K)
    reward[0] = rhigh
    reward[1:fixed_competitor_num] = rmid
    reward[fixed_competitor_num:] = rlow

    demand = np.zeros((L, K))
    demand[:, 0 : K // 2] = dhigh
    demand[:, K // 2 :] = dlow
    return reward, demand


def Balanced_Trap_HalfConsumption_Fix64_MultiR_HL(K: int, L: int = 2, rlow=0.1, rmid=0.8, rhigh=0.9, dlow: float = 0.1, dhigh: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
    """Arms with higher reward consume less resource, but the mean reward and consumption only have 3 values

    Args:
        K (int): number of arms.
        L (int, optional): number of resources. Defaults to 2.
        rlow (float, optional): smaller value of mean reward. Defaults to 0.1.
        rmid (float, optional): middle value of mean reward. Defaults to 0.8.
        rhigh (float, optional): bigger value of mean reward. Defaults to 0.9.
        dlow (float, optional): smaller value of mean consumption. Defaults to 0.1.
        dhigh (float, optional): bigger value of mean consumption. Defaults to 0.9.
    """
    fixed_competitor_num = 64
    assert K > fixed_competitor_num

    reward = np.zeros(K)
    reward[0] = rhigh
    reward[1:fixed_competitor_num] = rmid
    reward[fixed_competitor_num:] = rlow

    demand = np.zeros((L, K))
    demand[:, 0] = dlow
    demand[:, 1 : K // 2] = dlow
    demand[:, K // 2 :] = dhigh
    return reward, demand


def Balanced_Trap_HalfConsumption_Fix64_MultiR_Mixture(K: int, L: int = 2, rlow=0.1, rmid=0.8, rhigh=0.9, dlow: float = 0.1, dhigh: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
    """Arms with higher reward consume less resource, but the mean reward and consumption only have 3 values

    Args:
        K (int): number of arms.
        L (int, optional): number of resources. Defaults to 2.
        rlow (float, optional): smaller value of mean reward. Defaults to 0.1.
        rmid (float, optional): middle value of mean reward. Defaults to 0.8.
        rhigh (float, optional): bigger value of mean reward. Defaults to 0.9.
        dlow (float, optional): smaller value of mean consumption. Defaults to 0.1.
        dhigh (float, optional): bigger value of mean consumption. Defaults to 0.9.
    """
    fixed_competitor_num = 64
    assert K > fixed_competitor_num

    reward = np.zeros(K)
    reward[0] = rhigh
    reward[1:fixed_competitor_num] = rmid
    reward[fixed_competitor_num:] = rlow

    demand = np.zeros((L, K))
    demand[0 : L // 2, 0] = dlow
    demand[0 : L // 2, 1 : K // 2] = dlow
    demand[0 : L // 2, K // 2 :] = dhigh

    demand[L // 2 :, 0] = dhigh
    demand[L // 2 :, 1 : K // 2] = dhigh
    demand[L // 2 :, K // 2 :] = dlow
    return reward, demand


def Balanced_Trap_HalfConsumption_Fix64_MultiR_HH(K: int, L: int = 2, rlow=0.1, rmid=0.8, rhigh=0.9, dlow: float = 0.1, dhigh: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
    """Arms with higher reward consume more resource, but the mean reward and consumption only have 3 values

    Args:
        K (int): number of arms
        L (int, optional): number of resources. Defaults to 2.
        rlow (float, optional): smaller value of mean reward. Defaults to 0.1.
        rmid (float, optional): middle value of mean reward. Defaults to 0.8.
        rhigh (float, optional): bigger value of mean reward. Defaults to 0.9.
        dlow (float, optional): smaller value of mean consumption. Defaults to 0.1.
        dhigh (float, optional): bigger value of mean consumption. Defaults to 0.9.
    """
    fixed_competitor_num = 64
    assert K > fixed_competitor_num

    reward = np.zeros(K)
    reward[0] = rhigh
    reward[1:fixed_competitor_num] = rmid
    reward[fixed_competitor_num:] = rlow

    demand = np.zeros((L, K))
    demand[:, 0 : K // 2] = dhigh
    demand[:, K // 2 :] = dlow
    return reward, demand
