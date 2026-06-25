"""
Copyright (C) 2002, 2003  E. M. Williams

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.
"""

from abc import ABC, abstractmethod


class MembershipFunction(ABC):

    def __init__(self):
        self._typical_value: float = 0.0

    @abstractmethod
    def get_membership(self, x: float) -> float:
        pass

    def get_typical_value(self) -> float:
        return self._typical_value


class StandardMBF_Z(MembershipFunction):
    """Z-shaped: 1 below min, linear decrease to 0 at max."""

    def __init__(self, min: float, max: float):
        super().__init__()
        self._min = min
        self._max = max
        self._typical_value = min
        self._slope = -1.0 / (max - min)
        self._y_intercept = -self._slope * min + 1.0

    def get_membership(self, x: float) -> float:
        if x <= self._min:
            return 1.0
        if x <= self._max:
            return self._slope * x + self._y_intercept
        return 0.0


class StandardMBF_S(MembershipFunction):
    """S-shaped: 0 below min, linear rise to 1 at max."""

    def __init__(self, min: float, max: float):
        super().__init__()
        self._min = min
        self._max = max
        self._typical_value = max
        self._slope = 1.0 / (max - min)
        self._y_intercept = -self._slope * min

    def get_membership(self, x: float) -> float:
        if x <= self._min:
            return 0.0
        if x <= self._max:
            return self._slope * x + self._y_intercept
        return 1.0


class StandardMBF_Lambda(MembershipFunction):
    """Lambda (triangular): rises from min to mid, falls from mid to max."""

    def __init__(self, min: float, mid: float, max: float):
        super().__init__()
        self._min = min
        self._mid = mid
        self._max = max
        self._typical_value = mid
        self._slope_up = 1.0 / (mid - min)
        self._y_intercept_up = -self._slope_up * min
        self._slope_down = -1.0 / (max - mid)
        self._y_intercept_down = -self._slope_down * mid + 1.0

    def get_membership(self, x: float) -> float:
        if x <= self._min:
            return 0.0
        if x <= self._mid:
            return self._slope_up * x + self._y_intercept_up
        if x <= self._max:
            return self._slope_down * x + self._y_intercept_down
        return 0.0


class StandardMBF_Pi(MembershipFunction):
    """Pi (trapezoidal): rises from min to lower_mid, flat to higher_mid, falls to max."""

    def __init__(self, min: float, lower_mid: float, higher_mid: float, max: float):
        super().__init__()
        self._min = min
        self._lower_mid = lower_mid
        self._higher_mid = higher_mid
        self._max = max
        self._typical_value = (lower_mid + higher_mid) / 2.0
        self._slope_up = 1.0 / (lower_mid - min)
        self._y_intercept_up = -self._slope_up * min
        self._slope_down = -1.0 / (max - higher_mid)
        self._y_intercept_down = -self._slope_down * higher_mid + 1.0

    def get_membership(self, x: float) -> float:
        if x <= self._min:
            return 0.0
        if x <= self._lower_mid:
            return self._slope_up * x + self._y_intercept_up
        if x <= self._higher_mid:
            return 1.0
        if x <= self._max:
            return self._slope_down * x + self._y_intercept_down
        return 0.0
