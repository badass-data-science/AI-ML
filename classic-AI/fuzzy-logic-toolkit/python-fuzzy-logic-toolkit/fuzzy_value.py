"""
Copyright (C) 2002, 2003  E. M. Williams

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.
"""

from linguistic_domain import LinguisticDomain


class FuzzyValue:

    def __init__(self, linguistic_domain: LinguisticDomain):
        self._linguistic_domain = linguistic_domain
        self._values: list[float] = [0.0] * linguistic_domain.get_number_of_linguistic_sets()

    def set_crisp_value(self, x: float) -> None:
        self._values = self._linguistic_domain.get_membership(x)

    def get_crisp_value(self) -> float:
        """Defuzzify using the center-of-maximum method (weighted sum of typical values)."""
        sets = self._linguistic_domain.get_linguistic_set_list()
        return sum(
            sets[i].get_membership_function().get_typical_value() * self._values[i]
            for i in range(len(sets))
        )

    def clear(self) -> None:
        self._values = [0.0] * len(self._values)

    def set_set_membership(self, x: float, name: str) -> None:
        sets = self._linguistic_domain.get_linguistic_set_list()
        for i, s in enumerate(sets):
            if s.get_name() == name:
                self._values[i] = x
                return

    def get_set_membership(self, name: str) -> float:
        sets = self._linguistic_domain.get_linguistic_set_list()
        for i, s in enumerate(sets):
            if s.get_name() == name:
                return self._values[i]
        return 0.0

    def AND(self, name_a: str, other: "FuzzyValue", name_b: str) -> float:
        """Fuzzy AND: minimum of the two named set memberships."""
        return min(self.get_set_membership(name_a), other.get_set_membership(name_b))

    def OR_set_set_membership(self, x: float, name: str) -> None:
        """Fuzzy OR: set named membership to max of x and its current value."""
        sets = self._linguistic_domain.get_linguistic_set_list()
        for i, s in enumerate(sets):
            if s.get_name() == name:
                if x > self._values[i]:
                    self._values[i] = x
                return
