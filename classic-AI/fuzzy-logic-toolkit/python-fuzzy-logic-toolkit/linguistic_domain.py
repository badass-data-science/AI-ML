"""
Copyright (C) 2002, 2003  E. M. Williams

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.
"""

from linguistic_set import LinguisticSet


class LinguisticDomain:

    def __init__(self, name: str):
        self._name = name
        self._linguistic_set_list: list[LinguisticSet] = []

    def add_linguistic_set(self, linguistic_set: LinguisticSet) -> None:
        self._linguistic_set_list.append(linguistic_set)

    def get_membership(self, x: float, name: str | None = None) -> list[float] | float:
        if name is None:
            return [s.get_membership_function().get_membership(x) for s in self._linguistic_set_list]
        for s in self._linguistic_set_list:
            if s.get_name() == name:
                return s.get_membership_function().get_membership(x)
        return 0.0

    def get_number_of_linguistic_sets(self) -> int:
        return len(self._linguistic_set_list)

    def get_linguistic_set_list(self) -> list[LinguisticSet]:
        return self._linguistic_set_list
