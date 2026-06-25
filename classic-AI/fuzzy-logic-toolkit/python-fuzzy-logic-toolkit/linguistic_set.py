"""
Copyright (C) 2002, 2003  E. M. Williams

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.
"""

from membership_function import MembershipFunction


class LinguisticSet:

    def __init__(self, membership_function: MembershipFunction, name: str):
        self._membership_function = membership_function
        self._name = name

    def get_membership_function(self) -> MembershipFunction:
        return self._membership_function

    def get_name(self) -> str:
        return self._name
