/*
    Copyright (C) 2002, 2003  E. M. Williams

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2.1 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

    The full text of the GNU Lesser General Public License is
    contained in the files COPYING and LICENSE.

*/

#include "LinguisticDomain.hh"

LinguisticDomain::LinguisticDomain(std::string nm)
{
  name = nm;
}

void LinguisticDomain::addLinguisticSet(LinguisticSet* l)
{
  linguistic_set_list.push_back(l);
}

std::vector<long double> LinguisticDomain::getMembership(long double x)
{
  std::vector<long double> membership_vector;
  for (int i = 0; i < linguistic_set_list.size(); i++)
    {
     membership_vector.push_back(linguistic_set_list[i]->getMembershipFunction()->getMembership(x));
    }
  return membership_vector;
}

long double LinguisticDomain::getMembership(long double x, std::string s)
{
  for (int i = 0; i < linguistic_set_list.size(); i++)
    {
      if (linguistic_set_list[i]->getName() == s)
	{
	  return linguistic_set_list[i]->getMembershipFunction()->getMembership(x);
	}
    }
  return 0.;
}

int LinguisticDomain::getNumberOfLinguisticSets()
{
  return linguistic_set_list.size();
}

std::vector<LinguisticSet*> LinguisticDomain::getLinguisticSetList()
{
  return linguistic_set_list;
}
