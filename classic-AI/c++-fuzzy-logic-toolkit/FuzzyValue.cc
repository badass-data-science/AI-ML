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

#include <iostream>

#include "FuzzyValue.hh"

FuzzyValue::FuzzyValue(LinguisticDomain* ld)
{
  linguistic_domain = ld;
  long double dummy = 0;
  int i;
  for (i=0; i<ld->getNumberOfLinguisticSets(); i++)
    {
      values.push_back(dummy);
    }
}

void FuzzyValue::setCrispValue(long double x)
{
  int i;
  std::vector<long double> membership_vector = linguistic_domain->getMembership(x);
  for (i=0; i<linguistic_domain->getNumberOfLinguisticSets(); i++)
    {
      values[i] = membership_vector[i];
    }
}

long double FuzzyValue::getCrispValue()
{

  // center-of-maximum method

  int i;
  std::vector<long double> typical_value_vector;
  for (i=0; i<linguistic_domain->getNumberOfLinguisticSets(); i++)
    {
      typical_value_vector.push_back( linguistic_domain->getLinguisticSetList()[i]->getMembershipFunction()->getTypicalValue());
    }

  long double answer = 0;
  for (i=0; i<linguistic_domain->getNumberOfLinguisticSets(); i++)
    {
      answer = answer + typical_value_vector[i] * values[i];
    }
  return answer;
}

void FuzzyValue::clear()
{
  int i;
  for (i=0; i<linguistic_domain->getNumberOfLinguisticSets(); i++)
    {
      values[i] = 0;
    }
}

void FuzzyValue::setSetMembership(long double x, std::string nm)
{
  int i;
  for (i=0; i<linguistic_domain->getNumberOfLinguisticSets(); i++)
    {
      if (linguistic_domain->getLinguisticSetList()[i]->getName() == nm)
	{
	  values[i] = x;
	}
    }
}

long double FuzzyValue::getSetMembership(std::string nm)
{
  try {
    int length = linguistic_domain->getNumberOfLinguisticSets();
    int i;
    for (i=0; i < length; i++)
      {
	if (linguistic_domain->getLinguisticSetList()[i]->getName() == nm)
	  {
	    return values[i];
	  }
      }
  } catch (const std::exception& e) {
    std::cerr << "Error getting set membership:  " << e.what() << std::endl;
  }
}

long double FuzzyValue::AND(std::string sa, FuzzyValue* fv, std::string sb)
{
  long double a = getSetMembership(sa);
  long double b = fv->getSetMembership(sb);
  if (a <= b)
    {
      return a;
    }
else
  {
    return b;
  }
}

void FuzzyValue::OR_setSetMembership(long double x, std::string nm)
{
  int i;
  for (i=0; i<linguistic_domain->getNumberOfLinguisticSets(); i++)
    {
      if (linguistic_domain->getLinguisticSetList()[i]->getName() == nm)
	{
	  if (x > values[i])
	    {
	      values[i] = x;
	    }
	}
    }
}

