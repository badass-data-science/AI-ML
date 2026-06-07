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

#ifndef LINGUISTICDOMAIN
#define LINGUISTICDOMAIN


#include "LinguisticSet.hh"
#include <vector>
#include <string>

class LinguisticDomain
{
 public:
  LinguisticDomain(std::string);
  void addLinguisticSet(LinguisticSet*);
  std::vector<long double> getMembership(long double);
  long double getMembership(long double, std::string);
  int getNumberOfLinguisticSets();
  std::vector<LinguisticSet*> getLinguisticSetList();

 protected:
  std::vector<LinguisticSet*> linguistic_set_list;
  std::string name;
};

#endif
