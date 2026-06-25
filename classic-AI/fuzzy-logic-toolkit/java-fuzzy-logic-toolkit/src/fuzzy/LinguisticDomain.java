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
*/

package fuzzy;

import java.util.ArrayList;
import java.util.List;

public class LinguisticDomain {

    private final String name;
    private final List<LinguisticSet> linguisticSetList = new ArrayList<>();

    public LinguisticDomain(String name) {
        this.name = name;
    }

    public void addLinguisticSet(LinguisticSet set) {
        linguisticSetList.add(set);
    }

    public List<Double> getMembership(double x) {
        List<Double> memberships = new ArrayList<>();
        for (LinguisticSet set : linguisticSetList) {
            memberships.add(set.getMembershipFunction().getMembership(x));
        }
        return memberships;
    }

    public double getMembership(double x, String name) {
        for (LinguisticSet set : linguisticSetList) {
            if (set.getName().equals(name)) {
                return set.getMembershipFunction().getMembership(x);
            }
        }
        return 0.0;
    }

    public int getNumberOfLinguisticSets() {
        return linguisticSetList.size();
    }

    public List<LinguisticSet> getLinguisticSetList() {
        return linguisticSetList;
    }
}
