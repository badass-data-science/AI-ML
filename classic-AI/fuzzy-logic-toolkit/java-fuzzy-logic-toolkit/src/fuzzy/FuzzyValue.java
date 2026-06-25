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

public class FuzzyValue {

    private final LinguisticDomain linguisticDomain;
    private final double[] values;

    public FuzzyValue(LinguisticDomain linguisticDomain) {
        this.linguisticDomain = linguisticDomain;
        this.values = new double[linguisticDomain.getNumberOfLinguisticSets()];
    }

    public void setCrispValue(double x) {
        List<Double> memberships = linguisticDomain.getMembership(x);
        for (int i = 0; i < values.length; i++) {
            values[i] = memberships.get(i);
        }
    }

    /** Defuzzifies using the center-of-maximum method (weighted sum of typical values). */
    public double getCrispValue() {
        List<LinguisticSet> sets = linguisticDomain.getLinguisticSetList();
        double answer = 0.0;
        for (int i = 0; i < values.length; i++) {
            answer += sets.get(i).getMembershipFunction().getTypicalValue() * values[i];
        }
        return answer;
    }

    public void clear() {
        for (int i = 0; i < values.length; i++) {
            values[i] = 0.0;
        }
    }

    public void setSetMembership(double x, String name) {
        List<LinguisticSet> sets = linguisticDomain.getLinguisticSetList();
        for (int i = 0; i < sets.size(); i++) {
            if (sets.get(i).getName().equals(name)) {
                values[i] = x;
                return;
            }
        }
    }

    public double getSetMembership(String name) {
        List<LinguisticSet> sets = linguisticDomain.getLinguisticSetList();
        for (int i = 0; i < sets.size(); i++) {
            if (sets.get(i).getName().equals(name)) {
                return values[i];
            }
        }
        return 0.0;
    }

    /** Fuzzy AND: returns the minimum of the two named set memberships. */
    public double AND(String nameA, FuzzyValue other, String nameB) {
        double a = getSetMembership(nameA);
        double b = other.getSetMembership(nameB);
        return Math.min(a, b);
    }

    /** Fuzzy OR: sets the named membership to the max of x and its current value. */
    public void OR_setSetMembership(double x, String name) {
        List<LinguisticSet> sets = linguisticDomain.getLinguisticSetList();
        for (int i = 0; i < sets.size(); i++) {
            if (sets.get(i).getName().equals(name)) {
                if (x > values[i]) {
                    values[i] = x;
                }
                return;
            }
        }
    }
}
