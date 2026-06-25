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

/** S-shaped membership function: 0 below min, linear rise to 1 at max. */
public class StandardMBF_S extends MembershipFunction {

    private final double min;
    private final double max;
    private final double slope;
    private final double yIntercept;

    public StandardMBF_S(double min, double max) {
        this.min = min;
        this.max = max;
        this.typicalValue = max;
        this.slope = 1.0 / (max - min);
        this.yIntercept = slope * min * -1.0;
    }

    @Override
    public double getMembership(double x) {
        if (x <= min) return 0.0;
        if (x <= max)  return slope * x + yIntercept;
        return 1.0;
    }
}
