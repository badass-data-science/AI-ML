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

/** Lambda (triangular) membership function: rises from min to mid, falls from mid to max. */
public class StandardMBF_Lambda extends MembershipFunction {

    private final double min;
    private final double mid;
    private final double max;
    private final double slopeUp;
    private final double yInterceptUp;
    private final double slopeDown;
    private final double yInterceptDown;

    public StandardMBF_Lambda(double min, double mid, double max) {
        this.min = min;
        this.mid = mid;
        this.max = max;
        this.typicalValue = mid;
        this.slopeUp = 1.0 / (mid - min);
        this.yInterceptUp = slopeUp * min * -1.0;
        this.slopeDown = -1.0 * (1.0 / (max - mid));
        this.yInterceptDown = -1.0 * slopeDown * mid + 1.0;
    }

    @Override
    public double getMembership(double x) {
        if (x <= min) return 0.0;
        if (x <= mid)  return slopeUp * x + yInterceptUp;
        if (x <= max)  return slopeDown * x + yInterceptDown;
        return 0.0;
    }
}
