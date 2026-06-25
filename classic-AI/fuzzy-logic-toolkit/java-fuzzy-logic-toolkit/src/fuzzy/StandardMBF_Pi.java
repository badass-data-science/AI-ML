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

/** Pi (trapezoidal) membership function: rises from min to lowerMid, flat to higherMid, falls to max. */
public class StandardMBF_Pi extends MembershipFunction {

    private final double min;
    private final double lowerMid;
    private final double higherMid;
    private final double max;
    private final double slopeUp;
    private final double yInterceptUp;
    private final double slopeDown;
    private final double yInterceptDown;

    public StandardMBF_Pi(double min, double lowerMid, double higherMid, double max) {
        this.min = min;
        this.lowerMid = lowerMid;
        this.higherMid = higherMid;
        this.max = max;
        this.typicalValue = (lowerMid + higherMid) / 2.0;
        this.slopeUp = 1.0 / (lowerMid - min);
        this.yInterceptUp = slopeUp * min * -1.0;
        this.slopeDown = -1.0 * (1.0 / (max - higherMid));
        this.yInterceptDown = -1.0 * slopeDown * higherMid + 1.0;
    }

    @Override
    public double getMembership(double x) {
        if (x <= min)       return 0.0;
        if (x <= lowerMid)  return slopeUp * x + yInterceptUp;
        if (x <= higherMid) return 1.0;
        if (x <= max)       return slopeDown * x + yInterceptDown;
        return 0.0;
    }
}
