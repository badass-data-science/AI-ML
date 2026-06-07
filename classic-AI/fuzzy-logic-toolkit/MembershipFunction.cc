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

#include <cmath>

#include "MembershipFunction.hh"

long double MembershipFunction::getTypicalValue()
{
  return typical_value;
}

StandardMBF_Z::StandardMBF_Z(long double mn, long double mx)
{
  min = mn;
  max = mx;
  typical_value = min;
  slope = -1 * ( (1-0) / (max - min) );
  y_intercept = -1 * slope * min + 1;
}

long double StandardMBF_Z::getMembership(long double x)
{
  if (x <= min)
    {
      return 1;
    }
  if ( (x >= min) && (x <= max) )
    {
      return slope * x + y_intercept;
    }
  return 0;
}
 
StandardMBF_S::StandardMBF_S(long double mn, long double mx)
{
  min = mn;
  max = mx;
  typical_value = max;
  slope = (1-0) / (max - min);
  y_intercept = slope * min * -1;
}

long double StandardMBF_S::getMembership(long double x)
{
  if (x <= min)
    {
      return 0;
    }
  if ( (x >= min) && (x <= max) )
    {
      return slope * x + y_intercept;
    }
  return 1;
}

StandardMBF_Lambda::StandardMBF_Lambda(long double mn, long double md, long double mx)
{
  min = mn;
  mid = md;
  max = mx;
  typical_value = mid;
  slope_up = (1-0) / (mid - min);
  y_intercept_up = slope_up * min * -1;
  slope_down = -1 * ( (1-0) / (max - mid) );
  y_intercept_down = -1 * slope_down * mid + 1;
}

long double StandardMBF_Lambda::getMembership(long double x)
{
  if (x <= min)
    {
      return 0;
    }
  if ( (x >= min) && (x <= mid) )
    {
      return slope_up * x + y_intercept_up;
    }
  if ( (x >= mid) && (x <= max) )
    {
      return slope_down * x + y_intercept_down;
    }
  return 0;
}

StandardMBF_Pi::StandardMBF_Pi(long double mn, long double lmd, long double hmd, long double mx)
{
  min = mn;
  lower_mid = lmd;
  higher_mid = hmd;
  max = mx;
  typical_value = (lower_mid + higher_mid) / 2;
  slope_up = (1-0) / (lower_mid - min);
  y_intercept_up = slope_up * min * -1;
  slope_down = -1 * ( (1-0) / (max - higher_mid) );
  y_intercept_down = -1 * slope_down * higher_mid + 1;
}

long double StandardMBF_Pi::getMembership(long double x)
{
  if (x <= min)
    {
      return 0;
    }
  if ( (x >= min) && (x <= lower_mid) )
    {
      return slope_up * x + y_intercept_up;
    }
  if ( (x >= lower_mid) && (x <= higher_mid) )
    {
      return 1;
    }
  if ( (x >= higher_mid) && (x <= max) )
    {
      return slope_down * x + y_intercept_down;
    }
  return 0;
}
