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

#ifndef MEMBERSHIPFUNCTION
#define MEMBERSHIPFUNCTION

class MembershipFunction
{
 public:
  virtual long double getMembership(long double) = 0;
  long double getTypicalValue();
  
 protected:
  long double typical_value;
};

class StandardMBF_Z : public MembershipFunction
{
 public:
  StandardMBF_Z(long double, long double);
  long double getMembership(long double);

 protected:
  long double min;
  long double max;
  long double slope;
  long double y_intercept;
};

class StandardMBF_S : public MembershipFunction
{
 public:
  StandardMBF_S(long double, long double);
  long double getMembership(long double);

 protected:
  long double min;
  long double max;
  long double slope;
  long double y_intercept;
};

class StandardMBF_Lambda : public MembershipFunction
{
 public:
  StandardMBF_Lambda(long double, long double, long double);
  long double getMembership(long double);

 protected:
  long double min;
  long double mid;
  long double max;
  long double slope_up;
  long double y_intercept_up;
  long double slope_down;
  long double y_intercept_down;
};

class StandardMBF_Pi : public MembershipFunction
{
 public:
  StandardMBF_Pi(long double, long double, long double, long double);
  long double getMembership(long double);

 protected:
  long double min;
  long double lower_mid;
  long double higher_mid;
  long double max;
  long double slope_up;
  long double y_intercept_up;
  long double slope_down;
  long double y_intercept_down;
};

#endif
