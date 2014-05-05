#ifndef MY_UNWRAP_H
#define MY_UNWRAP_H

#include "armadillo"

using namespace arma;
using namespace std;

/* Function Declarations */
static double rt_roundd_snf(double u);

/* Function Definitions */
static double rt_roundd_snf(double u)
{
  double y;
  if (fabs(u) < 4.503599627370496E+15) {
    if (u >= 0.5) {
      y = floor(u + 0.5);
    } else if (u > -0.5) {
      y = -0.0;
    } else {
      y = ceil(u - 0.5);
    }
  } else {
    y = u;
  }

  return y;
}

/*
 *
 */
void my_unwrap(vec& p)
{
  double cumsum_dp_corr = 0.0;
  double pkm1;
  int exitg1;
  double r;
  double dps;
  int k = 0;
  while ((k + 1 < 256) && (!((!std::isinf(p[k])) && (!std::isnan(p[k]))))) {
    k++;
  }

  if (k + 1 < 256) {
    pkm1 = p[k];
    do {
      exitg1 = 0;
      k++;
      while ((k + 1 <= 256) && (!((!std::isinf(p[k])) && (!std::isnan(p[k]))))) {
        k++;
      }

      if (k + 1 > 256) {
        exitg1 = 1;
      } else {
        pkm1 = p[k] - pkm1;
        r = (pkm1 + 3.1415926535897931) / 6.2831853071795862;
        if (fabs(r - rt_roundd_snf(r)) <= 2.2204460492503131E-16 * fabs(r)) {
          r = 0.0;
        } else {
          r = (r - floor(r)) * 6.2831853071795862;
        }

        dps = r - 3.1415926535897931;
        if ((r - 3.1415926535897931 == -3.1415926535897931) && (pkm1 > 0.0)) {
          dps = 3.1415926535897931;
        }

        if (fabs(pkm1) >= 3.1415926535897931) {
          cumsum_dp_corr += dps - pkm1;
        }

        pkm1 = p[k];
        p[k] += cumsum_dp_corr;
      }
    } while (exitg1 == 0);
  }
}

#endif // MY_UNWRAP_H
