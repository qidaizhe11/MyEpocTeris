#ifndef FILTER_H
#define FILTER_H

#include "armadillo"

using namespace arma;
using namespace std;

void filter(const double b[7], const double a[7], mat* in_mat)
{
    double dbuffer[7];
    int k;

    if (in_mat->n_cols == 0) {
        qDebug() << "in [filter], 0 col in_mat. Will return directly since "
                    "I just simple cut the last flag col in this function.";
        return;
    }

    for (uint c = 0; c < in_mat->n_cols - 1; ++c) {
        for (k = 0; k < 6; ++k) {
            dbuffer[k + 1] = 0.0;
        }

        for (uint j = 0; j < in_mat->n_rows; ++j) {
            for (k = 0; k < 6; ++k) {
                dbuffer[k] = dbuffer[k + 1];
            }

            dbuffer[6] = 0.0;
            for (k = 0; k < 7; ++k) {
                dbuffer[k] += in_mat->at(j, c) * b[k];
            }

            for (k = 0; k < 6; ++k) {
                dbuffer[k + 1] -= dbuffer[0] * a[k + 1];
            }

            in_mat->at(j, c) = dbuffer[0];

//            qDebug() << "in_mat, at:" << j << c << in_mat->at(j, c);
        }
    }
}

#endif // FILTER_H
