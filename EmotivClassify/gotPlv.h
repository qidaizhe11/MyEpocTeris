#ifndef HILBERT_H
#define HILBERT_H

#include <QDebug>
#include <QString>

#include "armadillo"
#include "my_unwrap.h"

using namespace arma;
using namespace std;

cx_vec hilbert(const vec& in_mat)
{
    //    cout << "gotPlv::hilbert.";
    //    in_mat.print("in mat:");
    cx_vec out_mat = fft(in_mat);
    //    out_mat.print("after fft, out_mat:");

    int n = in_mat.n_rows;
    mat h = zeros(n, 1);

    if (n > 0 && 2.0 * floor(n / 2.0) == n) {
        h(0) = 1;
        h(n / 2) = 1;
        for (int i = 1; i < n / 2; ++i) {
            h(i) = 2;
        }
    } else if (n > 0) {
        h(0) = 1;
        for (int i = 1; i < (n + 1) / 2; ++i) {
            h(i) = 2;
        }
    }

    out_mat = ifft( out_mat % h.cols(0, out_mat.n_cols - 1) );
    //    out_mat.print("after ifft, out_mat:");
    return out_mat;
}

vec angle(const cx_vec& in_mat)
{
    vec imag_mat = imag(in_mat);
    vec real_mat = real(in_mat);

    vec out_mat = zeros(in_mat.n_rows);

    for (int i = 0; i < (int)in_mat.n_rows; ++i) {
        out_mat(i) = atan2(imag_mat(i), real_mat(i));
    }

    return out_mat;
}

cx_double b_exp(cx_double x)
{
    double r;
    double x_im;
    double b_x_im;
    r = exp(x.real() / 2.0);
    x_im = x.imag();
    b_x_im = x.imag();

    cx_double out_x( r * (r * cos(x_im)), r * (r * sin(b_x_im)) );
    return out_x;
}

/* Function Declarations */
static double rt_hypotd_snf(double u0, double u1);

/* Function Definitions */
static double rt_hypotd_snf(double u0, double u1)
{
    double y;
    double a;
    double b;
    a = fabs(u0);
    b = fabs(u1);
    if (a < b) {
        a /= b;
        y = b * sqrt(a * a + 1.0);
    } else if (a > b) {
        b /= a;
        y = a * sqrt(b * b + 1.0);
    } else if (std::isnan(b)) {
        y = b;
    } else {
        y = a * 1.4142135623730951;
    }

    return y;
}

double plv_hilbert(const vec& filtered_1, const vec& filtered_2)
{
    //    cout << "gotPlv::plv_hilbert." << endl;
    //    filtered_1.print("in filtered_1:");

    cx_vec hilbert_1 = hilbert(filtered_1);
    //    hilbert_1.print("hilbert_1:");
    cx_vec hilbert_2 = hilbert(filtered_2);

    int length = hilbert_1.n_elem;

    vec phase_1 = angle(hilbert_1);
    //    phase_1.print("phase_1:");
    vec phase_2 = angle(hilbert_2);

    my_unwrap(phase_1);
    //    phase_1.print("after unwrap, phase_1:");
    my_unwrap(phase_2);

    //    double plits = 0;
    //    for (int i = 0; i < length; ++i) {
    //        plits += ( exp((phase_1(i) - phase_2(i))) );
    //    }

    //    cout << "return:" << abs(plits) / length << endl;

    //    return abs(plits) / length;

    cx_double plits(0.0, 0.0);
    cx_double temp_cx(0.0, 0.0);
    for (int i = 0; i < length; ++i) {
        temp_cx.real(0.0);
        temp_cx.imag(phase_1(i) - phase_2(i));
        temp_cx = b_exp(temp_cx);
        plits.real(plits.real() + temp_cx.real());
        plits.imag(plits.imag() + temp_cx.imag());
    }

    /* 'plv_hilbert:35' plv=abs(plits)/len; */
    return rt_hypotd_snf(plits.real(), plits.imag()) / length;
}

void gotPlv(const mat& in_data, int trials_count,
            field<mat>* out_grouped_data, mat* out_plv_mat)
{
    mat temp_plv = zeros(22, 22);

    for (int i = 0; i < trials_count; ++i) {
        (*out_grouped_data)(i) = in_data.submat( 256*i, 0, 256*(i+1) - 1, 22 - 1 );

        for (int m = 0; m < 22; ++m) {
            for (int n = 0; n < 22; ++n) {
                temp_plv(m, n) = plv_hilbert((*out_grouped_data)(i).col(m),
                                             (*out_grouped_data)(i).col(n));
                temp_plv(n, m) = temp_plv(m, n);
            }
        }

        (*out_plv_mat) = (*out_plv_mat) + temp_plv;
    }
}

#endif // HILBERT_H
