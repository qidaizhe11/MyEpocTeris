#ifndef MY_CSP_H
#define MY_CSP_H

#include "armadillo"

using namespace arma;
using namespace std;

mat my_csp(const field<mat>& data_first, const field<mat>& data_second)
{
//    cout << "my_csp::my_csp." << endl;
    int trials_count_1 = (int)data_first.n_rows;
    int trials_count_2 = (int)data_second.n_rows;
    int sample_rate_1 = (int)(data_first(0, 0).n_rows);
    int sample_rate_2 = (int)(data_second(0, 0).n_rows);

    int n = data_first(0, 0).n_cols;
    mat r_1 = zeros(n, n);
    mat r_2 = zeros(n, n);

    for (int i = 0; i < trials_count_1; ++i) {
        r_1 = r_1 + trans(data_first(i)) * data_first(i) / sample_rate_1;
    }
    for (int i = 0; i < trials_count_2; ++i) {
        r_2 = r_2 + trans(data_second(i)) * data_second(i) / sample_rate_2;
    }

    r_1 = r_1 / trials_count_1;
    r_2 = r_2 / trials_count_2;
    mat R = r_1 + r_2;
    R = (R + R.t()) / 2;

//    R.print("before svd, R = ");

    mat U;
    vec s;
    mat V;
    svd(U, s, V, R);

//    U.print("U = ");
//    s.print("s = ");
//    V.print("V = ");

    vec temp_vec = arma::pow(s, -0.5);
//    temp_vec.print("temp_vec = ");

    mat W1 = U * diagmat(temp_vec);
    mat S1 = trans(W1) * r_1 * W1;

    mat U_2;
    vec s_2;
    mat V_2;
    mat X_2 = (S1 + trans(S1)) / 2;
    svd(U_2, s_2, V_2, X_2);

//    U_2.print("U_2 = ");
//    s_2.print("s_2 = ");
//    V_2.print("V_2 = ");

//    vec sorted_s = arma::sort(s_2);
    uvec sort_index_of_s = sort_index(s_2);

//    sorted_s.print("sorted of s =");
//    sort_index_of_s.print("sort index of s = ");

    U_2 = U_2.cols(sort_index_of_s);

//    U_2.print("after swap cols, U_2 = ");

    mat cropped_w = zeros(U_2.n_rows, 6);
    for (int i = 0; i < 3; ++i) {
        cropped_w.col(i) = U_2.col(i);
        cropped_w.col(3 + i) = U_2.col(U_2.n_cols - 3 + i);
    }

//    W.print("W = ");

    mat out_mat = W1 * cropped_w;

//    out_mat.print("out_mat = ");

    return out_mat;
}

#endif // MY_CSP_H
