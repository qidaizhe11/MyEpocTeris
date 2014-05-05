#ifndef CORR2_H
#define CORR2_H

#include "armadillo"

using namespace arma;
using namespace std;

double corr2(const mat& first_mat, const mat& second_mat)
{
//    cout << "corr2::corr2." << endl;
//    first_mat.print("in first_mat:");
//    second_mat.print("in second mat:");
    double mean_first = mean(mean(first_mat));
    double mean_second = mean(mean(second_mat));

    mat first = first_mat - mean_first;
    mat second = second_mat - mean_second;

//    first.print("first:");
//    second.print("second:");

//    r = sum(sum(a.*b))/sqrt(sum(sum(a.*a))*sum(sum(b.*b)));

    double out_double = sum(sum(first % second)) /
            sqrt( sum(sum(first % first)) * sum(sum(second % second)) );
    return out_double;
}

#endif // CORR2_H
