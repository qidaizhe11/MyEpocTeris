#ifndef FDA_TEST_H
#define FDA_TEST_H

#include "armadillo"

using namespace arma;
using namespace std;

int FDA_TEST(const vec& samples, const mat& weights, double intercept, int K1, int L1)
{
    int dim_num = (int)samples.n_rows;
    int weight_num = (int)weights.n_cols;

    if (dim_num != weight_num) {
        cout << "FDA_TEST Error: Samples and Weights mismatch." << endl;
        return 0;
    }

    mat label_test = weights * samples - intercept;

//    label_test.print("label_test =");

    if (label_test(0, 0) >= 0) {
        return L1;
    } else {
        return K1;
    }

    return 0;
}

#endif // FDA_TEST_H
