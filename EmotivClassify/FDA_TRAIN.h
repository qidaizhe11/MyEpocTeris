#ifndef FDA_TRAIN_H
#define FDA_TRAIN_H

#include "armadillo"

using namespace arma;
using namespace std;

//mat squeeze(const mat& in_mat)
//{
//    if (in_mat.is_vec()) {
//        siz = in_mat.n_elem;
//    }
//}

template <typename T> int my_sign(T val) {
    return (T(0) < val) - (val < T(0));
}

void FDA_TRAIN(const mat& samples, const rowvec& labels, mat& weights, double& intercept)
{
//    vec weights = zeros<vec>(samples.n_rows);

    rowvec class_name = unique(labels);
    if (class_name.n_elem != 2) {
        cout << "FDA_TRAIN Error: Designed only for 2 class." << endl;
        return;
    }

    int sample_num = samples.n_cols;
    int label_num = labels.n_cols;
    if (label_num != sample_num) {
        cout << "FDA_TRAIN Error: Samples and Labels mismatch." << endl;
        return;
    }

    mat class_1 = samples.cols( find( labels == class_name(0)) );
    class_1 = trans(class_1);

//    class_1.print("class_1 =");

    mat class_2 = samples.cols( find( labels == class_name(1)) );
    class_2 = trans(class_2);

//    class_2.print("class_2 =");

    mat m_1 = mean(class_1, 0);
    mat m_2 = mean(class_2, 0);

//    m_1.print("m_1 =");

    mat s_1 = zeros(class_1.n_cols, class_1.n_cols);
    mat s_2 = zeros(class_2.n_cols, class_2.n_cols);

    for (int i = 0; i < (int)class_1.n_rows; ++i) {
        s_1 = s_1 + trans(class_1.row(i) - m_1) * (class_1.row(i) - m_1);
    }
    for (int i = 0; i < (int)class_2.n_rows; ++i) {
        s_2 = s_2 + trans(class_2.row(i) - m_2) * (class_2.row(i) - m_2);
    }

//    s_1.print("s_1 =");

    mat sw = s_1 + s_2;
//    sw.print("sw = ");

    mat W = inv(sw) * trans(m_1 - m_2);

//    W.print("W =");

    mat s_y1 = zeros(1, class_1.n_rows);
    mat s_y2 = zeros(1, class_2.n_rows);

    for (int i = 0; i < (int)class_1.n_rows; ++i) {
        s_y1.col(i) = trans(W) * trans(class_1.row(i));
    }
    for (int i = 0; i < (int)class_2.n_rows; ++i) {
        s_y2.col(i) = trans(W) * trans(class_2.row(i));
    }

//    s_y1.print("s_y1 =");
//    s_y2.print("s_y2 =");

    double y1 = mean(rowvec(s_y1));
    double y2 = mean(rowvec(s_y2));

    intercept = (y1 + y2) / 2;
    weights = trans(W);

//    weights.print("weights =");

    mat output_values = weights * samples - intercept;
    rowvec temp_output_values = rowvec(output_values);
    if ( my_sign(max(temp_output_values(find(labels == class_name(0))))) !=
         my_sign(class_name(0)) ) {
        weights = -weights;
        intercept = -intercept;
    }

//    weights.print("weights =");
}

#endif // FDA_TRAIN_H
