#ifndef F_SEG_LB_H
#define F_SEG_LB_H

#include "armadillo"

using namespace arma;
using namespace std;

int gotEndOfBlock(int block_start_index, const vec& in_vec)
{
    int rows = (int)in_vec.n_rows;

    for (int i = block_start_index + 1; i < rows; ++i) {
        if (in_vec(i) != in_vec(block_start_index)) {
            return i - 1;
        }

        if (i == rows - 1) {
            return rows - 1;
        }
    }

    return block_start_index + 1;
}

void gotBlockFlags(const mat& in_mat, vector<pair<int, int> >& block_ranges, vector<int>& flags)
{
    int rows = (int)in_mat.n_rows;
    int cols = (int)in_mat.n_cols;

    vec data = in_mat.col(cols - 1);
    int block_count = 0;
    int block_start_index = 0;
    int block_end_index = 0;

    int flag = 0;

    while (block_end_index < rows - 1) {
        block_end_index = gotEndOfBlock(block_start_index, data);

        pair<int, int> block_index_pair(block_start_index, block_end_index);
        block_ranges.push_back(block_index_pair);

        flag = (int)data.at(block_start_index);
        if (flag >= 1 && flag <= 4) {
            flags.push_back(flag);
        } else {
            flags.push_back(0);
        }

        ++block_count;

        block_start_index = block_end_index + 1;
    }
}

#endif // F_SEG_LB_H
