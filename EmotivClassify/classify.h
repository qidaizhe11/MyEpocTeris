#ifndef CLASSIFY_H
#define CLASSIFY_H

#include <QDebug>
#include <QTime>
#include <QString>
#include <iostream>
#include "armadillo"

#include "filter.h"
#include "gotBlockFlags.h"
#include "gotPlv.h"
#include "corr2.h"
#include "my_csp.h"
#include "FDA_TRAIN.h"
#include "FDA_TEST.h"

using namespace arma;
using namespace std;

field<mat> grouped_data_1;
field<mat> grouped_data_2;
field<mat> grouped_data_3;
field<mat> grouped_data_4;

mat plv_1 = zeros(22, 22);
mat plv_2 = zeros(22, 22);
mat plv_3 = zeros(22, 22);
mat plv_4 = zeros(22, 22);

int sample_rate = 128 * 2;

static const double dv7[7] = { 1.0, -4.529164454361374, 8.8505270200022572,
                               -9.5805526346499423, 6.0714930158669018,
                               -2.1361120406300618, 0.325878984119881 };

static const double dv8[7] = { 0.013041793720968663, 0.0,
                               -0.039125381162905988, 0.0,
                               0.039125381162905988, 0.0,
                               -0.013041793720968663 };

mat classified_data_1;
mat classified_data_2;
mat classified_data_3;
mat classified_data_4;

void classifyReadData(const mat& unclassified_data,
                      const vector<pair<int, int> > block_ranges,
                      const vector<int> flags);

int classify(const mat& signal_data);

void train()
{
    QTime time_all;
    time_all.start();

    QTime time;

    time.start();
    qDebug() << "Begin to load file: 1_arma_v2.mat.";

    mat read_data;
    read_data.load("1_arma_v2.mat");

    qDebug() << QString("Load completed, cost: %1s").arg(time.elapsed() / 1000.0, 5);
    qDebug() << "read_data.n_rows:" << read_data.n_rows << " read_data.n_cols:" << read_data.n_cols;

//    read_data.save("1_arma_ascii.mat", raw_ascii);

    time.restart();
    filter(dv8, dv7, &read_data);
    qDebug() << QString("Filter completed, cost: %1s").arg(time.elapsed() / 1000.0, 5);

    mat unclassified_data = read_data.rows(0, read_data.n_rows / 2 - 1);

    qDebug() << "unclassified_data.n_rows:" << unclassified_data.n_rows;

    vector<pair<int, int> > block_ranges;
    vector<int> flags;
    gotBlockFlags(unclassified_data, block_ranges, flags);  // cost 1ms

    time.restart();

    classifyReadData(unclassified_data, block_ranges, flags);

    qDebug() << QString("classified completed, cost: %1s").arg(time.elapsed() / 1000.0, 5);

    int trials_count_1 = classified_data_1.n_rows / sample_rate;
    int trials_count_2 = classified_data_2.n_rows / sample_rate;
    int trials_count_3 = classified_data_3.n_rows / sample_rate;
    int trials_count_4 = classified_data_4.n_rows / sample_rate;

    grouped_data_1.set_size(trials_count_1);
    grouped_data_2.set_size(trials_count_2);
    grouped_data_3.set_size(trials_count_3);
    grouped_data_4.set_size(trials_count_4);

    time.restart();

    gotPlv(classified_data_1, trials_count_1, &grouped_data_1, &plv_1);

    qDebug() << QString("got_plv_1 completed, cost: %1s").arg(time.elapsed() / 1000.0, 5);

    time.restart();

    gotPlv(classified_data_2, trials_count_2, &grouped_data_2, &plv_2);
    qDebug() << QString("got_plv_2 completed, cost: %1s").arg(time.elapsed() / 1000.0, 5);

    time.restart();

    gotPlv(classified_data_3, trials_count_3, &grouped_data_3, &plv_3);
    qDebug() << QString("got_plv_3 completed, cost: %1s").arg(time.elapsed() / 1000.0, 5);

    time.restart();

    gotPlv(classified_data_4, trials_count_4, &grouped_data_4, &plv_4);
    qDebug() << QString("got_plv_4 completed, cost: %1s").arg(time.elapsed() / 1000.0, 5);

    plv_1 = plv_1 / trials_count_1;
    plv_2 = plv_2 / trials_count_2;
    plv_3 = plv_3 / trials_count_3;
    plv_4 = plv_4 / trials_count_4;

    qDebug() << QString("train completed, cost all: %1s").arg(time_all.elapsed() / 1000.0, 5);

    mat test_data = read_data.submat(27648, 0, read_data.n_rows - 1, read_data.n_cols - 2);

    mat block_data;
    for (int i = 0; i < (int)(test_data.n_rows / sample_rate); ++i) {
        time.restart();

        block_data = test_data.rows( 256 * i, 256 * (i+1) - 1 );

        int flag = classify(block_data);

        qDebug() << QString("%1 got signal flag %2, cost: %3s").arg(i, 3).arg(flag).arg(time.elapsed() / 1000.0, 5);

    }
}

int gotSignalFlag(const mat& signal_data, int K1, int L1);

void makeFlagPairs(const mat& block_data, vector<pair<double, int>>& flags);

int classify(const mat& signal_data)
{
    vector<pair<double, int>> signal_flags;
    makeFlagPairs(signal_data, signal_flags);

    int flag_1 = gotSignalFlag(signal_data, signal_flags.at(0).second,
                               signal_flags.at(1).second);
    int flag_2 = gotSignalFlag(signal_data, signal_flags.at(2).second,
                               signal_flags.at(3).second);

    int flag = gotSignalFlag(signal_data, flag_1, flag_2);

    return flag;
}

void classifyReadData(const mat& unclassified_data,
                      const vector<pair<int, int> > block_ranges,
                      const vector<int> flags)
{
    int begin = 0;
    int end = 0;
    int data_cols = unclassified_data.n_cols - 1;
    auto i = block_ranges.begin();
    auto j = flags.begin();
    for (; i != block_ranges.end(); ++i, ++j) {
        begin = (*i).first;
        end = (*i).second;

        switch (*j) {
        case 1:
            classified_data_1 = join_vert(classified_data_1,
                    unclassified_data.submat(begin, 0, end, data_cols - 1));
            break;
        case 2:
            classified_data_2 = join_vert(classified_data_2,
                    unclassified_data.submat(begin, 0, end, data_cols - 1));
            break;
        case 3:
            classified_data_3 = join_vert(classified_data_3,
                    unclassified_data.submat(begin, 0, end, data_cols - 1));
            break;
        case 4:
            classified_data_4 = join_vert(classified_data_4,
                    unclassified_data.submat(begin, 0, end, data_cols - 1));
            break;
        }
    }
}

void makeFlagPairs(const mat& block_data, vector<pair<double, int>>& flags)
{
    mat temp_plv = zeros(22, 22);

    for (int m = 0; m < 22; ++m) {
        for (int n = m; n < 22; ++n) {
            temp_plv(m, n) = plv_hilbert(block_data.col(m), block_data.col(n));
            temp_plv(n, m) = temp_plv(m, n);
        }
    }

    double a = corr2(temp_plv, plv_1);
    double b = corr2(temp_plv, plv_2);
    double c = corr2(temp_plv, plv_3);
    double d = corr2(temp_plv, plv_4);

    vec con;
    con << a << b << c << d;
    double mean_of_all = mean(con);

    for (int i = 0; i < 4; ++i) {
        flags.push_back(pair<double, int>( abs(con(i) - mean_of_all), i + 1 ));
    }

    std::sort(flags.begin(), flags.end(),
              [](const pair<double, int>& lhs, const pair<double, int>& rhs) {
                  return lhs.first > rhs.first; } );
}

int gotSignalFlag(const mat& signal_data, int K1, int L1)
{
    field<mat> data_first;
    field<mat> data_second;

    switch (K1) {
    case 1: data_first = grouped_data_1; break;
    case 2: data_first = grouped_data_2; break;
    case 3: data_first = grouped_data_3; break;
    case 4: data_first = grouped_data_4; break;
    default: break;
    }

    switch (L1) {
    case 1: data_second = grouped_data_1; break;
    case 2: data_second = grouped_data_2; break;
    case 3: data_second = grouped_data_3; break;
    case 4: data_second = grouped_data_4; break;
    default: break;
    }

    mat W = my_csp(data_first, data_second);

    mat p_first = zeros(W.n_cols, data_first.n_elem);
    mat p_second = zeros(W.n_cols, data_second.n_elem);

    vec temp_mat;
    for (int j = 0; j < (int)data_first.n_elem; ++j) {
        temp_mat = diagvec( trans(data_first(j) * W) * data_first(j) * W ) / sample_rate;
        p_first.col(j) = temp_mat.rows(0, W.n_cols - 1);
    }
    for (int j = 0; j < (int)data_second.n_elem; ++j) {
        temp_mat = diagvec( trans(data_second(j) * W) * data_second(j) * W ) / sample_rate;
        p_second.col(j) = temp_mat.rows(0, W.n_cols - 1);
    }

    mat sample = join_horiz(p_first, p_second);
    rowvec class_label = join_horiz( ones(1, data_first.n_elem) * -1,
                                     ones(1, data_second.n_elem));

    mat weights;
    double intercept = 0.0;
    FDA_TRAIN(sample, class_label, weights, intercept);  // cost 2ms

    vec samples = diagvec(trans(signal_data * W) * signal_data * W) / signal_data.n_rows;

    int flag = 0;
    flag = FDA_TEST(samples, weights, intercept, K1, L1);

    return flag;
}

#endif // CLASSIFY_H
