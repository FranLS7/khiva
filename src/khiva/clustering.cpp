// Copyright (c) 2018 Shapelets.io
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <arrayfire.h>
#include <khiva/clustering.h>
#include <khiva/distances.h>
#include <khiva/normalization.h>
#include <Eigen/Eigenvalues>
#include <limits>

af::array matrixNorm(af::array x, int axis = 0) {  // for 2 dimensional matrix
    // int _axis = (axis+1)%2;
    if (axis == 0) {
        return af::sqrt(af::sum(af::pow(x, 2)));
    } else {
        return af::sqrt(af::sum(af::pow(x, 2), 1));
    }
}

/**
 * 'x'.: 2D Array containing the input data
 * 'y'.: 1D Array usually containing the centroid
 *  Every signal is supposed to have been znormlized beforehand
 *  Returns a single arrayFire where the first column represents the distance
 *  and the second column, the indexes for the minimum distance
 */
void ncc2Dim(af::array x, af::array y, af::array &correlation, af::array &maxIndex) {
    af::array den = af::matmul(matrixNorm(x, 0).T(), matrixNorm(y, 0));

    af::array conv = af::constant(0, 2 * x.dims(0) - 1, x.dims(1));

    for (unsigned int i = 0; i < static_cast<unsigned int>(x.dims(1)); i++) {  // TODO gfor en vez de for
        conv.col(i) = af::convolve(x.col(i), af::flip(y, 0), AF_CONV_EXPAND) / (den(i).scalar<float>());
    }
    af::max(correlation, maxIndex, conv, 0);
}

void sbdPrivate(af::array x, af::array y, af::array &distance, af::array &xShifted) {
    af::array correlation;
    af::array index;

    xShifted = af::constant(0, x.dims(), x.type());
    ncc2Dim(x, y, correlation, index);
    distance = 1 - correlation;
    af::array shift = index - x.dims(0) + 1;
    float yLength = static_cast<float>(y.dims(0));
    for (unsigned int i = 0; i < static_cast<float>(x.dims(1)); i++) {
        if (shift(i).scalar<int>() >= 0) {
            xShifted.col(i) = af::join(0, af::constant(0, shift(i).scalar<float>()),
                                       y(af::range(yLength - shift(i).scalar<float>()), 0));
        } else
            xShifted.col(i) =
                af::join(0, y(af::range(yLength + shift(i).scalar<float>()), 0) + shift(i).scalar<float>(),
                         af::constant(0, -shift(i).scalar<float>()));
    }
}

/**
 * 'x'.: stands for the input data 2D array
 * 'y'.: will usually be used as the centroids' array
 */
af::array ncc3Dim(af::array x, af::array y) {
    af::array den = af::matmul(matrixNorm(y, 0).T(), matrixNorm(x, 0));  // combination of every pair of norms
    den(den == 0) = af::Inf;
    int distanceSize = static_cast<unsigned int>(y.dims(0)) * 2 - 1;

    af::array cc =
        af::constant(0, static_cast<unsigned int>(y.dims(1)), static_cast<unsigned int>(x.dims(1)), distanceSize);
    // TODO: el tamaño de la fft y el tamaño del resultado
    for (unsigned int i = 0; i < static_cast<unsigned int>(y.dims(1)); i++) {
        for (unsigned int j = 0; j < static_cast<unsigned int>(x.dims(1)); j++)
            cc(i, j, af::span) = af::convolve(x.col(j), af::flip(y.col(i), 0), AF_CONV_EXPAND);
    }
    den = af::tile(den, 1, 1, distanceSize);

    return (cc / den);
}

af::array eigenVectors(af::array matrix) {
    float *matHost = matrix.host<float>();
    Eigen::MatrixXf mat = Eigen::Map<Eigen::MatrixXf>(matHost, matrix.dims(0), matrix.dims(1));

    Eigen::EigenSolver<Eigen::MatrixXf> solution(mat);
    return af::array(matrix.dims(0), matrix.dims(1), solution.eigenvectors().real().data());
}

af::array eigenValues(af::array matrix) {
    float *matHost = matrix.host<float>();
    Eigen::MatrixXf mat = Eigen::Map<Eigen::MatrixXf>(matHost, matrix.dims(0), matrix.dims(1));

    Eigen::VectorXcf eivals = mat.eigenvalues();
    return af::array(matrix.dims(0), eivals.real().data());
}

/**
 * 'idx'.............: 1D Array containing the centroid every signal is assigned to
 * 'x'...............: 2D Array containing the input data
 * 'j'...............: number of the iteration (centroid)
 * 'currentCentroid'.: 1D Array containing the centroid we want to recalculate
 */
af::array extractShape(af::array idx, af::array x, af::array centroids) {
    af::array optX;
    af::array result = af::constant(0.0, centroids.dims());
    af::array distance;

    for (unsigned int i = 0; i < static_cast<unsigned int>(centroids.dims(1)); i++) {
        af::array a;
        for (unsigned int j = 0; j < static_cast<unsigned int>(idx.dims(0)); j++) {
            if (af::allTrue(af::iszero(centroids.col(i))).scalar<char>()) {
                optX = x.col(i);
            } else {
                sbdPrivate(x.col(i), centroids.col(i), distance, optX);
            }
            a = af::join(1, a, optX);
        }
        af_print(a);
        if (a.isempty()) {
            result.col(i) = af::constant(0, static_cast<unsigned int>(x.dims(0)));
            continue;
        }

        int columns = static_cast<unsigned int>(a.dims(0));
        af::array y = khiva::normalization::znorm(a);
        af::array s = af::matmul(y, y.T());

        af::array p = af::constant(1.0 / columns, columns, columns, x.type());
        af::array diagonal = af::constant(1, columns, x.type());
        p = af::diag(diagonal, 0, false) - p;

        af::array m = af::matmul(af::matmul(p, s), p);  // P*S*P
        result.col(i) = eigenVectors(m).col(af::end);   // highest order eigenvector

        float findDistance1 = af::sqrt(af::sum(af::pow((a(af::span, 0) - result.col(i)), 2))).scalar<float>();
        float findDistance2 = af::sqrt(af::sum(af::pow((a(af::span, 0) + result.col(i)), 2))).scalar<float>();

        if (findDistance1 >= findDistance2)
            result.col(i) = khiva::normalization::znorm(result.col(i) * (-1));
        else
            result.col(i) = khiva::normalization::znorm(result.col(i));
    }
    return result;
}

float computeError(af::array centroids, af::array newCentroids) {
    af_print(af::sum(af::sqrt(af::sum(af::pow(centroids - newCentroids, 2), 0))).as(af::dtype::f32));
    float *error = af::sum(af::sqrt(af::sum(af::pow(centroids - newCentroids, 2), 0))).as(af::dtype::f32).host<float>();
    return error[0];
}

void khiva::clustering::kShape(af::array tss, int k, float tolerance, af::array &idx,
                               af::array &centroids) {  // tolerance

    unsigned int nTimeSeries = static_cast<unsigned int>(tss.dims(1));  // number of signals in 'x'

    if (centroids.isempty()) {
        centroids = af::constant(0.0f, tss.dims(0), k);
    }

    if (idx.isempty()) {
        idx = af::floor(af::randu(nTimeSeries) * k);  // assigns a random centroid to every signal in 'x'
    }

    af::array oldIdx = idx;
    af::array min = af::constant(0, tss.dims(1));  // used to storage the minimum values

    af::array distances = af::constant(0, nTimeSeries, k);
    af::array newCentroids;

    float error = std::numeric_limits<float>::max();

    int iter = 0;

    af_print(tss);
    af_print(centroids);
    af_print(idx);

    while (error > tolerance) {
        /*af_print(centroids);
        af_print(idx);*/
        oldIdx = idx;

        newCentroids = extractShape(idx, tss, centroids);

        error = computeError(centroids, newCentroids);

        distances = af::max((1 - ncc3Dim(tss, newCentroids)), 2);
        centroids = newCentroids;

        af::min(min, idx, distances, 0);
        idx = idx.T();
        std::cout << iter++ << std::endl;
        // af_print(newCentroids);
    }
}