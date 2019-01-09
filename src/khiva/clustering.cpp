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
    conv = af::flip(conv, 0);
    af::max(correlation, maxIndex, conv, 0);
}

/**
 * 'x'........: signals compared against 'y'
 * 'y'........: reference signal
 * 'distance'.: output argument containing the distance between signals
 * 'xShifted'.: 'x' signal shifted in order to obtain minimal distance
 */
void sbdPrivate(af::array x, af::array y, af::array &distance, af::array &xShifted) {
    af::array correlation;
    af::array index;

    xShifted = af::constant(0, x.dims(), x.type());
    ncc2Dim(x, y, correlation, index);

    distance = 1 - correlation;

    af::array shift = index - x.dims(0) + 1;
    af_print(shift);
    float xLength = static_cast<float>(x.dims(0));
    for (int i = 0; i < static_cast<int>(x.dims(1)); i++) {
        if (shift(i).scalar<int>() >= 0) {
            xShifted.col(i) =
                af::join(0, af::constant(0, shift(i).scalar<int>()), x(af::range(xLength - shift(i).scalar<int>()), 0));
        } else {
            xShifted.col(i) = af::join(0, x(af::range(xLength + shift(i).scalar<int>()) - shift(i).scalar<int>(), 0),
                                       af::constant(0, -shift(i).scalar<int>()));
        }
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

    Eigen::MatrixXf re;
    re = solution.eigenvectors().real();

    return af::array(matrix.dims(0), matrix.dims(1), re.data());
}

af::array eigenValues(af::array matrix) {
    float *matHost = matrix.host<float>();
    Eigen::MatrixXf mat = Eigen::Map<Eigen::MatrixXf>(matHost, matrix.dims(0), matrix.dims(1));

    Eigen::VectorXcf eivals = mat.eigenvalues();

    Eigen::VectorXf re = eivals.real();
    return af::array(matrix.dims(0), re.data());
}

/**
 * 'idx'.............: 1D Array containing the centroid every signal is assigned to
 * 'x'...............: 2D Array containing the input data
 * 'j'...............: number of the iteration (centroid)
 * 'currentCentroid'.: 1D Array containing the centroid we want to recalculate
 */
af::array extractShape(af::array idx, af::array x, af::array centroids) {
    af::array xShifted;
    af::array result = af::constant(0.0, centroids.dims(), af::dtype::f32);
    af::array distance;

    for (int i = 0; i < static_cast<int>(centroids.dims(1)); i++) {
        std::cout << "/****************CÁLCULO CENTROIDE " << i << "********************/" << std::endl;
        af::array a;
        for (int j = 0; j < static_cast<int>(idx.dims(0)); j++) {
            if (idx(j).scalar<int>() == i) {
                if (af::allTrue(af::iszero(centroids.col(i))).scalar<char>()) {
                    xShifted = x.col(j);
                } else {
                    sbdPrivate(x.col(j), centroids.col(i), distance, xShifted);
                    af_print(distance);
                }
                a = af::join(1, a, xShifted);
            }
        }
        if (a.isempty()) {
            result.col(i) = af::constant(0, static_cast<unsigned int>(x.dims(0)));
            continue;
        }
        af_print(a);
        int matrixSize = static_cast<unsigned int>(a.dims(0));
        af::array y = khiva::normalization::znorm(a);
        af::array S = af::matmul(y, y.T());

        af::array Q = af::constant(1.0 / matrixSize, matrixSize, matrixSize, x.type());
        af::array diagonal = af::constant(1, matrixSize, x.type());
        Q = af::diag(diagonal, 0, false) - Q;

        af::array M = af::matmul(af::matmul(Q, S), Q);  // Q_T*S*Q. Q is a simmetric matrix
        af::array eigenvalues = eigenValues(M);
        af::array maxEigenValue;
        af::array indMaxEigenValue;
        af::max(maxEigenValue, indMaxEigenValue, eigenvalues, 0);
        result.col(i) = eigenVectors(M).col(indMaxEigenValue(0).scalar<int>());  // highest order eigenvector

        // delete these lines
        af_print(eigenvalues);
        af_print(M);
        af_print(eigenVectors(M));
        af_print(maxEigenValue);
        af_print(indMaxEigenValue);

        float findDistance1 = af::sqrt(af::sum(af::pow((a(af::span, 0) - result.col(i)), 2))).scalar<float>();
        float findDistance2 = af::sqrt(af::sum(af::pow((a(af::span, 0) + result.col(i)), 2))).scalar<float>();

        if (findDistance1 >= findDistance2)
            result.col(i) = khiva::normalization::znorm(result.col(i) * (-1));
        else
            result.col(i) = khiva::normalization::znorm(result.col(i));
        af_print(result.col(i));
        getchar();
    }
    return result;
}

float computeError(af::array centroids, af::array newCentroids) {
    af_print(af::sum(af::sqrt(af::sum(af::pow(centroids - newCentroids, 2), 0))).as(af::dtype::f32));
    float *error = af::sum(af::sqrt(af::sum(af::pow(centroids - newCentroids, 2), 0))).as(af::dtype::f32).host<float>();
    return error[0];
}

void khiva::clustering::kShape(af::array tss, int k, float tolerance, af::array &idx,
                               af::array &centroids) {  // TODO: include tolerance

    unsigned int nTimeSeries = static_cast<unsigned int>(tss.dims(1));  // number of signals in 'x'

    if (centroids.isempty()) {
        centroids = af::constant(0.0f, tss.dims(0), k);
    }

    if (idx.isempty()) {
        idx = af::floor(af::randu(nTimeSeries) * (k)).as(af::dtype::s32);
        // assigns a random centroid to every signal in 'x'
    }

    // TODO: NORMALIZATION (change name in other calls)
    // af::array normalizedTss = khiva::normalization::znorm(tss);

    // Delete these both lines
    int ar[5] = {1, 0, 0, 1, 0};
    idx = af::array(5, ar);

    af::array oldIdx = idx;
    af::array min = af::constant(0, tss.dims(1));  // used to storage the minimum values

    af::array distances = af::constant(0, nTimeSeries, k);
    af::array newCentroids;

    float error = std::numeric_limits<float>::max();

    int iter = 0;

    af_print(tss);
    af_print(idx);

    while (error > tolerance) {
        std::cout << "/****************************NEW ITERATION****************************/" << std::endl;
        std::cout << iter++ << std::endl;

        oldIdx = idx;

        newCentroids = extractShape(idx, tss, centroids);

        error = computeError(centroids, newCentroids);

        distances = 1 - af::max(ncc3Dim(tss, newCentroids), 2);
        af_print(distances);
        centroids = newCentroids;

        af::min(min, idx, distances, 0);
        idx = idx.T();

        af_print(idx);
        af_print(newCentroids);
        getchar();
        /*af_print(idx);
        af_print(oldIdx);*/

        // if ((idx == oldIdx).scalar<char>()) break;
        // std::cin >> c;
    }
}