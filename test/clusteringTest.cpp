// Copyright (c) 2018 Shapelets.io
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <gtest/gtest.h>
#include <khiva/clustering.h>
#include <khiva/normalization.h>
#include <fstream>
#include <iostream>
#include "khivaTest.h"

void kShape() {
    float tolerance = 10e-10;

    float a[35] = {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,   6.0f,  7.0f,  0.0f, 10.0f, 4.0f, 5.0f, 7.0f,
                   -3.0f, 0.0f,  -1.0f, 15.0f, -12.0f, 8.0f,  9.0f,  4.0f, 5.0f,  2.0f, 8.0f, 7.0f,
                   -6.0f, -1.0f, 2.0f,  9.0f,  -5.0f,  -5.0f, -6.0f, 7.0f, 9.0f,  9.0f, 0.0f};
    af::array data = af::array(7, 5, a);
    data = khiva::normalization::znorm(data);

    // int indices[5] = {1, 0, 1, 2, 0};

    af::array idx;
    af::array centroids;

    khiva::clustering::kShape(data, 3, tolerance, idx, centroids);

    af_print(idx);
    af_print(centroids);

    /*float *index = idx.host<float>();

    for (unsigned int i = 0; i < static_cast<unsigned int>(idx.dims(0)); i++) {
        ASSERT_EQ(index[i], indices[i]);
    }*/
}

void kShape2() {
    float tolerance = 10e-10;

    float a[80] = {1.0f,   2.0f,   3.0f,   4.0f,   5.0f,   6.0f,   7.0f,   8.0f,   9.0f,   10.0f,  15.0f, 16.0f,
                   17.0f,  18.0f,  19.0f,  20.0f,  21.0f,  22.0f,  23.0f,  24.0f,  0.0f,   5.88f,  9.50f, 9.50f,
                   5.9f,   0.0f,   -5.87f, -9.50f, -9.50f, -5.9f,  2.0f,   7.88f,  11.50f, 11.50f, 7.9f,  2.0f,
                   -3.87f, -7.50f, -7.50f, -3.9f,  1.0f,   2.0f,   3.0f,   4.0f,   5.0f,   6.0f,   7.0f,  8.0f,
                   9.0f,   10.0f,  15.0f,  16.0f,  17.0f,  18.0f,  19.0f,  20.0f,  21.0f,  22.0f,  23.0f, 24.0f,
                   0.0f,   5.88f,  9.50f,  9.50f,  5.9f,   0.0f,   -5.87f, -9.50f, -9.50f, -5.9f,  2.0f,  7.88f,
                   11.50f, 11.50f, 7.9f,   2.0f,   -3.87f, -7.50f, -7.50f, -3.9f};

    af::array data = af::array(10, 8, a);
    af_print(data);

    data = khiva::normalization::znorm(data);
    af::array idx;
    af::array centroids;

    khiva::clustering::kShape(data, 2, tolerance, idx, centroids);

    af_print(idx);
    af_print(centroids);

    // float *index = idx.host<float>();

    /*for (unsigned int i = 0; i < static_cast<unsigned int>(idx.dims(0)); i++)
    {
        ASSERT_EQ(index[i], indices[i]);
    }*/
}

KHIVA_TEST(ClusteringTests, KShape, kShape)
KHIVA_TEST(ClusteringTests, KShape2, kShape2)
