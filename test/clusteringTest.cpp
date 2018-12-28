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

void kShape()
{
    float tolerance = 10e-10;

    float a[35] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.0, 10.0, 4.0, 5.0, 7.0, -3.0, 0.0, -1.0, 15.0, -12.0, 8.0,
                   9.0, 4.0, 5.0, 2.0, 8.0, 7.0, -6.0, -1.0, 2.0, 9.0, -5.0, -5.0, -6.0, 7.0, 9.0, 9.0, 0.0};
    af::array data = af::array(7, 5, a);
    data = khiva::normalization::znorm(data);

    float indices[5] = {1, 0, 1, 2, 0};

    af::array idx;
    af::array centroids;

    khiva::clustering::kShape(data, 3, tolerance, idx, centroids);

    af_print(idx);
    af_print(centroids);

    float *index = idx.host<float>();

    for (unsigned int i = 0; i < static_cast<unsigned int>(idx.dims(0)); i++)
    {
        ASSERT_EQ(index[i], indices[i]);
    }
}

KHIVA_TEST(ClusteringTests, KShape, kShape)
