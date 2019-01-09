// Copyright (c) 2018 Shapelets.io
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <arrayfire.h>

namespace khiva
{

namespace clustering
{

/**
 * @brief Calculates the k-shape algorithm.
 *
 * @param tss Expects an input array whose dimension zero is the length of the time series (all the same) and
 * dimension one indicates the number of time series.
 * @param k The number of means to be computed.
 * @param centroids The resulting means or centroids.
 * @param idx The resulting labels of each time series which is the closest centroid.
 */
void kShape(af::array tss, int k, float tolerance, af::array &idx, af::array &centroids);
} // namespace clustering
} // namespace khiva