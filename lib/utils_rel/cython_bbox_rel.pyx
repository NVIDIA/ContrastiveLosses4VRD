# Adapted by Ji Zhang for this project in 2019
#
# Original license text below:
#############################################################################
# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
#
# Based on:
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Sergey Karayev
# --------------------------------------------------------

cimport cython
import numpy as np
cimport numpy as np

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t


@cython.boundscheck(False)
def bbox_pair_overlaps(
        np.ndarray[DTYPE_t, ndim=2] boxes1,
        np.ndarray[DTYPE_t, ndim=2] boxes2):
    """
    Parameters
    ----------
    boxes1: (N, 4) ndarray of float
    boxes2: (N, 4) ndarray of float
    Returns
    -------
    overlaps: (N,) ndarray of overlaps between each pair of boxes1 and boxes2
    """
    assert boxes1.shape[0] == boxes2.shape[0]
    cdef unsigned int N = boxes1.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] overlaps = np.zeros(N, dtype=DTYPE)
    cdef DTYPE_t iw, ih, box_area
    cdef DTYPE_t ua
    cdef unsigned int n
    with nogil:
        for n in range(N):
            box_area = (
                (boxes2[n, 2] - boxes2[n, 0] + 1) *
                (boxes2[n, 3] - boxes2[n, 1] + 1)
            )
            iw = (
                min(boxes1[n, 2], boxes2[n, 2]) -
                max(boxes1[n, 0], boxes2[n, 0]) + 1
            )
            if iw > 0:
                ih = (
                    min(boxes1[n, 3], boxes2[n, 3]) -
                    max(boxes1[n, 1], boxes2[n, 1]) + 1
                )
                if ih > 0:
                    ua = float(
                        (boxes1[n, 2] - boxes1[n, 0] + 1) *
                        (boxes1[n, 3] - boxes1[n, 1] + 1) +
                        box_area - iw * ih
                    )
                    overlaps[n] = iw * ih / ua
    return overlaps
