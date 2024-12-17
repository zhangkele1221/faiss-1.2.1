/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved

#include "IndexFlat.h"

#include <cstring>
#include "utils.h"
#include "Heap.h"

#include "FaissAssert.h"

#include "AuxIndexStructures.h"

namespace faiss {

IndexFlat::IndexFlat (idx_t d, MetricType metric):
            Index(d, metric)
{
}



void IndexFlat::add (idx_t n, const float *x) {
    xb.insert(xb.end(), x, x + n * d);
    ntotal += n;
}


void IndexFlat::reset() {
    xb.clear();
    ntotal = 0;
}


void IndexFlat::search (idx_t n, const float *x, idx_t k,
                               float *distances, idx_t *labels) const
{
    // 在Faiss中，向量搜索的结果（距离和索引ID）通常被组织成一种"堆"的形式（heap）
    // 对于内积搜索与L2搜索，会使用不同的堆策略（最小堆或最大堆）来维护当前最优的k个候选结果。
    // 内积搜索使用最小堆，因为我们要最大化内积值（相似度越大越好，相当于距离越小越好，所以使用最小堆存储负的内积或相反的比较逻辑）。
    // L2距离搜索使用最大堆，因为我们要寻找最小的L2距离，相对逻辑刚好相反。

    // 根据度量类型 metric_type 选择不同的knn函数：
    // METRIC_INNER_PRODUCT 使用 knn_inner_product
    // METRIC_L2 使用 knn_L2sqr

    if (metric_type == METRIC_INNER_PRODUCT) {
        // 对于内积度量（METRIC_INNER_PRODUCT）：
        // 构造一个表示最小堆结构的float_minheap_array_t对象，用于存放搜索结果。
        // 这里的res中保存要返回的labels和distances的指针，以及n和k的大小信息。
        float_minheap_array_t res = {
            size_t(n),   // 查询向量的数量
            size_t(k),   // 每个查询需要返回的最近邻数
            labels,      // 用于存储结果ID的数组指针
            distances    // 用于存储结果距离的数组指针
        };
        // 调用knn_inner_product函数对x与xb进行内积相似度计算，并将结果存放在res中
        knn_inner_product (x, xb.data(), d, n, ntotal, &res);

    } else if (metric_type == METRIC_L2) {
        // 对于L2距离度量（METRIC_L2）：
        // 构造一个表示最大堆结构的float_maxheap_array_t对象
        // 原因是对于L2距离，我们是在寻找距离最小的k个点，因此使用最大堆便于在搜索过程中替换较大距离的候选点。
        float_maxheap_array_t res = {
            size_t(n),
            size_t(k),
            labels,
            distances
        };
        // 调用knn_L2sqr函数对x与xb进行L2距离计算，并将结果存放在res中
        knn_L2sqr (x, xb.data(), d, n, ntotal, &res);
    }
}

void IndexFlat::range_search (idx_t n, const float *x, float radius,
                              RangeSearchResult *result) const
{
    switch (metric_type) {
        case METRIC_INNER_PRODUCT:
            range_search_inner_product (x, xb.data(), d, n, ntotal,
                                        radius, result);
            break;
        case METRIC_L2:
            range_search_L2sqr (x, xb.data(), d, n, ntotal, radius, result);
            break;
    }
}


void IndexFlat::compute_distance_subset (
            idx_t n,
            const float *x,
            idx_t k,
            float *distances,
            const idx_t *labels) const
{
    switch (metric_type) {
        case METRIC_INNER_PRODUCT:
            fvec_inner_products_by_idx (
                 distances,
                 x, xb.data(), labels, d, n, k);
            break;
        case METRIC_L2:
            fvec_L2sqr_by_idx (
                 distances,
                 x, xb.data(), labels, d, n, k);
            break;
    }

}

long IndexFlat::remove_ids (const IDSelector & sel)
{
    idx_t j = 0;
    for (idx_t i = 0; i < ntotal; i++) {
        if (sel.is_member (i)) {
            // should be removed
        } else {
            if (i > j) {
                memmove (&xb[d * j], &xb[d * i], sizeof(xb[0]) * d);
            }
            j++;
        }
    }
    long nremove = ntotal - j;
    if (nremove > 0) {
        ntotal = j;
        xb.resize (ntotal * d);
    }
    return nremove;
}



void IndexFlat::reconstruct (idx_t key, float * recons) const
{
    memcpy (recons, &(xb[key * d]), sizeof(*recons) * d);
}

/***************************************************
 * IndexFlatL2BaseShift
 ***************************************************/

IndexFlatL2BaseShift::IndexFlatL2BaseShift (idx_t d, size_t nshift, const float *shift):
    IndexFlatL2 (d), shift (nshift)
{
    memcpy (this->shift.data(), shift, sizeof(float) * nshift);
}

void IndexFlatL2BaseShift::search (
            idx_t n,
            const float *x,
            idx_t k,
            float *distances,
            idx_t *labels) const
{
    FAISS_THROW_IF_NOT (shift.size() == ntotal);

    float_maxheap_array_t res = {
        size_t(n), size_t(k), labels, distances};
    knn_L2sqr_base_shift (x, xb.data(), d, n, ntotal, &res, shift.data());
}



/***************************************************
 * IndexRefineFlat
 ***************************************************/

IndexRefineFlat::IndexRefineFlat (Index *base_index):
    Index (base_index->d, base_index->metric_type),
    refine_index (base_index->d, base_index->metric_type),
    base_index (base_index), own_fields (false),
    k_factor (1)
{
    is_trained = base_index->is_trained;
    FAISS_THROW_IF_NOT_MSG (base_index->ntotal == 0,
                      "base_index should be empty in the beginning");
}

IndexRefineFlat::IndexRefineFlat () {
    base_index = nullptr;
    own_fields = false;
    k_factor = 1;
}


void IndexRefineFlat::train (idx_t n, const float *x)
{
    base_index->train (n, x);
    is_trained = true;
}

void IndexRefineFlat::add (idx_t n, const float *x) {
    FAISS_THROW_IF_NOT (is_trained);
    base_index->add (n, x);
    refine_index.add (n, x);
    ntotal = refine_index.ntotal;
}

void IndexRefineFlat::reset ()
{
    base_index->reset ();
    refine_index.reset ();
    ntotal = 0;
}

namespace {
typedef faiss::Index::idx_t idx_t;

template<class C>
static void reorder_2_heaps (
      idx_t n,
      idx_t k, idx_t *labels, float *distances,
      idx_t k_base, const idx_t *base_labels, const float *base_distances)
{
#pragma omp parallel for
    for (idx_t i = 0; i < n; i++) {
        idx_t *idxo = labels + i * k;
        float *diso = distances + i * k;
        const idx_t *idxi = base_labels + i * k_base;
        const float *disi = base_distances + i * k_base;

        heap_heapify<C> (k, diso, idxo, disi, idxi, k);
        if (k_base != k) { // add remaining elements
            heap_addn<C> (k, diso, idxo, disi + k, idxi + k, k_base - k);
        }
        heap_reorder<C> (k, diso, idxo);
    }
}


}


void IndexRefineFlat::search (
              idx_t n, const float *x, idx_t k,
              float *distances, idx_t *labels) const
{
    FAISS_THROW_IF_NOT (is_trained);
    idx_t k_base = idx_t (k * k_factor);
    idx_t * base_labels = labels;
    float * base_distances = distances;
    ScopeDeleter<idx_t> del1;
    ScopeDeleter<float> del2;


    if (k != k_base) {
        base_labels = new idx_t [n * k_base];
        del1.set (base_labels);
        base_distances = new float [n * k_base];
        del2.set (base_distances);
    }

    base_index->search (n, x, k_base, base_distances, base_labels);

    for (int i = 0; i < n * k_base; i++)
        assert (base_labels[i] >= -1 &&
                base_labels[i] < ntotal);

    // compute refined distances
    refine_index.compute_distance_subset (
        n, x, k_base, base_distances, base_labels);

    // sort and store result
    if (metric_type == METRIC_L2) {
        typedef CMax <float, idx_t> C;
        reorder_2_heaps<C> (
            n, k, labels, distances,
            k_base, base_labels, base_distances);

    } else if (metric_type == METRIC_INNER_PRODUCT) {
        typedef CMin <float, idx_t> C;
        reorder_2_heaps<C> (
            n, k, labels, distances,
            k_base, base_labels, base_distances);
    }

}



IndexRefineFlat::~IndexRefineFlat ()
{
    if (own_fields) delete base_index;
}

/***************************************************
 * IndexFlat1D
 ***************************************************/


IndexFlat1D::IndexFlat1D (bool continuous_update):
    IndexFlatL2 (1),
    continuous_update (continuous_update)
{
}

/// if not continuous_update, call this between the last add and
/// the first search
void IndexFlat1D::update_permutation ()
{
    perm.resize (ntotal);
    if (ntotal < 1000000) {
        fvec_argsort (ntotal, xb.data(), (size_t*)perm.data());
    } else {
        fvec_argsort_parallel (ntotal, xb.data(), (size_t*)perm.data());
    }
}

void IndexFlat1D::add (idx_t n, const float *x)
{
    IndexFlatL2::add (n, x);
    if (continuous_update)
        update_permutation();
}

void IndexFlat1D::reset()
{
    IndexFlatL2::reset();
    perm.clear();
}

void IndexFlat1D::search (
            idx_t n,
            const float *x,
            idx_t k,
            float *distances,
            idx_t *labels) const
{
    FAISS_THROW_IF_NOT_MSG (perm.size() == ntotal,
                    "Call update_permutation before search");

#pragma omp parallel for
    for (idx_t i = 0; i < n; i++) {

        float q = x[i]; // query
        float *D = distances + i * k;
        idx_t *I = labels + i * k;

        // binary search
        idx_t i0 = 0, i1 = ntotal;
        idx_t wp = 0;

        if (xb[perm[i0]] > q) {
            i1 = 0;
            goto finish_right;
        }

        if (xb[perm[i1 - 1]] <= q) {
            i0 = i1 - 1;
            goto finish_left;
        }

        while (i0 + 1 < i1) {
            idx_t imed = (i0 + i1) / 2;
            if (xb[perm[imed]] <= q) i0 = imed;
            else                    i1 = imed;
        }

        // query is between xb[perm[i0]] and xb[perm[i1]]
        // expand to nearest neighs

        while (wp < k) {
            float xleft = xb[perm[i0]];
            float xright = xb[perm[i1]];

            if (q - xleft < xright - q) {
                D[wp] = q - xleft;
                I[wp] = perm[i0];
                i0--; wp++;
                if (i0 < 0) { goto finish_right; }
            } else {
                D[wp] = xright - q;
                I[wp] = perm[i1];
                i1++; wp++;
                if (i1 >= ntotal) { goto finish_left; }
            }
        }
        goto done;

    finish_right:
        // grow to the right from i1
        while (wp < k) {
            if (i1 < ntotal) {
                D[wp] = xb[perm[i1]] - q;
                I[wp] = perm[i1];
                i1++;
            } else {
                D[wp] = 1.0 / 0.0;
                I[wp] = -1;
            }
            wp++;
        }
        goto done;

    finish_left:
        // grow to the left from i0
        while (wp < k) {
            if (i0 >= 0) {
                D[wp] = q - xb[perm[i0]];
                I[wp] = perm[i0];
                i0--;
            } else {
                D[wp] = 1.0 / 0.0;
                I[wp] = -1;
            }
            wp++;
        }
    done:  ;
    }

}



} // namespace faiss