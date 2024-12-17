/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved
// -*- c++ -*-

#ifndef FAISS_INDEX_H
#define FAISS_INDEX_H


#include <cstdio>
#include <typeinfo>
#include <string>
#include <sstream>


/**
 * @namespace faiss
 *
 * 在整个库中，向量通过 float * 指针提供。
 * 大多数算法在处理多个向量（添加/搜索）时可以优化性能，即批量处理。
 * 在这种情况下，向量作为矩阵传递。当提供 n 个维度为 d 的向量作为 float * x 时，
 * 向量 i 的第 j 个分量为
 *
 *   x[ i * d + j ]
 *
 * 其中 0 <= i < n 且 0 <= j < d。换句话说，矩阵总是紧凑存储的。
 * 指定矩阵的大小时，我们称之为 n*d 矩阵，这意味着行主序存储。
 */


namespace faiss {


/// Some algorithms support both an inner product version and a L2 search version.
/// 一些算法同时支持内积版本和 L2 搜索版本。
enum MetricType {
    METRIC_INNER_PRODUCT = 0, // 内积度量
    METRIC_L2 = 1,            // L2 距离度量
};


/// Forward declarations see AuxIndexStructures.h
/// 前置声明，具体定义见 AuxIndexStructures.h
struct IDSelector;
struct RangeSearchResult;

/** Abstract structure for an index
 *
 * Supports adding vertices and searching them.
 *
 * Currently only asymmetric queries are supported:
 * database-to-database queries are not implemented.
 * 
 * @struct Index
 * @brief 索引的抽象结构
 * 支持添加向量和搜索向量。
 * 当前仅支持非对称查询：
 * 数据库到数据库的查询尚未实现。
 */
struct Index {

    typedef long idx_t;    ///< 所有索引的类型 all indices are this type 

    int d;                 ///< 向量的维度 vector dimension
    idx_t ntotal;          ///< 索引的总向量数量 total nb of indexed vectors
    bool verbose;          ///< 是否启用详细输出 verbosity level

    /// set if the Index does not require training, or if training is done already
    /// 如果索引不需要训练，或者已经完成训练，则设置为 true
    bool is_trained;

    /// type of metric this index uses for search
    /// 索引用于搜索的度量类型
    MetricType metric_type;
    /**
     * @brief 构造函数
     * @param d 向量的维度，默认为 0
     * @param metric 度量类型，默认为 METRIC_L2
     */
    explicit Index (idx_t d = 0, MetricType metric = METRIC_L2):
                    d(d),
                    ntotal(0),
                    verbose(false),
                    is_trained(true),
                    metric_type (metric) {}
    /// 虚析构函数，确保子类的析构函数被正确调用
    virtual ~Index ();


    /**
     * @brief 对代表性向量集进行训练
     *
     * @param n 向量数量
     * @param x 训练向量，大小为 n * d
     */
    virtual void train(idx_t n, const float* x);

    /**
     * @brief 向索引中添加 n 个维度为 d 的向量
     *
     * 向量被隐式分配标签 ntotal .. ntotal + n - 1
     * 此函数将输入向量切分为小于 blocksize_add 的块，并调用 add_core。
     *
     * @param n 向量数量
     * @param x 输入矩阵，大小为 n * d
     */
    virtual void add (idx_t n, const float *x) = 0;

    /**
     * @brief 添加向量并存储 xids 而不是顺序 ID
     *
     * 默认实现通过断言失败，因为并非所有索引都支持此功能。
     *
     * @param xids 如果非空，则存储向量的 ID（大小为 n）
     */
    virtual void add_with_ids (idx_t n, const float * x, const long *xids);

    /**
     * @brief 查询 n 个维度为 d 的向量
     *
     * 返回最多 k 个向量。如果查询结果不足 k 个，结果数组将填充 -1。
     *
     * @param n 查询向量数量
     * @param x 查询向量，大小为 n * d
     * @param k 返回最近的 k 个向量
     * @param distances 输出距离数组，大小为 n * k
     * @param labels 输出标签数组，大小为 n * k
     */
    virtual void search (idx_t n, const float *x, idx_t k,
                         float *distances, idx_t *labels) const = 0;

    /**
     * @brief 查询 n 个维度为 d 的向量，返回距离小于半径的所有向量
     *
     * 注意许多索引未实现 range_search（仅 k-NN 搜索是强制性的）。
     *
     * @param n 查询向量数量
     * @param x 查询向量，大小为 n * d
     * @param radius 搜索半径
     * @param result 结果表
     */
    virtual void range_search (idx_t n, const float *x, float radius,
                               RangeSearchResult *result) const;

    /**
     * @brief 返回与查询向量最接近的 k 个向量的索引
     *
     * 此函数与 search 相同，但仅返回邻居的标签。
     *
     * @param n 查询向量数量
     * @param x 查询向量，大小为 n * d
     * @param labels 输出标签数组，大小为 n * k
     * @param k 最近邻数量，默认为 1
     */
    void assign (idx_t n, const float * x, idx_t * labels, idx_t k = 1);

    /// removes all elements from the database.
    /// 移除数据库中的所有元素
    virtual void reset() = 0;

    /**
     * @brief 从索引中移除指定的 ID
     * 并非所有索引都支持此操作。
     * @param sel ID 选择器，用于确定要移除哪些 ID
     * @return 实际移除的 ID 数量
     */
    virtual long remove_ids (const IDSelector & sel);

    /**
     * @brief 重构存储的向量（或使用有损编码的近似向量）
     * 某些索引可能未定义此函数。
     * @param key 要重构的向量 ID
     * @param recons 重构后的向量（大小为 d）
     */
    virtual void reconstruct (idx_t key, float * recons) const;

    /**
     * @brief 重构从 i0 到 i0 + ni - 1 的向量
     * 某些索引可能未定义此函数。
     * @param i0 起始向量索引
     * @param ni 重构的向量数量
     * @param recons 重构后的向量（大小为 ni * d）
     */
    virtual void reconstruct_n (idx_t i0, idx_t ni, float *recons) const;

    /**
     * @brief 类似于 search，但同时重构搜索结果中的存储向量（或有损编码的近似向量）
     * 如果查询结果不足 k 个，结果数组将填充 -1。
     * @param n 查询向量数量
     * @param x 查询向量，大小为 n * d
     * @param k 最近邻数量
     * @param distances 输出距离数组，大小为 n * k
     * @param labels 输出标签数组，大小为 n * k
     * @param recons 重构后的向量，大小为 n * k * d
     */
    virtual void search_and_reconstruct (idx_t n, const float *x, idx_t k,
                                         float *distances, idx_t *labels,
                                         float *recons) const;

    /**
     * @brief 计算索引编码后的残差向量
     * 残差向量是向量与其在索引中表示的重构向量之间的差异。
     * 残差向量可用于多阶段索引方法，如 IndexIVF 的方法。
     * @param x 输入向量，大小为 d
     * @param residual 输出残差向量，大小为 d
     * @param key 索引中的编码 ID，由 search 和 assign 返回
     */
    void compute_residual (const float * x, float * residual, idx_t key) const;

    /**
     * @brief 显示实际的类名和一些额外的信息 便于调试和查看索引的状态。
    */
    void display () const;



};

} // namespace faiss


#endif // FAISS_INDEX_H
