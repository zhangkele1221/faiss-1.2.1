/**
 * 版权声明与许可证信息
 * BSD+Patents license
 */

#include <cstdio>    // printf等标准输出函数
#include <cstdlib>   // 用于drand48等随机数函数

#include <faiss/IndexFlat.h> // 包含Faiss中的IndexFlat类定义

int main() {
    int d = 64;                            // 向量的维度 dimension
    int nb = 100000;                       // 数据库中向量的数量 database size
    int nq = 10000;                        // 查询向量数量 number of queries

    // 分配存储空间，xb存放数据库向量，xq存放查询向量
    float *xb = new float[d * nb];
    float *xq = new float[d * nq];

    // 为数据库向量xb赋值
    // 这里的赋值方案是随机生成float值，同时对第0维增加一个小的线性偏移(i/1000.)，使得数据有一定模式
    for(int i = 0; i < nb; i++) {
        for(int j = 0; j < d; j++)
            xb[d * i + j] = drand48(); // drand48()返回0到1之间的随机double
        xb[d * i] += i / 1000.; // 在第一个分量上增加i/1000.，相当于给数据添加一个可区分的偏移量
    }

    // 为查询向量xq赋值，与xb类似的随机方式
    for(int i = 0; i < nq; i++) {
        for(int j = 0; j < d; j++)
            xq[d * i + j] = drand48();
        xq[d * i] += i / 1000.; // 同样给查询向量第0维加上偏移量
    }

    // 创建一个IndexFlatL2索引对象，用L2距离度量
    // IndexFlatL2是Faiss中最简单的索引类型，它不进行任何压缩或分块，只是将所有向量存储起来并在搜索时进行全表扫描。
    faiss::IndexFlatL2 index(d);           
    // 打印索引是否已训练
    // 对于IndexFlatL2来说，无需训练，因此is_trained=true
    printf("is_trained = %s\n", index.is_trained ? "true" : "false");

    // 将数据库向量添加到索引中
    index.add(nb, xb);
    // 打印当前索引中向量的总数目
    // 应该为 nb，即100000
    printf("ntotal = %ld\n", index.ntotal);

    int k = 4; // 搜索时希望返回的最近邻数量

    {   // 这一段代码是一个小的"sanity check"
        // 用数据库本身的前5个向量作为查询进行检索，期望它们应该能在索引中找到自己

        long *I = new long[k * 5];    // 用于存放搜索结果的ID数组
        float *D = new float[k * 5];  // 用于存放搜索结果的距离数组

        // 对前5个数据库向量进行搜索，查询5个向量，返回每个向量的4个最近邻
        index.search(5, xb, k, D, I);

        // 打印搜索结果的ID矩阵I
        // I[i * k + j]表示第i个查询的第j个最近邻的ID
        printf("I=\n");
        for(int i = 0; i < 5; i++) {
            for(int j = 0; j < k; j++)
                printf("%5ld ", I[i * k + j]);
            printf("\n");
        }

        // 打印对应的距离矩阵D
        // D[i * k + j]表示第i个查询的第j个最近邻的距离
        // 对于相同的向量（查询即来自数据库本身），距离通常应为0
        printf("D=\n");
        for(int i = 0; i < 5; i++) {
            for(int j = 0; j < k; j++)
                printf("%7g ", D[i * k + j]);
            printf("\n");
        }

        // 释放临时分配的内存
        delete [] I;
        delete [] D;
    }


    {   // 对真正的查询向量进行搜索
        long *I = new long[k * nq];   // 存放查询结果的ID
        float *D = new float[k * nq]; // 存放查询结果的距离

        // 使用nq个查询向量xq进行搜索，每个查询返回k=4个最近邻
        index.search(nq, xq, k, D, I);

        // 打印查询结果的前5个查询的ID
        printf("I (5 first results)=\n");
        for(int i = 0; i < 5; i++) {
            for(int j = 0; j < k; j++)
                printf("%5ld ", I[i * k + j]);
            printf("\n");
        }

        // 再打印最后5个查询的结果ID
        printf("I (5 last results)=\n");
        for(int i = nq - 5; i < nq; i++) {
            for(int j = 0; j < k; j++)
                printf("%5ld ", I[i * k + j]);
            printf("\n");
        }

        // 释放内存
        delete [] I;
        delete [] D;
    }

    // 释放分配的向量数据内存
    delete [] xb;
    delete [] xq;

    return 0;
}