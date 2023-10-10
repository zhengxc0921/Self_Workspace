#pragma once
//#include <cstdlib>
//#include <ctime>
//#include <map>
//#include <fstream>
//#include <sstream>
//#include <iostream>

#include"AIParse.h"
#include "nanoflann.hpp"
#define UNCLASSIFIED -2
#define CORE_POINT 1
#define BORDER_POINT 2
#define NOISE -1
#define SUCCESS 0
#define FAILURE -3



template <typename T>
struct MulDimPointCloud
{
    struct DBPoint
    {
        T Array[256];
        //bool IsMerge = false; // 是否作为搜索中心，开展搜索
        //bool IsMark = false;  // 是否被以其他像素为中心的搜索，搜索到
        int clusterID = UNCLASSIFIED;
    };

    std::vector<DBPoint> pts;

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const { return pts.size(); }

    // Returns the dim'th component of the idx'th point in the class:
    // Since this is inlined and the "dim" argument is typically an immediate
    // value, the
    //  "if/else's" are actually solved at compile time.
    inline T kdtree_get_pt(const size_t idx, const size_t dim) const
    {
        return pts[idx].Array[dim];
    }

    // Optional bounding-box computation: return false to default to a standard
    // bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned
    //   in "bb" so it can be avoided to redo it again. Look at bb.size() to
    //   find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const
    {
        return false;
    }
};

//template <typename num_t>
//void readCloudFromTxt(MulDimPointCloud<num_t>& cloud)
//{
//    cloud.pts.resize(11087);
//
//    string linestr;
//    ifstream inf;
//    inf.open("ImgsData.txt");
//    size_t row = 0;
//    while (getline(inf, linestr))  // getline(inf,s)是逐行读取inf中的文件信息
//    {
//        size_t col = 0;
//        stringstream ss(linestr);  //存成二维表结构
//        string str;  //每行中的单个字符
//        while (getline(ss, str, ','))
//        {
//            cloud.pts[row].Array[col] = (num_t)atof(str.c_str());
//            col++;
//        }
//        row++;
//    }
//    inf.close();
//}

// construct a kd-tree index:
template <typename num_t>
using my_kd_tree_t = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_MulDim_Adaptor<num_t, MulDimPointCloud<num_t>>,
    MulDimPointCloud<num_t>, 256 /* dim */>;


template <typename num_t>
int expandCluster(my_kd_tree_t<num_t>& index,
    MulDimPointCloud<num_t>& cloud, int pt_index,
    int clusterID, num_t search_radius,const int m_minPoints)
{
    num_t query_point[256];
    for (size_t j = 0; j < 256; j++) { query_point[j] = cloud.pts[pt_index].Array[j]; }

    vector<nanoflann::ResultItem<uint32_t, num_t>> MatchSeeds;
    const size_t nMatcheSeeds = index.radiusSearch(&query_point[0], search_radius, MatchSeeds);

    vector<int> clusterSeeds;
    for (int i = 0; i < nMatcheSeeds; i++) {
        clusterSeeds.emplace_back(MatchSeeds[i].first);}

    if (nMatcheSeeds < m_minPoints)
    { cloud.pts[pt_index].clusterID = NOISE; return FAILURE;}
    else
    {
        //cloud.pts[pt_index].clusterID = clusterID;
        int id = 0, indexCorePoint = 0;
        for (auto iterSeeds = clusterSeeds.begin(); iterSeeds != clusterSeeds.end(); ++iterSeeds)
        {
            cloud.pts.at(*iterSeeds).clusterID = clusterID;
            ////判断两个vector相同
            //bool f = true;
            //for (int i = 0; i < 256; i++) {
            //    if (cloud.pts[*iterSeeds].Array[i] != query_point[i]) { f = false; }
            //}
            //if (f)
            //{ indexCorePoint = id; }
            //++id;
        }
        clusterSeeds.erase(clusterSeeds.begin() + indexCorePoint);
        for (vector<int>::size_type i = 0, n = clusterSeeds.size(); i < n; ++i)
        {
            num_t query_nb[256];
            for (size_t j = 0; j < 256; j++) { query_nb[j] = cloud.pts[clusterSeeds[i]].Array[j]; }

            vector<nanoflann::ResultItem<uint32_t, num_t>> MatchNeighors;
            const size_t nMNeighors = index.radiusSearch(&query_nb[0], search_radius, MatchNeighors);
            vector<int> clusterNeighors;
            for (int i = 0; i < nMNeighors; i++) {
                clusterNeighors.emplace_back(MatchNeighors[i].first);
            }
            //vector<int> clusterNeighors = calculateCluster(cloud.pts.at(clusterSeeds[i].first));
            if (nMNeighors >= m_minPoints)
            {
                //vector<int>::iterator iterNeighors;
                for (auto iterNeighors = clusterNeighors.begin(); iterNeighors != clusterNeighors.end(); ++iterNeighors)
                {
                    if (cloud.pts[*iterNeighors].clusterID == UNCLASSIFIED || cloud.pts[*iterNeighors].clusterID == NOISE)
                    {
                        if (cloud.pts[*iterNeighors].clusterID == UNCLASSIFIED)
                        {
                            clusterSeeds.push_back(*iterNeighors);
                            n = clusterSeeds.size();
                        }
                        cloud.pts[*iterNeighors].clusterID = clusterID;
                    }
                }
            }
        }
        return SUCCESS;
    }
}

template <typename num_t>
void kdtree_dbscan(MulDimPointCloud<num_t>& cloud,
    const num_t radius, 
    const int m_minPoints,
    vector<vector<int>>&Clst ,
    int& clusterNum)
{
    // construct a kd-tree index:
    using my_kd_tree_t = nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_MulDim_Adaptor<num_t, MulDimPointCloud<num_t>>,
        MulDimPointCloud<num_t>, 256 /* dim */>;
    my_kd_tree_t index(256 /*dim*/, cloud, { 256 /* max leaf */ });
    num_t search_radius = radius * radius * 256 * 256;

    int clusterID = 0;
    for (size_t i = 0; i < cloud.pts.size(); i++)
    {
        if (cloud.pts[i].clusterID == UNCLASSIFIED)
        {
            if (expandCluster(index, cloud, i, clusterID, search_radius, m_minPoints) != FAILURE)
            {
                clusterID += 1;
            }
        }
    }

    clusterNum = clusterID + 1;
    Clst.resize(clusterID+1);
    for (int i = 0; i < cloud.pts.size(); i++) {
        int n = cloud.pts[i].clusterID+1;
        Clst[n].emplace_back(i);
    }

}









