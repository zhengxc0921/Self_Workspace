// nanoflann-PicDbScan.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <cstdlib>
#include <ctime>
#include <map>

#include <iostream>
#include <fstream>
#include <sstream>

#include "utils.h"
#include "nanoflann.hpp"

using namespace std;

template <typename num_t>
void readCloudFromTxt(MulDimPointCloud<num_t>& cloud)
{
    cloud.pts.resize(11087);

    string linestr;
    ifstream inf;
    inf.open("ImgsData.txt");
    size_t row = 0;
    while (getline(inf, linestr))  // getline(inf,s)是逐行读取inf中的文件信息
    {
        size_t col = 0;
        stringstream ss(linestr);  //存成二维表结构
        string str;  //每行中的单个字符
        while (getline(ss, str, ','))
        {
            cloud.pts[row].Array[col] = (num_t)atof(str.c_str());
            col++;
        }
        row++;
    }
    inf.close();
}

// construct a kd-tree index:
template <typename num_t>
using my_kd_tree_t = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_MulDim_Adaptor<num_t, MulDimPointCloud<num_t>>,
    MulDimPointCloud<num_t>, 256 /* dim */
>;

template <typename num_t>
void InterCluster(my_kd_tree_t<num_t>& index, MulDimPointCloud<num_t>& cloud, vector<uint32_t>& matches, size_t point_index, const num_t search_radius, size_t CluIndex)
{
    bool IsFirst = false;
    if (matches.size() == 0) IsFirst = true;

    num_t query_point[256];
    for (size_t j = 0; j < 256; j++)
        query_point[j] = cloud.pts[point_index].Array[j];

    vector<nanoflann::ResultItem<uint32_t, num_t>> ret_matches;
    const size_t nMatches =
        index.radiusSearch(&query_point[0], search_radius, ret_matches);
    cloud.pts[point_index].IsMerge = true;

    for (size_t i = 0; i < nMatches; i++) {
        if (cloud.pts[ret_matches[i].first].IsMark) continue;
        matches.push_back(ret_matches[i].first);
        cloud.pts[ret_matches[i].first].IsMark = true;
    }
    ret_matches.clear();
    ret_matches.shrink_to_fit();

    if (!IsFirst) 
        return; // 只深入迭代一次

    for (size_t i = 1; i < matches.size(); i++)
    {
        InterCluster(index, cloud, matches, matches[i], search_radius, CluIndex);
    }
}

//template <typename num_t>
//bool isVecSame(vector<num_t>v1, vector<num_t> v2)
//{
//    int nLen1 = v1.size();
//    int nLen2 = v1.size();
//    if (nLen1 != nLen2) {
//        return false;
//    }
//    for (int i = 0; i < nLen1; i++) {
//
//        if (v1[i] != v2[i]) {
//            return false;
//        }
//    }
//
//    return true;
//}



template <typename num_t>
void PicCluster(my_kd_tree_t<num_t>& index, MulDimPointCloud<num_t>& cloud, map<size_t, vector<uint32_t>>& TotalPicMap, const num_t search_radius)
{
    size_t CluIndex = 0;
    for (size_t i = 0; i < cloud.pts.size(); i++)
    {
        if (cloud.pts[i].IsMerge) continue;
        CluIndex++;
        vector<uint32_t> matches;  // 当前这块合并的像素区域内，包括的像素索引
        InterCluster(index, cloud, matches, i, search_radius, CluIndex);
        if (matches.size() < 60)
            continue;
        TotalPicMap.insert({ CluIndex, matches });
    }
}

template <typename num_t>
int expandCluster(my_kd_tree_t<num_t>& index,
    MulDimPointCloud<num_t>& cloud, int pt_index,
    int clusterID,  num_t search_radius, num_t m_minPoints)
{
    num_t query_point[256];
    for (size_t j = 0; j < 256; j++) { query_point[j] = cloud.pts[pt_index].Array[j]; }
        

    vector<nanoflann::ResultItem<uint32_t, num_t>> MatchSeeds;
    const size_t nMatcheSeeds =index.radiusSearch(&query_point[0], search_radius, MatchSeeds);

    vector<int> clusterSeeds;
    for (int i = 0; i < nMatcheSeeds; i++) {
        clusterSeeds.emplace_back(MatchSeeds[i].first); }

    if (nMatcheSeeds < m_minPoints)
    {
        cloud.pts[pt_index].clusterID = NOISE;
        return FAILURE;
    }
    else
    {
        int id = 0, indexCorePoint = 0;
        for (auto iterSeeds = clusterSeeds.begin(); iterSeeds != clusterSeeds.end(); ++iterSeeds)
        {
            cloud.pts.at(*iterSeeds).clusterID = clusterID;
            //判断两个vector相同

            bool f = true;
            for (int i = 0; i < 256; i++) {
            if (cloud.pts[*iterSeeds].Array[i] != query_point[i]) {f =  false;}}
            if (f)
            {
                indexCorePoint = id;
            }
            ++id;
        }
        clusterSeeds.erase(clusterSeeds.begin() + indexCorePoint);
        for (vector<int>::size_type i = 0, n = clusterSeeds.size(); i < n; ++i)
        {
            num_t query_nb[256];
            for (size_t j = 0; j < 256; j++) { query_nb[j] = cloud.pts[clusterSeeds[i]].Array[j]; }
            //const size_t nMNeighors = 0;
            //vector<nanoflann::ResultItem<uint32_t, num_t>> MatchNeighors;
            //nMNeighors = index.radiusSearch(&query_nb[0], search_radius, MatchNeighors);
      
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
void kdtree_dbscan(const num_t radius)
{
    MulDimPointCloud<num_t> cloud;
    readCloudFromTxt(cloud);

    // construct a kd-tree index:
    using my_kd_tree_t = nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_MulDim_Adaptor<num_t, MulDimPointCloud<num_t>>,
        MulDimPointCloud<num_t>, 256 /* dim */
    >;
    my_kd_tree_t index(256 /*dim*/, cloud, { 256 /* max leaf */ });

    //map<size_t, vector<uint32_t>> TotalPicMap;
    num_t search_radius = radius * radius * 256 * 256;
    num_t m_minPoints = 60;
    //size_t CluIndex = 0;
    int clusterID = 1;
    for (size_t i = 0; i < cloud.pts.size(); i++)
    {
        if (cloud.pts[i].clusterID == UNCLASSIFIED)
        {

            if (expandCluster(index, cloud, i , clusterID, search_radius, m_minPoints) != FAILURE)
            {
                clusterID += 1;
            }
        }
    }
    cout << "clusterID: " << clusterID << endl;


    vector<MIL_STRING>& vecImgPaths;
    string file
    ifstream in(file, ios::in);
    string line;
    while (getline(in, line)) {
        MIL_STRING Mline = string2MIL_STRING(line);
        vecImgPaths.emplace_back(Mline);
    }

}


template <typename num_t>
void kdtree_demo(const num_t radius)
{
    MulDimPointCloud<num_t> cloud;
    readCloudFromTxt(cloud);

    // construct a kd-tree index:
    using my_kd_tree_t = nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_MulDim_Adaptor<num_t, MulDimPointCloud<num_t>>,
        MulDimPointCloud<num_t>, 256 /* dim */
    >;
    my_kd_tree_t index(256 /*dim*/, cloud, { 256 /* max leaf */ });

    map<size_t, vector<uint32_t>> TotalPicMap;
    num_t search_radius = radius * radius * 256 * 256;
    PicCluster(index, cloud, TotalPicMap, search_radius);

    for (map<size_t, vector<uint32_t>>::iterator it_cur = TotalPicMap.begin(); it_cur != TotalPicMap.end(); it_cur++)
    {
        vector<uint32_t> temp = it_cur->second;
    }
}


int main()
{
    double radius = 1.2;
    //kdtree_demo<double>(radius);

    kdtree_dbscan<double>(radius);

    return 0;
}
