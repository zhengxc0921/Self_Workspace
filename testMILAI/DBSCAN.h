#pragma once
//#ifdef DBSCAN_EXPORTS
//#define DBSCAN_DECLSPEC __declspec(dllexport)
//#else
//#define DBSCAN_DECLSPEC __declspec(dllimport)
//#endif 

//#include<Mil.h>
#include"AIParse.h"
#include "nanoflann.hpp"
#define UNCLASSIFIED -2
#define CORE_POINT 1
#define BORDER_POINT 2
#define NOISE -1
#define SUCCESS 0
#define FAILURE -3

using namespace std;
class CDBSCAN;
typedef boost::shared_ptr<CDBSCAN>CDBSCANPtr;

class  CDBSCAN {
    public:
    CDBSCAN(MIL_ID MilSystem);

    virtual ~CDBSCAN();

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

    // construct a kd-tree index:
    template <typename num_t>
    using my_kd_tree_t = nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_MulDim_Adaptor<num_t, MulDimPointCloud<num_t>>,
        MulDimPointCloud<num_t>, 256 /* dim */>;

    template <typename num_t>
    int expandCluster(my_kd_tree_t<num_t>& index,
        MulDimPointCloud<num_t>& cloud, int pt_index,
        int clusterID, num_t search_radius, const int m_minPoints);

    template <typename num_t>
    void kdtree_dbscan(MulDimPointCloud<num_t>& cloud,
        const num_t radius,
        const int m_minPoints,
        vector<vector<int>>& Clst);

    void ImgCluster(double radius,
        int m_minPoints,
        string ImgDir,
        double AspectRatioTHD,
        vector<MIL_STRING>& efftImgPaths,
        vector<vector<int>>& Labels,
        vector<MIL_STRING>& unefftImgPaths);

    void saveClusterRst(MIL_STRING DstImgDir, vector<MIL_STRING>& efftImgPaths,
        vector<vector<int>>& Labels,
        vector<MIL_STRING>& unefftImgPaths);

    void removalImg(vector<MIL_STRING>& efftImgPaths,
        vector<vector<int>>& Labels,
        vector<MIL_STRING>& unefftImgPaths);

public:
    MIL_INT m_InSizeX = 16;
    MIL_INT m_InSizeY = 16;
    MIL_ID m_MilSystem;
    CAIParsePtr m_AIParse;
};


