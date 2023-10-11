#include "DBSCAN.h"


CDBSCAN::CDBSCAN(MIL_ID MilSystem) :m_MilSystem(MilSystem) {

    m_AIParse = CAIParsePtr(new CAIParse(MilSystem));
}
CDBSCAN::~CDBSCAN()
{
}
template<typename num_t>
 int CDBSCAN::expandCluster(my_kd_tree_t<num_t>& index, MulDimPointCloud<num_t>& cloud, int pt_index, int clusterID, num_t search_radius, const int m_minPoints)
{

     num_t query_point[256];
     for (size_t j = 0; j < 256; j++) { query_point[j] = cloud.pts[pt_index].Array[j]; }

     vector<nanoflann::ResultItem<uint32_t, num_t>> MatchSeeds;
     const size_t nMatcheSeeds = index.radiusSearch(&query_point[0], search_radius, MatchSeeds);

     vector<int> clusterSeeds;
     for (int i = 0; i < nMatcheSeeds; i++) {
         clusterSeeds.emplace_back(MatchSeeds[i].first);
     }

     if (nMatcheSeeds < m_minPoints)
     {
         cloud.pts[pt_index].clusterID = NOISE; return FAILURE;
     }
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

 template<typename num_t>
 void CDBSCAN::kdtree_dbscan(MulDimPointCloud<num_t>& cloud, const num_t radius, const int m_minPoints, vector<vector<int>>& Clst)
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

     //clusterNum = clusterID + 1;
     Clst.resize(clusterID + 1);
     for (int i = 0; i < cloud.pts.size(); i++) {
         int n = cloud.pts[i].clusterID + 1;
         Clst[n].emplace_back(i);
     }

 }


 void CDBSCAN::ImgCluster(double radius, 
     int m_minPoints,
     string ImgDir,
     double AspectRatioTHD,
     vector<MIL_STRING>& efftImgPaths,
     vector<vector<int>>& Labels, 
     vector<MIL_STRING>& unefftImgPaths)
 {
         clock_t start, finish;
         double  duration;
         start = clock();
         
         //double radius = 1.2;
         //int m_minPoints = 60;
         //string ImgDir = "G:/DefectDataCenter/Test/Src/90";
         //double AspectRatioTHD = 3;


         MulDimPointCloud<double> cloud;
         //针对黑白图的DBSCAN
         vector<MIL_STRING> ImgPaths;
         m_AIParse->getFilesInFolder(ImgDir, "bmp", ImgPaths);
         //分割出长宽比过大的图为非法图片，不参与训练
         //vector<MIL_STRING> unefftImgPaths;
         //vector<MIL_STRING> efftImgPaths;
         for (auto iter = ImgPaths.begin(); iter != ImgPaths.end(); iter++) {
     
             MIL_ID Image = MbufRestore(*iter, m_MilSystem, M_NULL);
             MIL_INT ImageSizeX = MbufInquire(Image, M_SIZE_X, M_NULL);
             MIL_INT ImageSizeY = MbufInquire(Image, M_SIZE_Y, M_NULL);
             MIL_INT ImageBand = MbufInquire(Image, M_SIZE_BAND, M_NULL);
     
             bool illegalImg = (double)max(ImageSizeX, ImageSizeY) / (double)min(ImageSizeX, ImageSizeY) > AspectRatioTHD;
             if (illegalImg || ImageBand > 2) {
                 unefftImgPaths.emplace_back(*iter);
                 continue;
             }
             efftImgPaths.emplace_back(*iter);
             MIL_ID ImageReduce = MbufAlloc2d(m_MilSystem, m_InSizeX, m_InSizeY, 8 + M_UNSIGNED, M_IMAGE + M_PROC, M_NULL);
             MimResize(Image, ImageReduce, M_FILL_DESTINATION, M_FILL_DESTINATION, M_BILINEAR);
     
             vector<MIL_UINT8>ImgPixels;
             ImgPixels.resize(m_InSizeX * m_InSizeY * ImageBand);
             MbufGet2d(ImageReduce, 0, 0, m_InSizeX, m_InSizeY, &ImgPixels[0]);
             MulDimPointCloud<double>::DBPoint TmpPts;
             for (int i = 0; i < ImgPixels.size(); i++) {
                 TmpPts.Array[i] = ImgPixels[i] / 1.0;
             }
             cloud.pts.emplace_back(TmpPts);
         }
     
         //int clusterNum;
         //vector<vector<int>> Labels;
         kdtree_dbscan<double>(cloud, radius, m_minPoints, Labels);
         finish = clock();
         duration = (double)(finish - start) / CLOCKS_PER_SEC;
         //cout << "duration: " << duration << endl;
         //////创建目的class的文件夹
         //MIL_STRING DstImgDir = L"G:/DefectDataCenter/Test/ImgCluster/Cpp_SPA90/";
         //m_AIParse->CreateFolder(DstImgDir);
     
         //for (int i = 0; i < clusterNum + 2; i++) {
         //    m_AIParse->CreateFolder(DstImgDir + to_wstring(i - 2));
         //}
         //////遍历所有文件及结果，并将图片保存到相应的Index
         //for (int i = 0; i < Labels.size(); i++) {
         //    vector<int> clst = Labels[i];
         //    for (int j = 0; j < clst.size(); j++) {
         //        MIL_STRING RawImagePath = efftImgPaths[clst[j]];
         //        MIL_STRING ClassNames = to_wstring(i);
         //        MIL_ID Image = MbufRestore(RawImagePath, m_MilSystem, M_NULL);
         //        string pth;
         //        m_AIParse->MIL_STRING2string(RawImagePath, pth);
         //        string::size_type iPos = pth.find_last_of('/') + 1;
         //        MIL_STRING ImageRawName = RawImagePath.substr(iPos, RawImagePath.length() - iPos);
         //        MIL_STRING DstRootPath = DstImgDir + ClassNames + MIL_TEXT("//") + ImageRawName;
         //        MbufExport(DstRootPath, M_BMP, Image);
         //        MbufFree(Image);
         //    }
         //}

 }

 void CDBSCAN::saveClusterRst(MIL_STRING DstImgDir,
     vector<MIL_STRING>& efftImgPaths, 
     vector<vector<int>>& Labels,
     vector<MIL_STRING>& unefftImgPaths)
 {
     ////创建目的class的文件夹
 
     m_AIParse->CreateFolder(DstImgDir);

     for (int i = 0; i < Labels.size() + 2; i++) {
         m_AIParse->CreateFolder(DstImgDir + to_wstring(i - 2));
     }
     //遍历所有聚类结果，并将图片保存到相应的Index
     for (int i = 0; i < Labels.size(); i++) {
         vector<int> clst = Labels[i];
         for (int j = 0; j < clst.size(); j++) {
             MIL_STRING RawImagePath = efftImgPaths[clst[j]];
             MIL_STRING ClassNames = to_wstring(i);
             MIL_ID Image = MbufRestore(RawImagePath, m_MilSystem, M_NULL);
             string pth;
             m_AIParse->MIL_STRING2string(RawImagePath, pth);
             string::size_type iPos = pth.find_last_of('/') + 1;
             MIL_STRING ImageRawName = RawImagePath.substr(iPos, RawImagePath.length() - iPos);
             MIL_STRING DstRootPath = DstImgDir + ClassNames + MIL_TEXT("//") + ImageRawName;
             MbufExport(DstRootPath, M_BMP, Image);
             MbufFree(Image);
         }
     }

     //遍历所有无效图片路径，存放入文件夹"-2"
     for (int i = 0; i < unefftImgPaths.size(); i++) {     
             MIL_STRING RawImagePath = efftImgPaths[i];
             MIL_ID Image = MbufRestore(RawImagePath, m_MilSystem, M_NULL);
             string pth;
             m_AIParse->MIL_STRING2string(RawImagePath, pth);
             string::size_type iPos = pth.find_last_of('/') + 1;
             MIL_STRING ImageRawName = RawImagePath.substr(iPos, RawImagePath.length() - iPos);
             MIL_STRING DstRootPath = DstImgDir  + MIL_TEXT("-2//") + ImageRawName;
             MbufExport(DstRootPath, M_BMP, Image);
             MbufFree(Image);
         }
     }

 void CDBSCAN::removalImg(vector<MIL_STRING>& efftImgPaths, vector<vector<int>>& Labels, vector<MIL_STRING>& unefftImgPaths)
 {
     //遍历所有聚类冗余结果，删除
     for (int i = 0; i < Labels.size(); i++) {
         vector<int> clst = Labels[i];
         for (int j = 0; j < clst.size(); j++) {
             MIL_STRING RawImagePath = efftImgPaths[clst[j]];
             string strImgPath;
             m_AIParse->MIL_STRING2string( RawImagePath, strImgPath);
             remove(strImgPath.c_str());
         }
     }
     //遍历所有无效图片路径，删除
     for (int i = 0; i < unefftImgPaths.size(); i++) {
         MIL_STRING RawImagePath = unefftImgPaths[i];
         string strImgPath;
         m_AIParse->MIL_STRING2string(RawImagePath, strImgPath);
         remove(strImgPath.c_str());
     }
 }
