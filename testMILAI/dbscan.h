#pragma once
#ifndef DBSCAN_H
#define DBSCAN_H

#include <vector>
#include <cmath>

#define UNCLASSIFIED -1
#define CORE_POINT 1
#define BORDER_POINT 2
#define NOISE -2
#define SUCCESS 0
#define FAILURE -3

using namespace std;

struct DBPoint {
    vector<float>vecImgPixel;
    float x, y, z;  // X, Y, Z position
    int clusterID = UNCLASSIFIED;
};

class DBSCAN {
public:    
    DBSCAN(unsigned int minPts, float eps, vector<DBPoint> points){
        m_minPoints = minPts;
        m_epsilon = eps;
        m_points = points;
        m_pointSize = points.size();
    }
    ~DBSCAN(){}

    int run();
    vector<int> calculateCluster(DBPoint point);
    int expandCluster(DBPoint point, int clusterID);
    bool isVecSame(vector<float>v1, vector<float>v2);
    inline double calculateDistance(const DBPoint& pointCore, const DBPoint& pointTarget);


    int getTotalPointSize() {return m_pointSize;}
    int getMinimumClusterSize() {return m_minPoints;}
    int getEpsilonSize() {return m_epsilon;}
    
public:
    vector<DBPoint> m_points;
    int m_ClassNum;
    
private:    
    unsigned int m_pointSize;
    unsigned int m_minPoints;
    float m_epsilon;
  
};

#endif // DBSCAN_H
