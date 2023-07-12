#include "dbscan.h"

int DBSCAN::run()
{
    int clusterID = 1;
    vector<DBPoint>::iterator iter;
    for(iter = m_points.begin(); iter != m_points.end(); ++iter)
    {
        if ( iter->clusterID == UNCLASSIFIED )
        {
            if ( expandCluster(*iter, clusterID) != FAILURE )
            {
                clusterID += 1;
            }
        }
    }
    m_ClassNum = clusterID+1;
    return 0;
}

int DBSCAN::expandCluster(DBPoint point, int clusterID)
{    
    vector<int> clusterSeeds = calculateCluster(point);

    if ( clusterSeeds.size() < m_minPoints )
    {
        point.clusterID = NOISE;
        return FAILURE;
    }
    else
    {
        int index = 0, indexCorePoint = 0;
        vector<int>::iterator iterSeeds;
        for( iterSeeds = clusterSeeds.begin(); iterSeeds != clusterSeeds.end(); ++iterSeeds)
        {
            m_points.at(*iterSeeds).clusterID = clusterID;
            //判断两个vector相同
            if(isVecSame(m_points.at(*iterSeeds).vecImgPixel, point.vecImgPixel))
            //if (m_points.at(*iterSeeds).x == point.x && m_points.at(*iterSeeds).y == point.y && m_points.at(*iterSeeds).z == point.z )
            {
                indexCorePoint = index;
            }
            ++index;
        }
        clusterSeeds.erase(clusterSeeds.begin()+indexCorePoint);

        for( vector<int>::size_type i = 0, n = clusterSeeds.size(); i < n; ++i )
        {
            vector<int> clusterNeighors = calculateCluster(m_points.at(clusterSeeds[i]));

            if ( clusterNeighors.size() >= m_minPoints )
            {
                vector<int>::iterator iterNeighors;
                for ( iterNeighors = clusterNeighors.begin(); iterNeighors != clusterNeighors.end(); ++iterNeighors )
                {
                    if ( m_points.at(*iterNeighors).clusterID == UNCLASSIFIED || m_points.at(*iterNeighors).clusterID == NOISE )
                    {
                        if ( m_points.at(*iterNeighors).clusterID == UNCLASSIFIED )
                        {
                            clusterSeeds.push_back(*iterNeighors);
                            n = clusterSeeds.size();
                        }
                        m_points.at(*iterNeighors).clusterID = clusterID;
                    }
                }
            }
        }

        return SUCCESS;
    }
}

bool DBSCAN::isVecSame(vector<float> v1, vector<float> v2)
{
    int nLen1 = v1.size();
    int nLen2 = v1.size();
    if (nLen1 != nLen2) {
        return false;
    }
    for (int i=0; i < nLen1; i++) {

        if (v1[i]!=v2[i]) {
            return false;
        }
    }

    return true;
}

vector<int> DBSCAN::calculateCluster(DBPoint point)
{
    int index = 0;
    vector<DBPoint>::iterator iter;
    vector<int> clusterIndex;
    for( iter = m_points.begin(); iter != m_points.end(); ++iter)
    {
        if ( calculateDistance(point, *iter) <= m_epsilon )
        {
            clusterIndex.push_back(index);
        }
        index++;
    }
    return clusterIndex;
}

inline double DBSCAN::calculateDistance(const DBPoint& pointCore, const DBPoint& pointTarget )
{
    double nDist = 0;
    for (int i = 0; i < pointCore.vecImgPixel.size(); i++) {
        //double fDist_i = pow(pointCore.vecImgPixel[i] - pointTarget.vecImgPixel[i],2);

        double fDist_i = abs(pointCore.vecImgPixel[i] - pointTarget.vecImgPixel[i]);
        nDist += fDist_i;
    }
    return nDist;
    //return pow(pointCore.x - pointTarget.x,2)+pow(pointCore.y - pointTarget.y,2)+pow(pointCore.z - pointTarget.z,2);
}


