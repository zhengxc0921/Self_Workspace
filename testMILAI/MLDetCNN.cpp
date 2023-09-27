#include "MLDetCNN.h"
//#include <Shlwapi.h>
#include <fstream>


CMLDetCNN::CMLDetCNN(MIL_ID MilSystem, MIL_ID MilDisplay):
	m_MilSystem(MilSystem),
	m_MilDisplay(MilDisplay)
{
	m_AIParse = CAIParsePtr(new CAIParse(MilSystem));
}

CMLDetCNN::~CMLDetCNN()
{
}

//MIL_INT CMLDetCNN::CnnTrainEngineDLLInstalled(MIL_ID MilSystem)
//{
//	MIL_INT IsInstalled = M_FALSE;
//	MIL_UNIQUE_CLASS_ID TrainCtx = MclassAlloc(MilSystem, M_TRAIN_DET, M_DEFAULT, M_UNIQUE_ID);
//	MclassInquire(TrainCtx, M_DEFAULT, M_TRAIN_ENGINE_IS_INSTALLED + M_TYPE_MIL_INT, &IsInstalled);
//	return IsInstalled;
//}

void CMLDetCNN::CreateFolder(const MIL_STRING& FolderPath)
{
    MIL_INT FolderExists = M_NO;
    MappFileOperation(M_DEFAULT, FolderPath, M_NULL, M_NULL, M_FILE_EXISTS, M_DEFAULT, &FolderExists);
    if (FolderExists == M_NO)
    {
        MappFileOperation(M_DEFAULT, FolderPath, M_NULL, M_NULL, M_FILE_MAKE_DIR, M_DEFAULT, M_NULL);
    }
}

bool CMLDetCNN::isfileNotExist(string fileNmae) {
    ifstream f(fileNmae);
    return !f.good();
};

bool CMLDetCNN::isfileNotExist(MIL_STRING fileNmae) {
    ifstream f(fileNmae.c_str());
    return !f.good();
}
LPTSTR CMLDetCNN::string2LPCTSTR(string inString)
{
    const char* path = inString.c_str();
    int path_num = MultiByteToWideChar(0, 0, path, -1, NULL, 0);
    LPTSTR outString = new wchar_t[path_num];
    MultiByteToWideChar(0, 0, path, -1, outString, path_num);

    return  outString;

}
;

void CMLDetCNN::readDetDataSetConfig(string DetDataSetConfigPath, string proj_n)
{
    

    //lpPath : char* 转换成 LPCTSTR
    //const char* path = DetDataSetConfigPath.c_str();
    //int path_num = MultiByteToWideChar(0, 0, path, -1, NULL, 0);
    //LPTSTR lpPath = new wchar_t[path_num];
    //MultiByteToWideChar(0, 0, path, -1, lpPath, path_num);
    LPTSTR lpPath = string2LPCTSTR(DetDataSetConfigPath);
    LPTSTR lpProj_n =  string2LPCTSTR(proj_n);

    //const char* path = DetDataSetConfigPath.c_str();
    //int path_num = MultiByteToWideChar(0, 0, path, -1, NULL, 0);
    //LPTSTR lpPath = new wchar_t[path_num];
    //MultiByteToWideChar(0, 0, path, -1, lpPath, path_num);

    int nWcharSize = 100;
    LPTSTR ClassesPathbuf = new wchar_t[nWcharSize];
    LPTSTR IconDirbuf = new wchar_t[nWcharSize];
    LPTSTR TrainDataInfoPathbuf = new wchar_t[nWcharSize];
    LPTSTR ValDataInfoPathbuf = new wchar_t[nWcharSize];
    LPTSTR WorkingDataDirbuf = new wchar_t[nWcharSize];
    LPTSTR PreparedDataDirbuf = new wchar_t[nWcharSize];

    GetPrivateProfileString(lpProj_n, L"ClassesPath", L"", ClassesPathbuf, nWcharSize, lpPath);
    GetPrivateProfileString(lpProj_n, L"IconDir", L"", IconDirbuf, nWcharSize, lpPath);
    GetPrivateProfileString(lpProj_n, L"TrainDataInfoPath", L"", TrainDataInfoPathbuf, nWcharSize, lpPath);
    GetPrivateProfileString(lpProj_n, L"ValDataInfoPath", L"", ValDataInfoPathbuf, nWcharSize, lpPath);
    GetPrivateProfileString(lpProj_n, L"WorkingDataDir", L"", WorkingDataDirbuf, nWcharSize, lpPath);
    GetPrivateProfileString(lpProj_n, L"PreparedDataDir", L"", PreparedDataDirbuf, nWcharSize, lpPath);

    wstring wstrClassesPath(ClassesPathbuf);
    wstring wstrIconDir(IconDirbuf);
    wstring wstrTrainDataInfoPath(TrainDataInfoPathbuf);
    wstring wstrValDataInfoPath(ValDataInfoPathbuf);
    wstring wstrWorkingDataDir(WorkingDataDirbuf);
    wstring wstrPreparedDataDir(PreparedDataDirbuf);

    m_DetDataSetPara.ClassesPath = std::string(wstrClassesPath.begin(), wstrClassesPath.end());
    m_DetDataSetPara.IconDir = std::string(wstrIconDir.begin(), wstrIconDir.end());
    m_DetDataSetPara.TrainDataInfoPath = std::string(wstrTrainDataInfoPath.begin(), wstrTrainDataInfoPath.end());
    m_DetDataSetPara.ValDataInfoPath = std::string(wstrValDataInfoPath.begin(), wstrValDataInfoPath.end());
    m_DetDataSetPara.WorkingDataDir = std::string(wstrWorkingDataDir.begin(), wstrWorkingDataDir.end());
    m_DetDataSetPara.PreparedDataDir = std::string(wstrPreparedDataDir.begin(), wstrPreparedDataDir.end());

    m_DetDataSetPara.ImageSizeX = GetPrivateProfileInt(lpProj_n, L"ImageSizeX", 0, lpPath);
    m_DetDataSetPara.ImageSizeY = GetPrivateProfileInt(lpProj_n, L"ImageSizeY", 0, lpPath);
    m_DetDataSetPara.TestDataRatio = GetPrivateProfileInt(lpProj_n, L"TestDataRatio", 0, lpPath);
    m_DetDataSetPara.AugFreq = GetPrivateProfileInt(lpProj_n, L"AugFreq", 0, lpPath);

    delete[] lpPath;
}

void CMLDetCNN::addInfo2Dataset(MIL_UNIQUE_CLASS_ID& Dataset)
{
   MIL_STRING MStrIconDir = m_AIParse->string2MIL_STRING(m_DetDataSetPara.IconDir);
    //m_AIParse->MIL_STRING2string(MStrWorkingDataPath, WorkingDataPath); 
    MclassControl(Dataset, M_DEFAULT, M_AUTHOR_ADD, MIL_TEXT("ZXC"));
    //step1:txt-->IconDataInfo
    vector<MIL_STRING>vecClasses;
    m_AIParse->readClasses2Vector(m_DetDataSetPara.ClassesPath, vecClasses);
    for (int i = 0; i < vecClasses.size(); i++) {
        MIL_STRING ClassIcon = MStrIconDir + vecClasses[i] + L".BMP";
        MclassControl(Dataset, M_DEFAULT, M_CLASS_ADD, vecClasses[i]);
        MIL_UNIQUE_BUF_ID IconImageId = MbufRestore(ClassIcon, m_MilSystem, M_UNIQUE_ID);
        MclassControl(Dataset, M_CLASS_INDEX(i), M_CLASS_ICON_ID, IconImageId);
    }

    //step2:txt-->ImgDataInfo
    vector<MIL_STRING> vecImgPaths;
    vector<vector<Box>> vec2Boxes;
    vector<vector<int>> veclabels;
    m_AIParse->readDataSet2Vector(m_DetDataSetPara.TrainDataInfoPath, vecImgPaths, vec2Boxes, veclabels);
    int nImgNum = vecImgPaths.size();
    vecImgPaths.resize(nImgNum);
    vec2Boxes.resize(nImgNum);
    veclabels.resize(nImgNum);
    for (int i = 0; i < nImgNum; i++) {
        MclassControl(Dataset, M_DEFAULT, M_ENTRY_ADD, M_DEFAULT);
        MclassControlEntry(Dataset, i, M_DEFAULT_KEY, M_DEFAULT, M_ENTRY_IMAGE_PATH, M_DEFAULT, vecImgPaths[i], M_DEFAULT);
        vector<Box> tmpBoxes = vec2Boxes[i];
        vector<int> tmplabels = veclabels[i];
        int nTempLen = tmpBoxes.size();
        for (int j = 0; j < nTempLen; j++) {
            MIL_UNIQUE_GRA_ID  MilGraphicList = MgraAllocList(m_MilSystem, M_DEFAULT, M_UNIQUE_ID);
            MgraRect(M_DEFAULT, MilGraphicList, tmpBoxes[j].x1, tmpBoxes[j].y1, tmpBoxes[j].x2, tmpBoxes[j].y2);
            MclassEntryAddRegion(Dataset, i, M_DEFAULT_KEY, M_DESCRIPTOR_TYPE_BOX, MilGraphicList, M_NULL, tmplabels[j], M_DEFAULT);
        }
    }
    MIL_STRING MStrWorkingDataDir = m_AIParse->string2MIL_STRING(m_DetDataSetPara.WorkingDataDir);
    CreateFolder(MStrWorkingDataDir);
 
}

int CMLDetCNN::predictPrepare(MIL_STRING TdDetCtxPath) {
    m_TrainedDetCtx = MclassRestore(TdDetCtxPath, m_MilSystem, M_DEFAULT, M_UNIQUE_ID);
    //获取模型输入尺寸
    MclassInquire(m_TrainedDetCtx, M_DEFAULT_SOURCE_LAYER, M_SIZE_X + M_TYPE_MIL_INT, &m_InputSizeX);
    MclassInquire(m_TrainedDetCtx, M_DEFAULT_SOURCE_LAYER, M_SIZE_Y + M_TYPE_MIL_INT, &m_InputSizeY);
    MclassInquire(m_TrainedDetCtx, M_CONTEXT, M_NUMBER_OF_CLASSES + M_TYPE_MIL_INT, &m_ClassesNum);
    MclassInquire(m_TrainedDetCtx, M_DEFAULT_SOURCE_LAYER, M_SIZE_BAND + M_TYPE_MIL_INT, &m_InputSizeBand);
    return 0;
}

int CMLDetCNN::predictPrepare(MIL_UNIQUE_CLASS_ID& TrainedDetCtx) {

    //获取模型输入尺寸
    MclassInquire(TrainedDetCtx, M_DEFAULT_SOURCE_LAYER, M_SIZE_X + M_TYPE_MIL_INT, &m_InputSizeX);
    MclassInquire(TrainedDetCtx, M_DEFAULT_SOURCE_LAYER, M_SIZE_Y + M_TYPE_MIL_INT, &m_InputSizeY);
    MclassInquire(TrainedDetCtx, M_CONTEXT, M_NUMBER_OF_CLASSES + M_TYPE_MIL_INT, &m_ClassesNum);
    MclassInquire(TrainedDetCtx, M_DEFAULT_SOURCE_LAYER, M_SIZE_BAND + M_TYPE_MIL_INT, &m_InputSizeBand);
    m_ModelNotPrePared = FALSE;

    return 0;
}

void CMLDetCNN::predict(MIL_ID Image, DET_RESULT_STRUCT& Result)
{
    MIL_ID ImageReduce;
    if (m_InputSizeBand == 3) {
        ImageReduce = MbufAllocColor(m_MilSystem, 3, m_InputSizeX, m_InputSizeY, 8 + M_UNSIGNED, M_IMAGE + M_PROC, M_NULL);
    }
    else {
        ImageReduce = MbufAlloc2d(m_MilSystem, m_InputSizeX, m_InputSizeY, 8 + M_UNSIGNED, M_IMAGE + M_PROC, M_NULL);
    }

    MimResize(Image, ImageReduce, M_FILL_DESTINATION, M_FILL_DESTINATION, M_BILINEAR);

    MIL_INT Status = M_FALSE;
    MclassInquire(m_TrainedDetCtx, M_DEFAULT, M_PREPROCESSED + M_TYPE_MIL_INT, &Status);
    if (M_FALSE == Status)
    {
        MclassPreprocess(m_TrainedDetCtx, M_DEFAULT);
    }
    MIL_UNIQUE_CLASS_ID ClassRes = MclassAllocResult(m_MilSystem, M_PREDICT_DET_RESULT, M_DEFAULT, M_UNIQUE_ID);
    MclassPredict(m_TrainedDetCtx, ImageReduce, ClassRes, M_DEFAULT);

    MbufFree(ImageReduce);
    MclassGetResult(ClassRes, M_GENERAL, M_NUMBER_OF_INSTANCES + M_TYPE_MIL_INT, &Result.InstanceNum);
    Result.Boxes.resize(Result.InstanceNum);
    Result.ClassIndex.resize(Result.InstanceNum);
    Result.Score.resize(Result.InstanceNum);
    Result.ClassName.resize(Result.InstanceNum);
    for (int i = 0; i < Result.InstanceNum; i++) {

        MclassGetResult(ClassRes, M_INSTANCE_INDEX(i), M_CENTER_X + M_TYPE_MIL_DOUBLE, &Result.Boxes[i].CX);
        MclassGetResult(ClassRes, M_INSTANCE_INDEX(i), M_CENTER_Y + M_TYPE_MIL_DOUBLE, &Result.Boxes[i].CY);
        MclassGetResult(ClassRes, M_INSTANCE_INDEX(i), M_HEIGHT + M_TYPE_MIL_DOUBLE, &Result.Boxes[i].H);
        MclassGetResult(ClassRes, M_INSTANCE_INDEX(i), M_WIDTH + M_TYPE_MIL_DOUBLE, &Result.Boxes[i].W);

        Result.Boxes[i].x1 = int(Result.Boxes[i].CX - Result.Boxes[i].W / 2);
        Result.Boxes[i].x2 = int(Result.Boxes[i].CX + Result.Boxes[i].W / 2);
        Result.Boxes[i].y1 = int(Result.Boxes[i].CY - Result.Boxes[i].H / 2);
        Result.Boxes[i].y2 = int(Result.Boxes[i].CY + Result.Boxes[i].H / 2);

        MIL_INT tmpClassIndex;
        MclassGetResult(ClassRes, M_INSTANCE_INDEX(i), M_BEST_CLASS_INDEX + M_TYPE_MIL_INT, &tmpClassIndex);
        Result.ClassIndex[i] = int(tmpClassIndex);

        MclassGetResult(ClassRes, M_INSTANCE_INDEX(i), M_BEST_CLASS_SCORE + M_TYPE_MIL_DOUBLE, &Result.Score[i]);
        MclassInquire(m_TrainedDetCtx, M_CLASS_INDEX(Result.ClassIndex[i]), M_CLASS_NAME, Result.ClassName[i]);
    }
}

void CMLDetCNN::saveResult2File(string strFilePath, vector<MIL_STRING>FilesInFolder, vector<DET_RESULT_STRUCT> vecDetResults) {

    //将结果保存到txt文件
    ofstream ODNetResult;
    int nFileNum = vecDetResults.size();
    ODNetResult.open(strFilePath, ios::out);
    for (int i = 0; i < nFileNum; i++) {
        string ImgInfo;
        m_AIParse->MIL_STRING2string(FilesInFolder[i], ImgInfo);
        //写入图片路径、box、conf、classname
        DET_RESULT_STRUCT R_i = vecDetResults[i];
        for (int j = 0; j < R_i.Boxes.size(); j++) {
            string strClassName;
            m_AIParse->MIL_STRING2string(R_i.ClassName[j], strClassName);
            ImgInfo = ImgInfo + " " + to_string(R_i.Boxes[j].CX)
                + " " + to_string(R_i.Boxes[j].CY)
                + " " + to_string(R_i.Boxes[j].W)
                + " " + to_string(R_i.Boxes[j].H)
                + " " + to_string(R_i.Score[j])
                + " " + strClassName
                ;
        }
        ODNetResult << ImgInfo << endl;
    }

    ODNetResult.close();
}

void CMLDetCNN::calcAP4Vector(vector<vector<Box>> vecGTBoxes,
    vector<vector<int>> vecGTlabels,
    vector<vector<Box>> vecPdBoxes, 
    vector<vector<int>> vecPdlabels,
    vector<float>&Precisions,
vector<float>&Recalls)
{
  
    //step1:构建mapGTClassNums/mapTPClassNums/mapFPClassNums
    //预测结果中TP/FP中的Class若不在GT中则不被考虑进去
    map<int, int>mapGTClassNums;
    map<int, int>mapTPClassNums;
    map<int, int>mapFPClassNums;
    for (int i = 0; i < vecGTlabels.size(); i++) {
        for (int j = 0; j < vecGTlabels[i].size(); j++) {
            int tmplabel = vecGTlabels[i][j];
            if (mapGTClassNums.find(tmplabel) == mapGTClassNums.end())
            {
                mapGTClassNums.insert(pair<int, int>(tmplabel, 1));
                mapTPClassNums.insert(pair<int, int>(tmplabel, 0));
                mapFPClassNums.insert(pair<int, int>(tmplabel, 0));
            }
            else {
                mapGTClassNums[tmplabel] += 1;
            }
        }
    }
    //step2:需保证4个vector的size一致，遍历统计每张图中的结果
    for (int i = 0; i < vecGTlabels.size(); i++) {  
        //map<int, int>mapTmpTPClassNums;
        //map<int, int>mapTmpFPClassNums;
        vector<Box>vecImgGTBoxes = vecGTBoxes[i];
        vector<int>vecImgGTlabels = vecGTlabels[i];
        vector<Box>vecImgPdBoxes = vecPdBoxes[i];
        vector<int>vecImgPdlabels = vecPdlabels[i];
        //记录某个GT被查找的次数，
        //若被多次查找，需调整mapTPClassNums、mapFPClassNums
        vector<int>vecFindTimes(vecImgGTlabels.size(),0);
        //按照vecImgPdlabels去遍历查找vecImgGTlabels
        for (int j = 0; j < vecImgPdlabels.size(); j++) {
            int pdlabel = vecImgPdlabels[j];
            Box pdBox = vecImgPdBoxes[j];
            //查找pdlabel在vecImgGTlabels中所有位置ids
            vector<int> ids;
            findLabelIndexs(pdlabel, vecImgGTlabels, ids);
            //找到对应ids的GTBoxes进行匹配
            for (auto it = ids.begin(); it != ids.end(); it++) {
                //若匹配成功，跳出该匹配循环，GT的*it处查找计数+1
                if (matchTwoBoxes(pdBox, vecImgGTBoxes[*it])) {
                    vecFindTimes[*it] += 1;
                    //GT的*it处查找计数>1,说明该label为FP(和其他TP_box重合了,重复计算了)
                    if (vecFindTimes[*it] > 1) {
                        mapFPClassNums[pdlabel] += 1;
                    }
                    else {
                        mapTPClassNums[pdlabel] += 1;
                    }      
                };
            }
            //Box pdBoxes = vecImgPdBoxes[j];          
        }
    }
    //step3:根据GT、TP、FP计算APs

    vector<int>GT;
    vector<int>TP;
    vector<int>FP;
    for (auto &it : mapGTClassNums) {
        GT.emplace_back(it.second);
    }
    for (auto& it : mapTPClassNums) {
        TP.emplace_back(it.second);
    }
    for (auto& it : mapFPClassNums) {
        FP.emplace_back(it.second);
    }
    for (int i = 0; i < GT.size(); i++) {

        float precision = (float)TP[i] / max((float)(TP[i] + FP[i]),1.0);
        float recall = (float)TP[i] / (float)GT[i];
        Precisions.emplace_back(precision);
        Recalls.emplace_back(recall);
    }
}

void CMLDetCNN::findLabelIndexs(int label, vector<int> vecLabels, vector<int>& ids)
{

    for (auto it = vecLabels.begin(); it != vecLabels.end(); it++) {
        if (*it == label) {
            ids.emplace_back(distance(vecLabels.begin(),it));
        }
}

}

bool CMLDetCNN::matchTwoBoxes(Box bx1, Box bx2)
{
    int min_x = max(bx1.x1, bx2.x1);  // 找出左上角坐标哪个大
    int max_x = min(bx1.x2, bx2.x2);  // 找出右上角坐标哪个小
    int min_y = max(bx1.y1, bx2.y1);
    int max_y = min(bx1.y2, bx2.y2);
    if (min_x >= max_x || min_y >= max_y) // 如果没有重叠
        return false;
    float over_area = (max_x - min_x) * (max_y - min_y);  // 计算重叠面积
    float area_a = (bx1.x2 - bx1.x1) * (bx1.y2 - bx1.y1);
    float area_b = (bx2.x2 - bx2.x1) * (bx2.y2 - bx2.y1);
    float iou = over_area / (area_a + area_b - over_area);
    return iou>= m_IOU_threshold;
}

void CMLDetCNN::GenDataSet(string DetDataSetConfigPath,string proj_n)
{
    //读取 DetDataSetConfig
    readDetDataSetConfig(DetDataSetConfigPath,  proj_n);
    //写入WorkingDataset
    MIL_UNIQUE_CLASS_ID  WorkingDataset = MclassAlloc(m_MilSystem, M_DATASET_IMAGES, M_DEFAULT, M_UNIQUE_ID);
    addInfo2Dataset(WorkingDataset);
    ////*******************************必须参数*******************************//
    MIL_UNIQUE_CLASS_ID DataContext = MclassAlloc(m_MilSystem, M_PREPARE_IMAGES_DET, M_DEFAULT, M_UNIQUE_ID);
    DataContextParasStruct DataCtxParas;
    DataCtxParas.ImageSizeX = m_DetDataSetPara.ImageSizeX;
    DataCtxParas.ImageSizeY = m_DetDataSetPara.ImageSizeY;
    DataCtxParas.DstFolderMode = 1;
    DataCtxParas.PreparedDataFolder = m_AIParse->string2MIL_STRING(m_DetDataSetPara.PreparedDataDir);
    memset(&DataCtxParas.AugParas, 0, sizeof(AugmentationParasStruct));
    DataCtxParas.AugParas.AugmentationNumPerImage = m_DetDataSetPara.AugFreq;
    ConstructDataContext(DataCtxParas, DataContext);
    MIL_UNIQUE_CLASS_ID PreparedDataset = MclassAlloc(m_MilSystem, M_DATASET_IMAGES, M_DEFAULT, M_UNIQUE_ID);
    PrepareDataset(DataContext, WorkingDataset, PreparedDataset, m_AIParse->string2MIL_STRING(m_DetDataSetPara.WorkingDataDir), m_DetDataSetPara.TestDataRatio);

}

void CMLDetCNN::GenDataSet(DET_DATASET_PARAS_STRUCT DetDataSetPara)
{
    //赋值m_DetDataSetPara
    m_DetDataSetPara.ClassesPath = DetDataSetPara.ClassesPath;
    m_DetDataSetPara.IconDir = DetDataSetPara.IconDir;
    m_DetDataSetPara.TrainDataInfoPath = DetDataSetPara.TrainDataInfoPath;
    //m_DetDataSetPara.ValDataInfoPath = DetDataSetPara.ValDataInfoPath;   //在MIL中无效，Val自动从Train中分割出来
    m_DetDataSetPara.WorkingDataDir = DetDataSetPara.WorkingDataDir;
    m_DetDataSetPara.PreparedDataDir = DetDataSetPara.PreparedDataDir;
    m_DetDataSetPara.ImageSizeX = DetDataSetPara.ImageSizeX;
    m_DetDataSetPara.ImageSizeY = DetDataSetPara.ImageSizeY;
    m_DetDataSetPara.TestDataRatio = DetDataSetPara.TestDataRatio;
    m_DetDataSetPara.AugFreq = DetDataSetPara.AugFreq;


    //写入WorkingDataset
    MIL_UNIQUE_CLASS_ID  WorkingDataset = MclassAlloc(m_MilSystem, M_DATASET_IMAGES, M_DEFAULT, M_UNIQUE_ID);
    addInfo2Dataset(WorkingDataset);
    ////*******************************必须参数*******************************//
    MIL_UNIQUE_CLASS_ID DataContext = MclassAlloc(m_MilSystem, M_PREPARE_IMAGES_DET, M_DEFAULT, M_UNIQUE_ID);
    DataContextParasStruct DataCtxParas;
    DataCtxParas.ImageSizeX = m_DetDataSetPara.ImageSizeX;
    DataCtxParas.ImageSizeY = m_DetDataSetPara.ImageSizeY;
    DataCtxParas.DstFolderMode = 1;
    DataCtxParas.PreparedDataFolder = m_AIParse->string2MIL_STRING(m_DetDataSetPara.PreparedDataDir);
    memset(&DataCtxParas.AugParas, 0, sizeof(AugmentationParasStruct));
    DataCtxParas.AugParas.AugmentationNumPerImage = m_DetDataSetPara.AugFreq;
    ConstructDataContext(DataCtxParas, DataContext);
    MIL_UNIQUE_CLASS_ID PreparedDataset = MclassAlloc(m_MilSystem, M_DATASET_IMAGES, M_DEFAULT, M_UNIQUE_ID);
    PrepareDataset(DataContext, WorkingDataset, PreparedDataset, m_AIParse->string2MIL_STRING(m_DetDataSetPara.WorkingDataDir), m_DetDataSetPara.TestDataRatio);

}

void CMLDetCNN::ConstructDataContext(DataContextParasStruct DataCtxParas, MIL_UNIQUE_CLASS_ID& PrepareDataCtx)
{    

    if (DataCtxParas.ImageSizeX > 0 && DataCtxParas.ImageSizeY > 0)
    {
        MclassControl(PrepareDataCtx, M_CONTEXT, M_SIZE_MODE, M_USER_DEFINED);
        MclassControl(PrepareDataCtx, M_CONTEXT, M_SIZE_X, DataCtxParas.ImageSizeX);
        MclassControl(PrepareDataCtx, M_CONTEXT, M_SIZE_Y, DataCtxParas.ImageSizeY);
        m_ImageSizeX = DataCtxParas.ImageSizeX;
        m_ImageSizeY = DataCtxParas.ImageSizeY;
    }
    //MclassControl(DataContext, M_CONTEXT, M_RESIZE_SCALE_FACTOR, M_FILL_DESTINATION);
    if (DataCtxParas.DstFolderMode == 1)
    {
        MclassControl(PrepareDataCtx, M_CONTEXT, M_DESTINATION_FOLDER_MODE, M_OVERWRITE);
    }
    ////数据保存
    CreateFolder(DataCtxParas.PreparedDataFolder);
    MclassControl(PrepareDataCtx, M_CONTEXT, M_PREPARED_DATA_FOLDER, DataCtxParas.PreparedDataFolder);
    // On average, we do two augmentations per image + the original images.
    MclassControl(PrepareDataCtx, M_CONTEXT, M_AUGMENT_NUMBER_FACTOR, DataCtxParas.AugParas.AugmentationNumPerImage);
    //// Ensure repeatability with a fixed seed.
    MclassControl(PrepareDataCtx, M_CONTEXT, M_RESIZE_SCALE_FACTOR, M_FILL_DESTINATION);

    //MIL_ID PrepareDataCtx{ M_NULL };
    //MclassInquire(TrainCtx, M_DEFAULT, M_PREPARE_DATA_CONTEXT_ID + M_TYPE_MIL_ID, &PrepareDataCtx);

    // Reproducibility.
    MclassControl(PrepareDataCtx, M_DEFAULT, M_SEED_MODE, M_USER_DEFINED);
    MclassControl(PrepareDataCtx, M_DEFAULT, M_SEED_VALUE, 16);

    //// Presets.
    MclassControl(PrepareDataCtx, M_DEFAULT, M_PRESET_TRANSLATION, M_ENABLE);
    MclassControl(PrepareDataCtx, M_DEFAULT, M_PRESET_FLIP, M_ENABLE);

    MIL_ID AugmentContext{ M_NULL };
    MclassInquire(PrepareDataCtx, M_DEFAULT, M_AUGMENT_CONTEXT_ID + M_TYPE_MIL_ID, &AugmentContext);

    //// Chosen probability to achieve on average 1.75 of the following augmentations 
    MIL_INT Probability = 35;

    MimControl(AugmentContext, M_AUG_HUE_OFFSET_OP, M_ENABLE);
    MimControl(AugmentContext, M_AUG_HUE_OFFSET_OP + M_PROBABILITY, Probability);
    MimControl(AugmentContext, M_AUG_HUE_OFFSET_OP_MAX, 360);
    MimControl(AugmentContext, M_AUG_HUE_OFFSET_OP_MIN, 0);

    MimControl(AugmentContext, M_AUG_LIGHTING_DIRECTIONAL_OP, M_ENABLE);
    MimControl(AugmentContext, M_AUG_LIGHTING_DIRECTIONAL_OP + M_PROBABILITY, Probability);
    MimControl(AugmentContext, M_AUG_LIGHTING_DIRECTIONAL_OP_INTENSITY_MAX, 1.2);
    MimControl(AugmentContext, M_AUG_LIGHTING_DIRECTIONAL_OP_INTENSITY_MIN, 0.8);

    MimControl(AugmentContext, M_AUG_INTENSITY_ADD_OP, M_ENABLE);
    MimControl(AugmentContext, M_AUG_INTENSITY_ADD_OP + M_PROBABILITY, Probability);
    MimControl(AugmentContext, M_AUG_INTENSITY_ADD_OP_DELTA, 32);
    MimControl(AugmentContext, M_AUG_INTENSITY_ADD_OP_MODE, M_LUMINANCE);
    MimControl(AugmentContext, M_AUG_INTENSITY_ADD_OP_VALUE, 0);

    MimControl(AugmentContext, M_AUG_SATURATION_GAIN_OP, M_ENABLE);
    MimControl(AugmentContext, M_AUG_SATURATION_GAIN_OP + M_PROBABILITY, Probability);
    MimControl(AugmentContext, M_AUG_SATURATION_GAIN_OP_MAX, 1.5);
    MimControl(AugmentContext, M_AUG_SATURATION_GAIN_OP_MIN, 0.75);

    MimControl(AugmentContext, M_AUG_NOISE_MULTIPLICATIVE_OP, M_ENABLE);
    MimControl(AugmentContext, M_AUG_NOISE_MULTIPLICATIVE_OP + M_PROBABILITY, Probability);
    MimControl(AugmentContext, M_AUG_NOISE_MULTIPLICATIVE_OP_DISTRIBUTION, M_UNIFORM);
    MimControl(AugmentContext, M_AUG_NOISE_MULTIPLICATIVE_OP_INTENSITY_MIN, 0);
    MimControl(AugmentContext, M_AUG_NOISE_MULTIPLICATIVE_OP_STDDEV, 0.1);
    MimControl(AugmentContext, M_AUG_NOISE_MULTIPLICATIVE_OP_STDDEV_DELTA, 0.1);

    //// Hook to show augmentations' progress.
    bool IsDevDataset = false;
    MclassHookFunction(PrepareDataCtx, M_PREPARE_ENTRY_POST, DetHookNumPreparedEntriesFunc, &IsDevDataset);
}

void CMLDetCNN::PrepareDataset(MIL_UNIQUE_CLASS_ID& DatasetContext, 
    MIL_UNIQUE_CLASS_ID& PrepareDataset,
    MIL_UNIQUE_CLASS_ID& PreparedDataset,
    MIL_STRING WorkingDataPath,
    MIL_DOUBLE TestDatasetPercentage)
{
    MclassPreprocess(DatasetContext, M_DEFAULT);
    MclassPrepareData(DatasetContext, PrepareDataset, PreparedDataset, M_NULL, M_DEFAULT);

    MIL_UNIQUE_CLASS_ID WorkingDataset = MclassAlloc(m_MilSystem, M_DATASET_IMAGES, M_DEFAULT, M_UNIQUE_ID);
    MIL_UNIQUE_CLASS_ID TestDataset = MclassAlloc(m_MilSystem, M_DATASET_IMAGES, M_DEFAULT, M_UNIQUE_ID);
    MclassSplitDataset(M_SPLIT_CONTEXT_FIXED_SEED, PreparedDataset, WorkingDataset, TestDataset,
        100.0 - TestDatasetPercentage, M_NULL, M_DEFAULT);
    //保存结果
    MclassSave(WorkingDataPath + MIL_TEXT("WorkingDataset.mclassd"), WorkingDataset, M_DEFAULT);
    MclassSave(WorkingDataPath + MIL_TEXT("TestDataset.mclassd"), TestDataset, M_DEFAULT);

    CreateFolder(WorkingDataPath + MIL_TEXT("Train_entries"));
    CreateFolder(WorkingDataPath + MIL_TEXT("Test_entries"));
    MclassExport(WorkingDataPath + MIL_TEXT("Train_entries"), M_IMAGE_DATASET_FOLDER, WorkingDataset, M_DEFAULT, M_COMPLETE, M_DEFAULT);
    MclassExport(WorkingDataPath + MIL_TEXT("Test_entries"), M_IMAGE_DATASET_FOLDER, TestDataset, M_DEFAULT, M_COMPLETE, M_DEFAULT);
}

void CMLDetCNN::ConstructTrainCtx(DET_TRAIN_STRUCT ClassifierParas, MIL_UNIQUE_CLASS_ID& TrainCtx)
{
    MIL_DOUBLE MPF;
    if (M_NULL == TrainCtx)
    {
        TrainCtx = MclassAlloc(m_MilSystem, M_TRAIN_DET, M_DEFAULT, M_UNIQUE_ID);
    }

    MclassControl(TrainCtx, M_CONTEXT, M_TRAIN_DESTINATION_FOLDER, ClassifierParas.TrainDstFolder);

    if (ClassifierParas.TrainMode == 1)
    {
        MclassControl(TrainCtx, M_CONTEXT, M_RESET_TRAINING_VALUES, M_FINE_TUNING);
    }
    else if (ClassifierParas.TrainMode == 2)
    {
        MclassControl(TrainCtx, M_CONTEXT, M_RESET_TRAINING_VALUES, M_TRANSFER_LEARNING);
    }

    if (ClassifierParas.SplitPercent > 0)
    {
        //MclassControl(TrainCtx, M_DEFAULT, M_SPLIT_SEED_MODE, M_FIXED);
        // Since we are performing a single dataset train, 
        //the dataset will be split into train/dev by the following percentage.
        MclassControl(TrainCtx, M_DEFAULT, M_SPLIT_PERCENTAGE, ClassifierParas.SplitPercent);
    }

    if (ClassifierParas.MaxNumberOfEpoch > 0)
    {
        MclassControl(TrainCtx, M_DEFAULT, M_MAX_EPOCH, ClassifierParas.MaxNumberOfEpoch);
    }

    if (ClassifierParas.MiniBatchSize > 0)
    {
        MclassControl(TrainCtx, M_DEFAULT, M_MINI_BATCH_SIZE, ClassifierParas.MiniBatchSize);
    }

    if (ClassifierParas.SchedulerType == 1)
    {
        MclassControl(TrainCtx, M_DEFAULT, M_SCHEDULER_TYPE, M_DECAY);
    }

    if (ClassifierParas.LearningRate > 0)
    {
        MclassControl(TrainCtx, M_DEFAULT, M_INITIAL_LEARNING_RATE, ClassifierParas.LearningRate);
    }

    if (ClassifierParas.LearningRateDecay > 0)
    {
        MclassControl(TrainCtx, M_DEFAULT, M_LEARNING_RATE_DECAY, ClassifierParas.LearningRate);
    }

    if (ClassifierParas.TrainEngineUsed == 1)
    {
        MclassControl(TrainCtx, M_CONTEXT, M_TRAIN_ENGINE, M_CPU);
    }
    MclassPreprocess(TrainCtx, M_DEFAULT);

}

void CMLDetCNN::TrainClassifier(MIL_UNIQUE_CLASS_ID& Dataset, 
    //MIL_UNIQUE_CLASS_ID& DatasetContext,
    MIL_UNIQUE_CLASS_ID& TrainCtx,
    MIL_UNIQUE_CLASS_ID& TrainedDetCtx,
    MIL_STRING& ClassifierDumpFile)
{
    MIL_INT MaxEpoch = 0;
    MIL_INT MinibatchSize = 0;
    MIL_INT TrainEngineUsed = 0;
    MIL_INT TrainDatasetNbImages = 0;
    MIL_INT DevDatasetNbImages = 0;
    MIL_DOUBLE LearningRate = 0.0;
    MIL_STRING TrainEngineDescription;
    MclassInquire(TrainCtx, M_DEFAULT, M_MAX_EPOCH + M_TYPE_MIL_INT, &MaxEpoch);
    MclassInquire(TrainCtx, M_DEFAULT, M_MINI_BATCH_SIZE + M_TYPE_MIL_INT, &MinibatchSize);
    MclassInquire(TrainCtx, M_CONTEXT, M_INITIAL_LEARNING_RATE + M_TYPE_MIL_DOUBLE, &LearningRate);
    MclassInquire(TrainCtx, M_CONTEXT, M_TRAIN_ENGINE_USED + M_TYPE_MIL_INT, &TrainEngineUsed);
    MclassInquire(TrainCtx, M_CONTEXT, M_TRAIN_ENGINE_USED_DESCRIPTION, TrainEngineDescription);

    MIL_UNIQUE_CLASS_ID TrainRes = MclassAllocResult(m_MilSystem, M_TRAIN_DET_RESULT, M_DEFAULT, M_UNIQUE_ID);
    DetHookDataStruct TheHookData;
    TheHookData.MilSystem = m_MilSystem;
    TheHookData.MilDisplay = m_MilDisplay;
    TheHookData.DumpTmpRst = 1;
    TheHookData.TrainModel = 1;
    TheHookData.ClassifierDumpFile = ClassifierDumpFile;


    TheHookData.DetDashboardPtr = DetDashboardPtr(new DetDashboard(
        m_MilSystem,
        TrainCtx,
        m_ImageSizeX,
        m_ImageSizeY,
        TrainEngineUsed,
        TrainEngineDescription));

    // Initialize the hook associated to the epoch trained event.
    MclassHookFunction(TrainCtx, M_EPOCH_TRAINED, DetHookFuncEpoch, &TheHookData);

    // Initialize the hook associated to the mini batch trained event.
    MclassHookFunction(TrainCtx, M_MINI_BATCH_TRAINED, DetHookFuncMiniBatch, &TheHookData);

    // Initialize the hook associated to the datasets prepared event.
    MclassHookFunction(TrainCtx, M_DATASETS_PREPARED, DetHookFuncDatasetsPrepared, &TheHookData);
    // Start the training process.
    double time = 0;
    MIL_TEXT_CHAR TheString[512];
    //timeStart();
    MclassTrain(TrainCtx, M_NULL, Dataset, M_NULL, TrainRes, M_DEFAULT);
    //timeEnd(time);
    if (TrainEngineUsed == M_CPU)

        MosPrintf(MIL_TEXT("Training is performed on the CPU used %f second\n"), time);
    else
        MosPrintf(MIL_TEXT("Training is performed on the GPU used %f second\n"), time);

    // Check the training status to ensure the training has completed properly.
    MIL_INT Status = -1;
    MclassGetResult(TrainRes, M_DEFAULT, M_STATUS + M_TYPE_MIL_INT, &Status);
    if (Status == M_COMPLETE)
    {
        TrainedDetCtx = MclassAlloc(m_MilSystem, M_CLASSIFIER_DET_PREDEFINED, M_DEFAULT, M_UNIQUE_ID);

        MclassCopyResult(TrainRes, M_DEFAULT, TrainedDetCtx, M_DEFAULT, M_TRAINED_CLASSIFIER, M_DEFAULT);

        MclassSave(ClassifierDumpFile, TrainedDetCtx, M_DEFAULT);
    }
}


int CMLDetCNN::TrainModel(DET_TRAIN_STRUCT DtParas) {
    //WriteLog(LOG_INFO, "Enter.");
    int RetVal = 0;
    MIL_UNIQUE_CLASS_ID TrainedDetCtx;
    MIL_UNIQUE_CLASS_ID TrainCtx = MclassAlloc(m_MilSystem, M_TRAIN_DET, M_DEFAULT, M_UNIQUE_ID);  

    MIL_STRING WorkDataDir = DtParas.WorkSpaceDir + L"//" + DtParas.DataSetName+ MIL_TEXT("//MIL_Data/");
    DtParas.TrainDstFolder = WorkDataDir + MIL_TEXT("//PreparedData/");
    MIL_STRING DetDumpFile = WorkDataDir + L"//" + DtParas.DataSetName+ L".mclass";
    MIL_STRING PreparedPath = WorkDataDir + MIL_TEXT("/WorkingDataset.mclassd");
    //判断WorkingDataset是否存在
     if (isfileNotExist(PreparedPath)) {
         //WriteLog(LOG_INFO, "%s is not exit.",PreparedPath.c_str());
         return -1;
     }

    ConstructTrainCtx(DtParas, TrainCtx);
    MIL_UNIQUE_CLASS_ID PreparedDataset = MclassRestore(PreparedPath, m_MilSystem, M_DEFAULT, M_UNIQUE_ID);
    TrainClassifier(PreparedDataset, TrainCtx, TrainedDetCtx, DetDumpFile);
    //WriteLog(LOG_INFO, "Exit.");
    return RetVal;

}

int CMLDetCNN::PredictFolderImgs(string SrcImgDir,
    MIL_STRING TdDetCtxPath,
    vector<DET_RESULT_STRUCT>& vecDetResults,
    bool SaveRst)
{
    predictPrepare(TdDetCtxPath);
    vector<MIL_STRING>FilesInFolder;
    m_AIParse->getFilesInFolder(SrcImgDir, "bmp", FilesInFolder);
    for (int i = 0; i < FilesInFolder.size(); i++) {
        DET_RESULT_STRUCT tmpRst;
        MIL_ID RawImage = MbufRestore(FilesInFolder[i], m_MilSystem, M_NULL);
        predict(RawImage, tmpRst);
        MbufFree(RawImage);
        vecDetResults.emplace_back(tmpRst);
    }
    if (SaveRst) {
        string strFilePath = "ODNetResult.txt";
        saveResult2File(strFilePath, FilesInFolder, vecDetResults);
    }
    return 0;
}

void CMLDetCNN::Predict(MIL_ID Image, MIL_UNIQUE_CLASS_ID& TdDetCtxPath, DET_RESULT_STRUCT& Result)
{
    //在线流程待确认 2023-9-22
    if (m_ModelNotPrePared) {
        predictPrepare(TdDetCtxPath);
    }
    MIL_ID ImageReduce;
    if (m_InputSizeBand == 3) {
        ImageReduce = MbufAllocColor(m_MilSystem, 3, m_InputSizeX, m_InputSizeY, 8 + M_UNSIGNED, M_IMAGE + M_PROC , M_NULL);
   }
    else {
        ImageReduce = MbufAlloc2d(m_MilSystem, m_InputSizeX, m_InputSizeY, 8 + M_UNSIGNED, M_IMAGE + M_PROC, M_NULL);
    }
    MimResize(Image, ImageReduce, M_FILL_DESTINATION, M_FILL_DESTINATION, M_BILINEAR);
    MIL_INT Status = M_FALSE;
    MclassInquire(TdDetCtxPath, M_DEFAULT, M_PREPROCESSED + M_TYPE_MIL_INT, &Status);
    if (M_FALSE == Status)
    {
        MclassPreprocess(TdDetCtxPath, M_DEFAULT);
    }
    MIL_UNIQUE_CLASS_ID ClassRes = MclassAllocResult(m_MilSystem, M_PREDICT_DET_RESULT, M_DEFAULT, M_UNIQUE_ID);
    
    LARGE_INTEGER t1, t2, tc;
    QueryPerformanceFrequency(&tc);
    QueryPerformanceCounter(&t1);
    MclassPredict(TdDetCtxPath, ImageReduce, ClassRes, M_DEFAULT);
    QueryPerformanceCounter(&t2);
    double time = (double)(t2.QuadPart - t1.QuadPart) / (double)tc.QuadPart ;
    //cout << "MclassPredict_time: " << time << endl;
    MbufFree(ImageReduce);
    MclassGetResult(ClassRes, M_GENERAL, M_NUMBER_OF_INSTANCES + M_TYPE_MIL_INT, &Result.InstanceNum);
    Result.Boxes.resize(Result.InstanceNum);
    Result.ClassIndex.resize(Result.InstanceNum);
    Result.Score.resize(Result.InstanceNum);
    Result.ClassName.resize(Result.InstanceNum);
    for (int i = 0; i < Result.InstanceNum; i++) {

        MclassGetResult(ClassRes, M_INSTANCE_INDEX(i), M_CENTER_X + M_TYPE_MIL_DOUBLE, &Result.Boxes[i].CX);
        MclassGetResult(ClassRes, M_INSTANCE_INDEX(i), M_CENTER_Y + M_TYPE_MIL_DOUBLE, &Result.Boxes[i].CY);
        MclassGetResult(ClassRes, M_INSTANCE_INDEX(i), M_HEIGHT + M_TYPE_MIL_DOUBLE, &Result.Boxes[i].H);
        MclassGetResult(ClassRes, M_INSTANCE_INDEX(i), M_WIDTH + M_TYPE_MIL_DOUBLE, &Result.Boxes[i].W);

        MIL_INT tmpClassIndex;
        MclassGetResult(ClassRes, M_INSTANCE_INDEX(i), M_BEST_CLASS_INDEX + M_TYPE_MIL_INT, &tmpClassIndex);
        Result.ClassIndex[i] = int(tmpClassIndex);

        
        MclassGetResult(ClassRes, M_INSTANCE_INDEX(i), M_BEST_CLASS_SCORE + M_TYPE_MIL_DOUBLE, &Result.Score[i]);
        MclassInquire(TdDetCtxPath, M_CLASS_INDEX(Result.ClassIndex[i]), M_CLASS_NAME, Result.ClassName[i]);
    }
}

void CMLDetCNN::ValModel_AP_50(string ValDataInfoPath, MIL_STRING TdDetCtxPath)
{
    //预测准备
    if (m_ModelNotPrePared) {
        predictPrepare(TdDetCtxPath);
    }
    //读取ValDataInfoPath 中的信息
    //step1:txt-->ImgDataInfo
    vector<MIL_STRING> vecImgPaths;
    vector<vector<Box>> vecGTBoxes;
    vector<vector<int>> vecGTlabels;
    m_AIParse->readDataSet2Vector(ValDataInfoPath, vecImgPaths, vecGTBoxes, vecGTlabels);

    //step3：对图像进行预测
    vector<vector<Box>> vecPdBoxes;
    vector<vector<int>> vecPdlabels;
    for (int i = 0; i < vecImgPaths.size(); i++) {
        DET_RESULT_STRUCT tmpRst;
        //vector<MIL_INT> ClassIndex;             //predict class
        //vector<MIL_DOUBLE> Score;               //predict score
        //vector<Box>Boxes;
        MIL_ID RawImage = MbufRestore(vecImgPaths[i], m_MilSystem, M_NULL);
        predict(RawImage, tmpRst);
        MbufFree(RawImage);
        vecPdBoxes.emplace_back(tmpRst.Boxes);
        vecPdlabels.emplace_back(tmpRst.ClassIndex);
    }
    
    //设置IOU_thes为定值0.5，计算AP50 
    //按照MIL中score_thres=0.5 ，不刻画P-R曲线：
    //计算AP
    vector<float> Precisions;
    vector<float> Recalls;
    calcAP4Vector(vecGTBoxes,
        vecGTlabels,
        vecPdBoxes,
        vecPdlabels,
        Precisions,
        Recalls);
    //for (int j = 0; j < vec2PdBoxes.size(); j++) {
    //    //计算单张图的TP/FP
    //    vector<Box>gt_box_label = vec2GTBoxes[j];
    //    vector<int> gt_labels = vecGTlabels[j];
    //    vector<Box>pd_box_label = vec2PdBoxes[j];
    //    vector<int> pd_labels = vecPdlabels[j];
    //    
    //}
    
}

void CMLDetCNN::PrintControls()
{
    MosPrintf(MIL_TEXT("Here are the dataset viewer controls:\n"));
    MosPrintf(MIL_TEXT("n: Display next image\n"));
    MosPrintf(MIL_TEXT("p: Display previous image\n"));
    MosPrintf(MIL_TEXT("t: Toggle between the GT overlay and the prediction overlay\n"));
    MosPrintf(MIL_TEXT("e: exit\n\n"));

    MosPrintf(MIL_TEXT("The possible colors in the overlay are:\n"));
    MosPrintf(MIL_TEXT("Green: Small knot\n"));
    MosPrintf(MIL_TEXT("Red: Large knot\n"));

    MosPrintf(MIL_TEXT("Select a dataset viewer control:\n"));
}

void CMLDetCNN::CDatasetViewer(MIL_ID Dataset)

{

    MIL_UNIQUE_DISP_ID MilDisplay = MdispAlloc(m_MilSystem, M_DEFAULT, MIL_TEXT("M_DEFAULT"), M_DEFAULT, M_UNIQUE_ID);
    MIL_INT ImageSizeX = m_InputSizeX;
    MIL_INT ImageSizeY = m_InputSizeY;
    BOOL m_DisplayGroundTruth = false;
    int NUMBER_OF_CLASSES = m_ClassesNum;
    PrintControls();

    //MIL_UNIQUE_DISP_ID MilDisplay = MdispAlloc(m_MilSystem, M_DEFAULT, MIL_TEXT("M_DEFAULT"), M_DEFAULT, M_UNIQUE_ID);
    //MIL_INT ImageSizeX = 0;
    //MIL_INT ImageSizeY = 0;
    //GetImageSizes(m_MilSystem, m_Dataset, &ImageSizeX, &ImageSizeY);

    const MIL_INT IconSize = ImageSizeY / NUMBER_OF_CLASSES;
    MIL_UNIQUE_BUF_ID DispImage = MbufAllocColor(m_MilSystem, 3, ImageSizeX + IconSize, ImageSizeY, 8 + M_UNSIGNED, M_IMAGE + M_PROC + M_DISP, M_UNIQUE_ID);
    MIL_UNIQUE_BUF_ID DispChild = MbufChild2d(DispImage, 0, 0, ImageSizeX, ImageSizeY, M_UNIQUE_ID);

    MdispSelect(MilDisplay, DispImage);
    MIL_ID MilOverlay = MdispInquire(MilDisplay, M_OVERLAY_ID, M_NULL);
    MIL_UNIQUE_BUF_ID OverlayChild = MbufChild2d(MilOverlay, 0, 0, ImageSizeX, ImageSizeY, M_UNIQUE_ID);

    MbufClear(DispImage, M_COLOR_BLACK);

    // For bounding boxes.
    MIL_UNIQUE_GRA_ID GraList = MgraAllocList(m_MilSystem, M_DEFAULT, M_UNIQUE_ID);
    MdispControl(MilDisplay, M_ASSOCIATED_GRAPHIC_LIST_ID, GraList);

    // Set annotation color.
    MgraColor(M_DEFAULT, M_COLOR_RED);

    // Set up the display.
    for (MIL_INT Iter = 0; Iter < NUMBER_OF_CLASSES; Iter++)
    {
        // Allocate a child buffer per product category.
        MIL_UNIQUE_BUF_ID MilChildSample = MbufChild2d(DispImage, ImageSizeX, Iter * IconSize, IconSize, IconSize, M_UNIQUE_ID);
        MIL_UNIQUE_BUF_ID MilOverlayChildSample = MbufChild2d(MilOverlay, ImageSizeX, Iter * IconSize, IconSize, IconSize, M_UNIQUE_ID);
        MbufClear(MilChildSample, M_COLOR_BLACK);
        MbufClear(MilOverlayChildSample, M_COLOR_BLACK);

        // Load the sample image.
        MIL_ID ClassIconId = MclassInquire(Dataset, M_CLASS_INDEX(Iter), M_CLASS_ICON_ID + M_TYPE_MIL_ID, M_NULL);

        // Retrieve the class description.
        MIL_STRING Text;
        MclassInquire(Dataset, M_CLASS_INDEX(Iter), M_CLASS_NAME, Text);

        if (ClassIconId != M_NULL)
        {
            // Retrieve the color associated to the class.
            MIL_DOUBLE ClassColor;
            MclassInquire(Dataset, M_CLASS_INDEX(Iter), M_CLASS_DRAW_COLOR, &ClassColor);

            // Draw the class name using the color associated to the class.
            MgraColor(M_DEFAULT, ClassColor);
            MgraText(M_DEFAULT, MilChildSample, 10, 10, Text);
            MgraText(M_DEFAULT, MilOverlayChildSample, 10, 10, Text);

            const MIL_INT ClassImageExampleSizeX = MbufInquire(ClassIconId, M_SIZE_X, M_NULL);
            const MIL_INT ClassImageExampleSizeY = MbufInquire(ClassIconId, M_SIZE_Y, M_NULL);

            if ((ClassImageExampleSizeX >= IconSize) || (ClassImageExampleSizeY >= IconSize))
            {
                MimResize(ClassIconId, MilChildSample, M_FILL_DESTINATION, M_FILL_DESTINATION, M_AVERAGE);
                MimResize(ClassIconId, MilOverlayChildSample, M_FILL_DESTINATION, M_FILL_DESTINATION, M_AVERAGE);
            }
            else
            {
                const MIL_INT OffsetX = (IconSize - ClassImageExampleSizeX) / 2;
                const MIL_INT OffsetY = (IconSize - ClassImageExampleSizeY) / 2;
                MbufCopyColor2d(ClassIconId, MilChildSample, M_ALL_BANDS, 0, 0, M_ALL_BANDS, OffsetX, OffsetY, ClassImageExampleSizeX, ClassImageExampleSizeY);
                MbufCopyColor2d(ClassIconId, MilOverlayChildSample, M_ALL_BANDS, 0, 0, M_ALL_BANDS, OffsetX, OffsetY, ClassImageExampleSizeX, ClassImageExampleSizeY);
            }
        }

        // Draw an initial red rectangle around the buffer.
        MgraRect(M_DEFAULT, MilChildSample, 0, 1, IconSize - 1, IconSize - 2);
        MgraRect(M_DEFAULT, MilOverlayChildSample, 0, 1, IconSize - 1, IconSize - 2);
    }

    MIL_UNIQUE_GRA_ID GraContext = MgraAlloc(m_MilSystem, M_UNIQUE_ID);

    MIL_INT NbEntries = 0;
    MclassInquire(m_Dataset, M_DEFAULT, M_NUMBER_OF_ENTRIES + M_TYPE_MIL_INT, &NbEntries);

    MIL_TEXT_CHAR IndexText[512];
    MIL_TEXT_CHAR OverlayText[512];
    MIL_INT EntryIndex = 0;
    bool Exit = false;
    while (!Exit)
    {
        MdispControl(MilDisplay, M_UPDATE, M_DISABLE);

        MIL_STRING EntryImagePath;
        MclassInquireEntry(Dataset, EntryIndex, M_DEFAULT_KEY, M_DEFAULT, M_ENTRY_IMAGE_PATH_ABS, EntryImagePath);
        MbufLoad(EntryImagePath, DispChild);

        MdispControl(MilDisplay, M_OVERLAY_OPACITY, 0.0);

        MgraClear(M_DEFAULT, GraList);
        MbufClear(OverlayChild, 0.0);

        // Draw the desired overlay
        if (m_DisplayGroundTruth)
        {
            /*  const MIL_INT NumGTs = GetNumberOfGTs(Dataset, EntryIndex);
              MclassDrawEntry(M_DEFAULT, Dataset, GraList, M_DESCRIPTOR_TYPE_BOX + M_PSEUDO_COLOR, EntryIndex, M_DEFAULT_KEY, M_DETECTION, M_DEFAULT, M_NULL, M_DEFAULT);
              MosSprintf(OverlayText, 512, MIL_TEXT("Ground truth overlay, there are %d GTs"), NumGTs);*/
        }
        else
        {
            MIL_INT PredictInfo{ M_FALSE };
            MclassGetResultEntry(Dataset, EntryIndex, M_DEFAULT_KEY, M_DETECTION, M_DEFAULT, M_PREDICT_INFO + M_TYPE_MIL_INT, &PredictInfo);
            if (PredictInfo == M_TRUE)
            {
                MIL_INT NumInstances{ 0 };
                MclassGetResultEntry(Dataset, EntryIndex, M_DEFAULT_KEY, M_DETECTION, M_DEFAULT, M_NUMBER_OF_INSTANCES + M_TYPE_MIL_INT, &NumInstances);

                MclassDrawEntry(GraContext, Dataset, GraList, M_DRAW_BOX + M_DRAW_BOX_NAME + M_DRAW_BOX_SCORE, EntryIndex, M_DEFAULT_KEY, M_DETECTION, M_DEFAULT, M_NULL, M_DEFAULT);
                MosSprintf(OverlayText, 512, MIL_TEXT("%d instance(s) found"), NumInstances);
            }
            else
            {
                MosSprintf(OverlayText, 512, MIL_TEXT("No prediction to display"));
            }
        }
        MIL_INT TextYPos = Y_MARGIN;

        MosSprintf(IndexText, 512, MIL_TEXT("Entry Index %d / %d"), EntryIndex, NbEntries - 1);
        MgraText(GraContext, DispChild, TEXT_MARGIN, TextYPos, IndexText);
        MgraText(GraContext, OverlayChild, TEXT_MARGIN, TextYPos, IndexText);
        TextYPos += TEXT_HEIGHT;

        MgraText(GraContext, DispChild, TEXT_MARGIN, TextYPos, OverlayText);
        MgraText(GraContext, OverlayChild, TEXT_MARGIN, TextYPos, OverlayText);

        MdispControl(MilDisplay, M_UPDATE, M_ENABLE);
    }

}