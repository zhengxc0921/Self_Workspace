#include"MLClassCNN.h"


//MIL_INT MFTYPE HookFuncDatasetsPrepared(
//    MIL_INT /*HookType*/,
//    MIL_ID  EventId,
//    void* UserData)
//{
//    auto HookData = (HookDataCNNStruct*)UserData;
//
//    MIL_ID TrainResult;
//    MclassGetHookInfo(EventId, M_RESULT_ID + M_TYPE_MIL_ID, &TrainResult);
//
//    MIL_UNIQUE_CLASS_ID TrainPreparedDataset = MclassAlloc(HookData->MilSystem, M_DATASET_IMAGES, M_DEFAULT, M_UNIQUE_ID);
//    MclassCopyResult(TrainResult, M_DEFAULT, TrainPreparedDataset, M_DEFAULT, M_PREPARED_TRAIN_DATASET, M_DEFAULT);
//    const MIL_INT TrainDatasetNbImages = MclassInquire(TrainPreparedDataset, M_DEFAULT, M_NUMBER_OF_ENTRIES, M_NULL);
//
//    MIL_UNIQUE_CLASS_ID DevPreparedDataset = MclassAlloc(HookData->MilSystem, M_DATASET_IMAGES, M_DEFAULT, M_UNIQUE_ID);
//    MclassCopyResult(TrainResult, M_DEFAULT, DevPreparedDataset, M_DEFAULT, M_PREPARED_DEV_DATASET, M_DEFAULT);
//    const MIL_INT DevDatasetNbImages = MclassInquire(DevPreparedDataset, M_DEFAULT, M_NUMBER_OF_ENTRIES, M_NULL);
//
//    HookData->DashboardPtr->AddDatasetsPreparedData(TrainDatasetNbImages, DevDatasetNbImages);
//
//    MdispSelect(HookData->MilDisplay, HookData->DashboardPtr->GetDashboardBufId());
//
//    return M_NULL;
//}
//
////.........................................................................
//MIL_INT MFTYPE HookFuncEpoch(
//    MIL_INT /*HookType*/,
//    MIL_ID  EventId,
//    void* UserData)
//{
//    auto HookData = (HookDataCNNStruct*)UserData;
//
//    MIL_DOUBLE CurBench = 0.0;
//    MIL_DOUBLE CurBenchMean = -1.0;
//
//    MIL_INT CurEpochIndex = 0;
//    MclassGetHookInfo(EventId, M_EPOCH_INDEX + M_TYPE_MIL_INT, &CurEpochIndex);
//
//    MappTimer(M_DEFAULT, M_TIMER_READ, &CurBench);
//    MIL_DOUBLE EpochBenchMean = CurBench / (CurEpochIndex + 1);
//
//    MIL_DOUBLE TrainErrorRate = 0;
//    MclassGetHookInfo(EventId, M_TRAIN_DATASET_ERROR_RATE, &TrainErrorRate);
//    MIL_DOUBLE DevErrorRate = 0;
//    MclassGetHookInfo(EventId, M_DEV_DATASET_ERROR_RATE, &DevErrorRate);
//
//    MIL_INT AreTrainedCNNParametersUpdated = M_FALSE;
//    MclassGetHookInfo(EventId,
//        M_TRAINED_PARAMETERS_UPDATED + M_TYPE_MIL_INT,
//        &AreTrainedCNNParametersUpdated);
//
//    // By default trained parameters are updated when the dev error rate
//    // is the best up to now.
//    bool TheEpochIsTheBestUpToNow = (AreTrainedCNNParametersUpdated == M_TRUE);
//
//    HookData->DashboardPtr->AddEpochData(
//        TrainErrorRate,
//        DevErrorRate,
//        CurEpochIndex,
//        TheEpochIsTheBestUpToNow,
//        EpochBenchMean);
//
//    return M_NULL;
//}
//
////............................................................................
//MIL_INT MFTYPE HookFuncMiniBatch(
//    MIL_INT HookType,
//    MIL_ID  EventId,
//    void* UserData)
//{
//    auto HookData = (HookDataCNNStruct*)UserData;
//
//    MIL_DOUBLE LossError = 0;
//    MclassGetHookInfo(EventId, M_MINI_BATCH_LOSS, &LossError);
//
//    MIL_INT MiniBatchIdx = 0;
//    MclassGetHookInfo(EventId, M_MINI_BATCH_INDEX + M_TYPE_MIL_INT, &MiniBatchIdx);
//
//    MIL_INT EpochIdx = 0;
//    MclassGetHookInfo(EventId, M_EPOCH_INDEX + M_TYPE_MIL_INT, &EpochIdx);
//
//    MIL_INT NbMiniBatchPerEpoch = 0;
//    MclassGetHookInfo(EventId, M_MINI_BATCH_PER_EPOCH + M_TYPE_MIL_INT, &NbMiniBatchPerEpoch);
//
//    if (EpochIdx == 0 && MiniBatchIdx == 0)
//    {
//        MappTimer(M_DEFAULT, M_TIMER_RESET, M_NULL);
//    }
//
//    if (MiniBatchIdx == NbMiniBatchPerEpoch - 1 && HookData->DumpTmpRst == 1)
//    {
//        MIL_ID TrainRes;
//        MclassGetHookInfo(EventId, M_RESULT_ID + M_TYPE_MIL_ID, &TrainRes);
//        MIL_UNIQUE_CLASS_ID TrainedClassifierCtx = MclassAlloc(HookData->MilSystem, M_CLASSIFIER_CNN_PREDEFINED, M_DEFAULT, M_UNIQUE_ID);
//        MclassCopyResult(TrainRes, M_DEFAULT, TrainedClassifierCtx, M_DEFAULT, M_TRAINED_CLASSIFIER, M_DEFAULT);
//        MclassSave(HookData->ClassifierDumpFile, TrainedClassifierCtx, M_DEFAULT);
//    }
//
//    HookData->DashboardPtr->AddMiniBatchData(LossError, MiniBatchIdx, EpochIdx, NbMiniBatchPerEpoch);
//
//    //MosPrintf(MIL_TEXT("\nBatch[%d] loss error = %f.\n"), MiniBatchIdx, LossError);
//
//    //EpochIdx > 1的时候再来判断
//
//    if (HookData->ControlType == 's') {
//        //在Stop后先保存当前模型
//        MIL_ID TrainRes;
//        MclassGetHookInfo(EventId, M_RESULT_ID + M_TYPE_MIL_ID, &TrainRes);
//        MIL_UNIQUE_CLASS_ID TrainedClassifierCtx = MclassAlloc(HookData->MilSystem, M_CLASSIFIER_CNN_PREDEFINED, M_DEFAULT, M_UNIQUE_ID);
//        MclassCopyResult(TrainRes, M_DEFAULT, TrainedClassifierCtx, M_DEFAULT, M_TRAINED_CLASSIFIER, M_DEFAULT);
//        //查询当前模型
//        //MIL_INT Status = -1;
//        //MclassGetResult(TrainRes, M_DEFAULT, M_STATUS + M_TYPE_MIL_INT, &Status);
//        //if (Status == M_FALSE) {
//        //    printf("FALSE");
//        //}
//        //if (Status == M_COMPLETE) {
//        //    printf("M_COMPLETE");
//        //}
//        //MclassPreprocess(TrainedClassifierCtx, M_DEFAULT);
//        MclassSave(HookData->ClassifierDumpFile, TrainedClassifierCtx, M_DEFAULT);
//
//        MIL_ID HookInfoTrainResId = M_NULL;
//        MclassGetHookInfo(EventId, M_RESULT_ID + M_TYPE_MIL_ID, &HookInfoTrainResId);
//        MclassControl(HookInfoTrainResId, M_DEFAULT, M_STOP_TRAIN, M_DEFAULT);
//        MosPrintf(MIL_TEXT("The training has been stopped.\n"));
//    }
//    else if (HookData->ControlType == 'p') {
//        MosPrintf(MIL_TEXT("\nPress 's' to stop the training or any other key to continue.\n"));
//        while (1) {
//            //Sleep(1000);
//            if (HookData->ControlType == 's')
//            {
//                //在Stop后先保存当前模型
//                MIL_ID TrainRes;
//                MclassGetHookInfo(EventId, M_RESULT_ID + M_TYPE_MIL_ID, &TrainRes);
//                MIL_UNIQUE_CLASS_ID TrainedClassifierCtx = MclassAlloc(HookData->MilSystem, M_CLASSIFIER_CNN_PREDEFINED, M_DEFAULT, M_UNIQUE_ID);
//                MclassCopyResult(TrainRes, M_DEFAULT, TrainedClassifierCtx, M_DEFAULT, M_TRAINED_CLASSIFIER, M_DEFAULT);
//                MclassSave(HookData->ClassifierDumpFile, TrainedClassifierCtx, M_DEFAULT);
//
//
//                MIL_ID HookInfoTrainResId = M_NULL;
//                MclassGetHookInfo(EventId, M_RESULT_ID + M_TYPE_MIL_ID, &HookInfoTrainResId);
//                MclassControl(HookInfoTrainResId, M_DEFAULT, M_STOP_TRAIN, M_DEFAULT);
//                MosPrintf(MIL_TEXT("The training has been stopped.\n"));
//                break;
//            }
//            else if (HookData->ControlType == 'r')
//            {
//                MosPrintf(MIL_TEXT("The training will continue.\n"));
//                break;
//            }
//
//        }
//
//    }
//    return M_NULL;
//}



CMLClassCNN::CMLClassCNN(MIL_ID MilSystem, MIL_ID MilDisplay) :
	m_MilSystem(MilSystem),
	m_MilDisplay(MilDisplay)
{

    m_AIParse = CAIParsePtr(new CAIParse(MilSystem));
}

MIL_INT CMLClassCNN::CnnTrainEngineDLLInstalled(MIL_ID MilSystem)
{

    MIL_INT IsInstalled = M_FALSE;
    MIL_UNIQUE_CLASS_ID TrainCtx = MclassAlloc(MilSystem, M_TRAIN_CNN, M_DEFAULT, M_UNIQUE_ID);
    MclassInquire(TrainCtx, M_DEFAULT, M_TRAIN_ENGINE_IS_INSTALLED + M_TYPE_MIL_INT, &IsInstalled);
    return IsInstalled;

}

bool CMLClassCNN::isTagSameClass(
    vector<MIL_STRING>BaseClsNames,
    vector<MIL_STRING> TagClsNames) {
    bool isSameCls=TRUE;
    for (int i = 0; i < TagClsNames.size(); i++) {
        if (std::find(BaseClsNames.begin(), BaseClsNames.end(), TagClsNames[i]) == BaseClsNames.end())
        {
            isSameCls = FALSE;
        }
    }
    return isSameCls;
}


void CMLClassCNN::InitClassWeights()
{
    
    m_ClassWeights.resize(m_ClassesNum, 1.0);
}

void CMLClassCNN::ConstructDataset(
    vector<MIL_STRING> ClassName, 
    vector<MIL_STRING> ClassIcon, 
    MIL_STRING AuthorName, 
    MIL_STRING OriginalDataPath, 
    const MIL_STRING& WorkingDataPath,
    MIL_UNIQUE_CLASS_ID& Dataset

)
{
    MIL_INT  NumberOfClasses = ClassName.size();

    if (M_NULL == Dataset)
    {
        Dataset = MclassAlloc(m_MilSystem, M_DATASET_IMAGES, M_DEFAULT, M_UNIQUE_ID);
    }
    AddClassDescription(Dataset, AuthorName, ClassName, ClassIcon, NumberOfClasses);

    for (MIL_INT ClassIdx = 0; ClassIdx < NumberOfClasses; ClassIdx++)
    {
        AddClassToDataset(ClassIdx, OriginalDataPath, ClassName[ClassIdx],
            AuthorName, Dataset);
    }

}

void CMLClassCNN::ConstructMergeDataset(
    MIL_STRING AuthorName,
    MIL_STRING BaseDataDir, 
    MIL_STRING TagDataDir,
    vector<MIL_DOUBLE> vecSampleRatio,
    MIL_UNIQUE_CLASS_ID& BaseDataSet,
    MIL_UNIQUE_CLASS_ID& TagDataSet)
{

    //提取BaseData 中的类型和图片：1、部分提取；2、所有提取
    //step1:查询Base、Tag的ClsNames
    vector<MIL_STRING>BaseClsNames,TagClsNames;
    MIL_STRING BaseImgDir = BaseDataDir + MIL_TEXT("Images\\");
    MIL_STRING BaseIconDir = BaseDataDir + MIL_TEXT("Icons\\");
    string strBaseImgDir, strTagDataDir;
    m_AIParse->MIL_STRING2string(BaseImgDir, strBaseImgDir);
    m_AIParse->MIL_STRING2string(TagDataDir, strTagDataDir);
    m_AIParse->getFoldersInFolder(strBaseImgDir, BaseClsNames);
    m_AIParse->getFoldersInFolder(strTagDataDir, TagClsNames);
    bool bTagSameClass = isTagSameClass(BaseClsNames,TagClsNames);

    //遍历BaseImg所有类型的图片，1、部分提取；2、所有提取
    map<MIL_STRING, vector<MIL_STRING>>PartBaseData, BaseData;
    for (int i = 0; i < BaseClsNames.size(); i++) {
        MIL_STRING FolderDir = BaseImgDir + BaseClsNames[i];
        string strFolderDir;
        m_AIParse->MIL_STRING2string(FolderDir, strFolderDir);

        vector<MIL_STRING>BaseClsImg;
        m_AIParse->getFilesInFolder(strFolderDir, "bmp", BaseClsImg);

        int nFileNum = ceil(BaseClsImg.size() * vecSampleRatio[i]);
        random_shuffle(BaseClsImg.begin(), BaseClsImg.end());
     
        vector<MIL_STRING> PartClsImg(BaseClsImg.begin(), BaseClsImg.begin() + nFileNum);
        PartBaseData.insert(pair<MIL_STRING, vector<MIL_STRING>>(BaseClsNames[i], PartClsImg));
        BaseData.insert(pair<MIL_STRING, vector<MIL_STRING>>(BaseClsNames[i], BaseClsImg));
    }

    //遍历TagImg所有类型的图片,获取TagData
    map<MIL_STRING, vector<MIL_STRING>>TagData;
    for (int i = 0; i < TagClsNames.size(); i++) {
        MIL_STRING TagFolderDir = TagDataDir + TagClsNames[i];
        string strTagFolderDir;
        m_AIParse->MIL_STRING2string(TagFolderDir, strTagFolderDir);
        vector<MIL_STRING>TagClsImg;
        m_AIParse->getFilesInFolder(strTagFolderDir, "bmp", TagClsImg);
        TagData.insert(pair<MIL_STRING, vector<MIL_STRING>>(BaseClsNames[i], TagClsImg));
    }
    //生成TagDataSet
    //MIL_UNIQUE_CLASS_ID TagDataSet = MclassAlloc(m_MilSystem, M_DATASET_IMAGES, M_DEFAULT, M_UNIQUE_ID);
    MclassControl(TagDataSet, M_DEFAULT, M_AUTHOR_ADD, AuthorName);
    for (MIL_INT i = 0; i < TagClsNames.size(); i++)
    {
        //加入Icon
        MclassControl(TagDataSet, M_DEFAULT, M_CLASS_ADD, TagClsNames[i]);
        MIL_STRING ClassIcon = TagDataDir + TagClsNames[i] + L".mim";
        MIL_UNIQUE_BUF_ID IconImageId = MbufRestore(ClassIcon, m_MilSystem, M_UNIQUE_ID);
        MclassControl(TagDataSet, M_CLASS_INDEX(i), M_CLASS_ICON_ID, IconImageId);
        //加入Img
        MIL_INT ClassIndex = i;
        MIL_INT NbEntries;
        MclassInquire(TagDataSet, M_DEFAULT, M_NUMBER_OF_ENTRIES + M_TYPE_MIL_INT, &NbEntries);
        vector<MIL_STRING> TagDataImgs = TagData[TagClsNames[i]];
        for (int j = 0; j < TagDataImgs.size();j++) {
        MclassControl(TagDataSet, M_DEFAULT, M_ENTRY_ADD, M_DEFAULT);
        MclassControlEntry(TagDataSet, NbEntries + j, M_DEFAULT_KEY, M_REGION_INDEX(0), M_CLASS_INDEX_GROUND_TRUTH, ClassIndex, M_NULL, M_DEFAULT);
        MclassControlEntry(TagDataSet, NbEntries + j, M_DEFAULT_KEY, M_DEFAULT, M_ENTRY_IMAGE_PATH, M_DEFAULT, TagDataImgs[j], M_DEFAULT);
        MclassControlEntry(TagDataSet, NbEntries + j, M_DEFAULT_KEY, M_REGION_INDEX(0), M_AUTHOR_NAME, M_DEFAULT, AuthorName, M_DEFAULT);
        }
    }
    MclassControl(TagDataSet, M_CONTEXT, M_CONSOLIDATE_ENTRIES_INTO_FOLDER, BaseDataDir);
    //MIL_STRING TagDataSetPath = BaseDataDir + MIL_TEXT("TagDataSet.mclassd");
    //MclassSave(TagDataSetPath, TagDataSet, M_DEFAULT);
    if (bTagSameClass)
    {
        //生成UpdateDataSet = PartBaseDataset+TagDataset
        for (MIL_INT i = 0; i < BaseClsNames.size(); i++)
    {
        //加入Img
        MIL_INT ClassIndex = i;
        MIL_INT NbEntries;
        MclassInquire(TagDataSet, M_DEFAULT, M_NUMBER_OF_ENTRIES + M_TYPE_MIL_INT, &NbEntries);
        vector<MIL_STRING> PartBaseImgData = PartBaseData[BaseClsNames[i]];
        for (int j = 0; j < PartBaseImgData.size(); j++) {
            MclassControl(TagDataSet, M_DEFAULT, M_ENTRY_ADD, M_DEFAULT);
            MclassControlEntry(TagDataSet, NbEntries + j, M_DEFAULT_KEY, M_REGION_INDEX(0), M_CLASS_INDEX_GROUND_TRUTH, ClassIndex, M_NULL, M_DEFAULT);
            MclassControlEntry(TagDataSet, NbEntries + j, M_DEFAULT_KEY, M_DEFAULT, M_ENTRY_IMAGE_PATH, M_DEFAULT, PartBaseImgData[j], M_DEFAULT);
            MclassControlEntry(TagDataSet, NbEntries + j, M_DEFAULT_KEY, M_REGION_INDEX(0), M_AUTHOR_NAME, M_DEFAULT, AuthorName, M_DEFAULT);
        }
    }
        MIL_STRING UpdateDataSetPath = BaseDataDir + MIL_TEXT("UpdateDataSet.mclassd");
        MclassSave(UpdateDataSetPath, TagDataSet, M_DEFAULT);
       //生成BaseDataSet = AllData
        vector<MIL_STRING>NBaseClsNames;
        m_AIParse->getFoldersInFolder(strBaseImgDir, NBaseClsNames);
        //MIL_UNIQUE_CLASS_ID BaseDataSet = MclassAlloc(m_MilSystem, M_DATASET_IMAGES, M_DEFAULT, M_UNIQUE_ID);
        for (MIL_INT i = 0; i < NBaseClsNames.size(); i++)
        {
            //加入Icon
            MclassControl(BaseDataSet, M_DEFAULT, M_CLASS_ADD, NBaseClsNames[i]);
            MIL_STRING ClassIcon = BaseIconDir + NBaseClsNames[i] + L".mim";
            MIL_UNIQUE_BUF_ID IconImageId = MbufRestore(ClassIcon, m_MilSystem, M_UNIQUE_ID);
            MclassControl(BaseDataSet, M_CLASS_INDEX(i), M_CLASS_ICON_ID, IconImageId);
            //加入Img
            MIL_STRING BaseFolderDir = BaseImgDir + NBaseClsNames[i];
            string strFolderDir;
            m_AIParse->MIL_STRING2string(BaseFolderDir, strFolderDir);
            vector<MIL_STRING>BaseClsImg;
            m_AIParse->getFilesInFolder(strFolderDir, "bmp", BaseClsImg);
            MIL_INT ClassIndex = i;
            MIL_INT NbEntries;
            MclassInquire(BaseDataSet, M_DEFAULT, M_NUMBER_OF_ENTRIES + M_TYPE_MIL_INT, &NbEntries);
            for (int j = 0; j < BaseClsImg.size(); j++) {
                MIL_INT Cd = i;
                MclassControl(BaseDataSet, M_DEFAULT, M_ENTRY_ADD, M_DEFAULT);
                MclassControlEntry(BaseDataSet, NbEntries + j, M_DEFAULT_KEY, M_REGION_INDEX(0), M_CLASS_INDEX_GROUND_TRUTH, i, M_NULL, M_DEFAULT);
                MclassControlEntry(BaseDataSet, NbEntries + j, M_DEFAULT_KEY, M_DEFAULT, M_ENTRY_IMAGE_PATH, M_DEFAULT, BaseClsImg[j], M_DEFAULT);
                //MclassControlEntry(BaseDataSet, NbEntries + j, M_DEFAULT_KEY, M_REGION_INDEX(0), M_AUTHOR_NAME, M_DEFAULT, L"A", M_DEFAULT);
            }
        }
        MIL_STRING BaseDataSetPath = BaseDataDir + MIL_TEXT("BaseDataSet.mclassd");
        MclassSave(BaseDataSetPath, BaseDataSet, M_DEFAULT);
    }
    else {
        //生成 BaseDataSet = AllData
        vector<MIL_STRING>NBaseClsNames;
        m_AIParse->getFoldersInFolder(strBaseImgDir, NBaseClsNames);
        //MIL_UNIQUE_CLASS_ID BaseDataSet = MclassAlloc(m_MilSystem, M_DATASET_IMAGES, M_DEFAULT, M_UNIQUE_ID);
        for (MIL_INT i = 0; i < NBaseClsNames.size(); i++)
        {
            //加入Icon
            MclassControl(BaseDataSet, M_DEFAULT, M_CLASS_ADD, NBaseClsNames[i]);
            MIL_STRING ClassIcon = BaseIconDir + NBaseClsNames[i] + L".mim";
            MIL_UNIQUE_BUF_ID IconImageId = MbufRestore(ClassIcon, m_MilSystem, M_UNIQUE_ID);
            MclassControl(BaseDataSet, M_CLASS_INDEX(i), M_CLASS_ICON_ID, IconImageId);
            //加入Img
            MIL_STRING BaseFolderDir = BaseImgDir + NBaseClsNames[i];
            string strFolderDir;
            m_AIParse->MIL_STRING2string(BaseFolderDir, strFolderDir);
            vector<MIL_STRING>BaseClsImg;
            m_AIParse->getFilesInFolder(strFolderDir, "bmp", BaseClsImg);
            MIL_INT ClassIndex = i;
            MIL_INT NbEntries;
            MclassInquire(BaseDataSet, M_DEFAULT, M_NUMBER_OF_ENTRIES + M_TYPE_MIL_INT, &NbEntries);
            for (int j = 0; j < BaseClsImg.size(); j++) {
                MIL_INT Cd = i;
                MclassControl(BaseDataSet, M_DEFAULT, M_ENTRY_ADD, M_DEFAULT);
                MclassControlEntry(BaseDataSet, NbEntries + j, M_DEFAULT_KEY, M_REGION_INDEX(0), M_CLASS_INDEX_GROUND_TRUTH, i, M_NULL, M_DEFAULT);
                MclassControlEntry(BaseDataSet, NbEntries + j, M_DEFAULT_KEY, M_DEFAULT, M_ENTRY_IMAGE_PATH, M_DEFAULT, BaseClsImg[j], M_DEFAULT);
                //MclassControlEntry(BaseDataSet, NbEntries + j, M_DEFAULT_KEY, M_REGION_INDEX(0), M_AUTHOR_NAME, M_DEFAULT, L"A", M_DEFAULT);
            }
        }
        MIL_STRING BaseDataSetPath = BaseDataDir + MIL_TEXT("BaseDataSet.mclassd");
        MclassSave(BaseDataSetPath, BaseDataSet, M_DEFAULT);
        MIL_STRING UpdateDataSetPath = BaseDataDir + MIL_TEXT("UpdateDataSet.mclassd");
        MclassSave(UpdateDataSetPath, TagDataSet, M_DEFAULT);
    }
}

void CMLClassCNN::MergeTagData2BaseSet(
    MIL_STRING BaseDataDir,
    MIL_STRING DataSetType,
    vector<MIL_STRING> BaseClsNames,
    MIL_STRING TagDataDir,
    vector<MIL_DOUBLE>vecSampleRatio,
    MIL_UNIQUE_CLASS_ID& MergeSet,
    MIL_UNIQUE_CLASS_ID& BaseSet)
{  
    MergeSet = MclassAlloc(m_MilSystem, M_DATASET_IMAGES, M_DEFAULT, M_UNIQUE_ID);
    MIL_STRING BaseSetPreName = L"COMPSet";
    MIL_STRING MergeSetPreName = L"SimplSet";

    MIL_INT NbEntries = 0;
    MIL_STRING BaseSetPath = BaseDataDir + BaseSetPreName+ DataSetType+L".mclassd";
    //BaseSet = MclassRestore(BaseSetPath, m_MilSystem, M_DEFAULT, M_UNIQUE_ID);
    //MclassInquire(BaseSet, M_DEFAULT, M_NUMBER_OF_ENTRIES + M_TYPE_MIL_INT, &NbEntries);
    
    //获取已有vecBaseData内容
    vector<vector<MIL_STRING>> vecBaseData, vecTagData, vecTagBaseData, vecPBaseData;

    m_ClassesNum = BaseClsNames.size();
    vecBaseData.resize(m_ClassesNum);
    vecTagData.resize(m_ClassesNum);
    vecTagBaseData.resize(m_ClassesNum);
    vecPBaseData.resize(m_ClassesNum);
    GetVecData4Set(BaseSetPath, vecBaseData,BaseSet);
    //for (int i = 0; i < NbEntries; i++) {
    //    MIL_STRING P;
    //    std::vector<MIL_INT> GTIdx;
    //    MclassInquireEntry(BaseSet, i, M_DEFAULT_KEY, M_DEFAULT, M_ENTRY_IMAGE_PATH_ABS, P);
    //    MclassInquireEntry(BaseSet, i, M_DEFAULT_KEY, M_REGION_INDEX(0), M_CLASS_INDEX_GROUND_TRUTH, GTIdx);
    //    vecBaseData[GTIdx[0]].emplace_back(P);
    //}
    for (int i = 0; i < m_ClassesNum; i++)
    {
        MIL_STRING ClassIconPrefix = TagDataDir + DataSetType +L"//" + BaseClsNames[i];
        int nFileNum = ceil(vecBaseData[i].size() * vecSampleRatio[i]);
        random_shuffle(vecBaseData[i].begin(), vecBaseData[i].end());
        vecPBaseData[i].assign(vecBaseData[i].begin(), vecBaseData[i].begin()+ nFileNum);
        vector<MIL_STRING>vecTagImg;
        m_AIParse->getFilesInFolder(ClassIconPrefix,"bmp", vecTagImg);
        vecTagData[i].assign(vecTagImg.begin(), vecTagImg.end());
    }
    GenralMergeSet(BaseDataDir,TagDataDir,BaseClsNames,vecTagData,MergeSet);
    MergeVecData2Set(MergeSet,vecPBaseData);
    MergeVecData2Set(BaseSet, vecTagData);
    BaseSetPath = BaseDataDir + BaseSetPreName + DataSetType +L".mclassd";
    MclassSave(BaseSetPath, BaseSet, M_DEFAULT);
    MIL_STRING SpBaseSetPath = BaseDataDir + MergeSetPreName + DataSetType + L".mclassd";
    MclassSave(SpBaseSetPath, MergeSet, M_DEFAULT);

}

void CMLClassCNN::GetVecData4Set(MIL_STRING SetPath,vector<vector<MIL_STRING>>&vecData, MIL_UNIQUE_CLASS_ID& BaseSet) {

    BaseSet = MclassRestore(SetPath, m_MilSystem, M_DEFAULT, M_UNIQUE_ID);
    MIL_INT ClsNb1, Nb;
    MclassInquire(BaseSet, M_CONTEXT, M_NUMBER_OF_CLASSES + M_TYPE_MIL_INT, &ClsNb1);
    MclassInquire(BaseSet, M_DEFAULT, M_NUMBER_OF_ENTRIES + M_TYPE_MIL_INT, &Nb);

    for (int i = 0; i < Nb; i++) {
        MIL_STRING P;
        std::vector<MIL_INT> GTIdx;
        MclassInquireEntry(BaseSet, i, M_DEFAULT_KEY, M_DEFAULT, M_ENTRY_IMAGE_PATH_ABS, P);
        MclassInquireEntry(BaseSet, i, M_DEFAULT_KEY, M_REGION_INDEX(0), M_CLASS_INDEX_GROUND_TRUTH, GTIdx);
        vecData[GTIdx[0]].emplace_back(P);
    }
}


void CMLClassCNN::GenralMergeSet(
    MIL_STRING BaseDataDir,
    MIL_STRING TagDataDir,
    vector<MIL_STRING> BaseClsNames, 
    vector<vector<MIL_STRING>>  vecTagData,
    MIL_UNIQUE_CLASS_ID& MergeSet ) {

    MIL_INT m_ClassesNum = BaseClsNames.size();

    for (MIL_INT i = 0; i < m_ClassesNum; i++)
    {
        //MergeSet step1：加入Icon 到MergeSet 
        MclassControl(MergeSet, M_DEFAULT, M_CLASS_ADD, BaseClsNames[i]);
        //MIL_STRING ClassIconPrefix = TagDataDir + DataSetType + L"//" + BaseClsNames[i];
        MIL_STRING ClassIconPath = TagDataDir + BaseClsNames[i] + L".mim";
        MIL_UNIQUE_BUF_ID IconImageId = MbufRestore(ClassIconPath, m_MilSystem, M_UNIQUE_ID);
        MclassControl(MergeSet, M_CLASS_INDEX(i), M_CLASS_ICON_ID, IconImageId);

        //MergeSet step2：加入vecTagData 到MergeSet ,并创建到本地文件夹
        MIL_INT MergeNb_1;
        MclassInquire(MergeSet, M_DEFAULT, M_NUMBER_OF_ENTRIES + M_TYPE_MIL_INT, &MergeNb_1);
        vector<MIL_STRING> TagDataImgs = vecTagData[i];
        for (int j = 0; j < TagDataImgs.size(); j++) {
            MclassControl(MergeSet, M_DEFAULT, M_ENTRY_ADD, M_DEFAULT);
            MclassControlEntry(MergeSet, MergeNb_1 + j, M_DEFAULT_KEY, M_REGION_INDEX(0), M_CLASS_INDEX_GROUND_TRUTH, i, M_NULL, M_DEFAULT);
            MclassControlEntry(MergeSet, MergeNb_1 + j, M_DEFAULT_KEY, M_DEFAULT, M_ENTRY_IMAGE_PATH, M_DEFAULT, TagDataImgs[j], M_DEFAULT);
        }
    }

    MclassControl(MergeSet, M_CONTEXT, M_CONSOLIDATE_ENTRIES_INTO_FOLDER, BaseDataDir);
}


void CMLClassCNN::MergeVecData2Set(
    MIL_UNIQUE_CLASS_ID& MergeSet,
    vector<vector<MIL_STRING>> vecData)
{
    MIL_INT ClsNb1;
    MclassInquire(MergeSet, M_CONTEXT, M_NUMBER_OF_CLASSES + M_TYPE_MIL_INT, &ClsNb1);
    for (MIL_INT i = 0; i < ClsNb1; i++)
    {
        MIL_INT MergeNb;
        vector<MIL_STRING> Imgs = vecData[i];
        MclassInquire(MergeSet, M_DEFAULT, M_NUMBER_OF_ENTRIES + M_TYPE_MIL_INT, &MergeNb);
        for (int k = 0; k < Imgs.size(); k++) {
            MclassControl(MergeSet, M_DEFAULT, M_ENTRY_ADD, M_DEFAULT);
            MclassControlEntry(MergeSet, MergeNb + k, M_DEFAULT_KEY, M_REGION_INDEX(0), M_CLASS_INDEX_GROUND_TRUTH, i, M_NULL, M_DEFAULT);
            MclassControlEntry(MergeSet, MergeNb + k, M_DEFAULT_KEY, M_DEFAULT, M_ENTRY_IMAGE_PATH, M_DEFAULT, Imgs[k], M_DEFAULT);
        }
    }
}


    void CMLClassCNN::Merge2Set(
        MIL_UNIQUE_CLASS_ID& BaseSet1,
        MIL_UNIQUE_CLASS_ID& BaseSet2,
        MIL_STRING BaseDataDir,
        MIL_STRING MergeSetName)
    {
        //要求Tag Folder必须包含所有
        //MIL_STRING DataSetType = L"C";
        //MIL_STRING BaseSetPreName = L"BaseSet_";
        //MIL_STRING BaseSetPath = BaseDataDir + BaseSetPreName + DataSetType + L".mclassd";
        //BaseSet = MclassRestore(BaseSetPath, m_MilSystem, M_DEFAULT, M_UNIQUE_ID);

        MIL_INT NbEnt1 = 0;
        MIL_INT ClsNb1;
        MclassInquire(BaseSet1, M_DEFAULT, M_NUMBER_OF_ENTRIES + M_TYPE_MIL_INT, &NbEnt1);

        MclassInquire(BaseSet1, M_CONTEXT, M_NUMBER_OF_CLASSES + M_TYPE_MIL_INT, &ClsNb1);

        vector<vector<MIL_STRING>> vecBaseData1, vecTagData, vecTagBaseData, vecPBaseData;

        //获取BaseSet1内容
        vecBaseData1.resize(ClsNb1);
        for (int i = 0; i < NbEnt1; i++) {
            MIL_STRING P;
            std::vector<MIL_INT> GTIdx;
            MclassInquireEntry(BaseSet1, i, M_DEFAULT_KEY, M_DEFAULT, M_ENTRY_IMAGE_PATH_ABS, P);
            MclassInquireEntry(BaseSet1, i, M_DEFAULT_KEY, M_REGION_INDEX(0), M_CLASS_INDEX_GROUND_TRUTH, GTIdx);
            vecBaseData1[GTIdx[0]].emplace_back(P);
        }
        for (MIL_INT i = 0; i < ClsNb1; i++)
        {
            MIL_INT NbEnt2;
            MclassInquire(BaseSet2, M_DEFAULT, M_NUMBER_OF_ENTRIES + M_TYPE_MIL_INT, &NbEnt2);
            vector<MIL_STRING> BaseImgs_1 = vecBaseData1[i];
            for (int j = 0; j < BaseImgs_1.size(); j++) {
                MclassControl(BaseSet2, M_DEFAULT, M_ENTRY_ADD, M_DEFAULT);
                MclassControlEntry(BaseSet2, NbEnt2 + j, M_DEFAULT_KEY, M_REGION_INDEX(0), M_CLASS_INDEX_GROUND_TRUTH, i, M_NULL, M_DEFAULT);
                MclassControlEntry(BaseSet2, NbEnt2 + j, M_DEFAULT_KEY, M_DEFAULT, M_ENTRY_IMAGE_PATH, M_DEFAULT, BaseImgs_1[j], M_DEFAULT);
            }
        }
        MIL_STRING MergeSetPath = BaseDataDir  + MergeSetName + L".mclassd";
        MclassSave(MergeSetPath, BaseSet2, M_DEFAULT);

    }


void CMLClassCNN::ConstructMergeRCDataset(
    MIL_STRING AuthorName,
    MIL_STRING strProject,
    MIL_STRING BaseDataDir,
    MIL_STRING TagDataDir,
    vector<MIL_DOUBLE> vecSampleRatio,
    MIL_UNIQUE_CLASS_ID& BaseDataSet,
    MIL_UNIQUE_CLASS_ID& TagDataSet)
{
    
    //提取ClassName

    MIL_STRING Tag_RDataDir = TagDataDir + L"R/";
    //step1:查询Base、Tag的ClsNames
    vector<MIL_STRING>BaseClsNames, TagClsNames;
    MIL_STRING BaseImgDir = BaseDataDir + MIL_TEXT("Images\\");
    MIL_STRING BaseIconDir = BaseDataDir + MIL_TEXT("Icons\\");
    string strBaseImgDir, strTagDataDir;
    m_AIParse->MIL_STRING2string(BaseImgDir, strBaseImgDir);
    m_AIParse->MIL_STRING2string(Tag_RDataDir, strTagDataDir);
    m_AIParse->getFoldersInFolder(strBaseImgDir, BaseClsNames);
    m_AIParse->getFoldersInFolder(strTagDataDir, TagClsNames);
    bool bTagSameClass = isTagSameClass(BaseClsNames, TagClsNames);
    m_ClassesNum = BaseClsNames.size();
    //计算bTagSameClass
    if (bTagSameClass)
    {
        //读取BaseDataSet_C.mclassd 、BaseDataSet_R.mclassd
        MIL_STRING BaseData_CPath = BaseDataDir + MIL_TEXT("/BaseDataSet_C.mclassd");
        MIL_UNIQUE_CLASS_ID BaseDataSet_C = MclassRestore(BaseData_CPath, m_MilSystem, M_DEFAULT, M_UNIQUE_ID);
        MIL_INT NbEntries = 0;
        MclassInquire(BaseDataSet_C, M_DEFAULT, M_NUMBER_OF_ENTRIES + M_TYPE_MIL_INT, &NbEntries);
        MclassInquire(BaseDataSet_C, M_CONTEXT, M_NUMBER_OF_CLASSES + M_TYPE_MIL_INT, &m_ClassesNum);//读取BaseDataSet_R求并集
        map<MIL_INT, vector<MIL_STRING>>BaseData;
        vector<vector<MIL_STRING>> vecP;
        vecP.resize(m_ClassesNum);
        for (int i = 0; i < NbEntries; i++) {
            MIL_STRING P;
            std::vector<MIL_INT> GTIdx;
            MclassInquireEntry(BaseDataSet_C, i, M_DEFAULT_KEY, M_DEFAULT, M_ENTRY_IMAGE_PATH_ABS, P);
            MclassInquireEntry(BaseDataSet_C, i, M_DEFAULT_KEY, M_REGION_INDEX(0), M_CLASS_INDEX_GROUND_TRUTH, GTIdx);
            vecP[GTIdx[0]].emplace_back(P);
        }
        //生成UpdateDataSet = PartBaseDataset+TagDataset
            //生成TagDataSet
    
    }



    else {
        //生成BaseDataSet = BaseDataSet+TagDataset

    }

   
    




    //提取BaseData 中的类型和图片：1、部分提取；2、所有提取


    //遍历BaseImg所有类型的图片，1、部分提取；2、所有提取
    map<MIL_STRING, vector<MIL_STRING>>PartBaseData, BaseData;
    for (int i = 0; i < BaseClsNames.size(); i++) {
        MIL_STRING FolderDir = BaseImgDir + BaseClsNames[i];
        string strFolderDir;
        m_AIParse->MIL_STRING2string(FolderDir, strFolderDir);

        vector<MIL_STRING>BaseClsImg;
        m_AIParse->getFilesInFolder(strFolderDir, "bmp", BaseClsImg);

        int nFileNum = ceil(BaseClsImg.size() * vecSampleRatio[i]);
        random_shuffle(BaseClsImg.begin(), BaseClsImg.end());

        vector<MIL_STRING> PartClsImg(BaseClsImg.begin(), BaseClsImg.begin() + nFileNum);
        PartBaseData.insert(pair<MIL_STRING, vector<MIL_STRING>>(BaseClsNames[i], PartClsImg));
        BaseData.insert(pair<MIL_STRING, vector<MIL_STRING>>(BaseClsNames[i], BaseClsImg));
    }

    //遍历TagImg所有类型的图片,获取TagData
    map<MIL_STRING, vector<MIL_STRING>>TagData;
    for (int i = 0; i < TagClsNames.size(); i++) {
        MIL_STRING TagFolderDir = TagDataDir + TagClsNames[i];
        string strTagFolderDir;
        m_AIParse->MIL_STRING2string(TagFolderDir, strTagFolderDir);
        vector<MIL_STRING>TagClsImg;
        m_AIParse->getFilesInFolder(strTagFolderDir, "bmp", TagClsImg);
        TagData.insert(pair<MIL_STRING, vector<MIL_STRING>>(BaseClsNames[i], TagClsImg));
    }
    //生成TagDataSet
    //MIL_UNIQUE_CLASS_ID TagDataSet = MclassAlloc(m_MilSystem, M_DATASET_IMAGES, M_DEFAULT, M_UNIQUE_ID);
    MclassControl(TagDataSet, M_DEFAULT, M_AUTHOR_ADD, AuthorName);
    for (MIL_INT i = 0; i < TagClsNames.size(); i++)
    {
        //加入Icon
        MclassControl(TagDataSet, M_DEFAULT, M_CLASS_ADD, TagClsNames[i]);
        MIL_STRING ClassIcon = TagDataDir + TagClsNames[i] + L".mim";
        MIL_UNIQUE_BUF_ID IconImageId = MbufRestore(ClassIcon, m_MilSystem, M_UNIQUE_ID);
        MclassControl(TagDataSet, M_CLASS_INDEX(i), M_CLASS_ICON_ID, IconImageId);
        //加入Img
        MIL_INT ClassIndex = i;
        MIL_INT NbEntries;
        MclassInquire(TagDataSet, M_DEFAULT, M_NUMBER_OF_ENTRIES + M_TYPE_MIL_INT, &NbEntries);
        vector<MIL_STRING> TagDataImgs = TagData[TagClsNames[i]];
        for (int j = 0; j < TagDataImgs.size(); j++) {
            MclassControl(TagDataSet, M_DEFAULT, M_ENTRY_ADD, M_DEFAULT);
            MclassControlEntry(TagDataSet, NbEntries + j, M_DEFAULT_KEY, M_REGION_INDEX(0), M_CLASS_INDEX_GROUND_TRUTH, ClassIndex, M_NULL, M_DEFAULT);
            MclassControlEntry(TagDataSet, NbEntries + j, M_DEFAULT_KEY, M_DEFAULT, M_ENTRY_IMAGE_PATH, M_DEFAULT, TagDataImgs[j], M_DEFAULT);
            MclassControlEntry(TagDataSet, NbEntries + j, M_DEFAULT_KEY, M_REGION_INDEX(0), M_AUTHOR_NAME, M_DEFAULT, AuthorName, M_DEFAULT);
        }
    }
    MclassControl(TagDataSet, M_CONTEXT, M_CONSOLIDATE_ENTRIES_INTO_FOLDER, BaseDataDir);
    //MIL_STRING TagDataSetPath = BaseDataDir + MIL_TEXT("TagDataSet.mclassd");
    //MclassSave(TagDataSetPath, TagDataSet, M_DEFAULT);
    if (bTagSameClass)
    {
        //生成UpdateDataSet = PartBaseDataset+TagDataset
        for (MIL_INT i = 0; i < BaseClsNames.size(); i++)
        {
            //加入Img
            MIL_INT ClassIndex = i;
            MIL_INT NbEntries;
            MclassInquire(TagDataSet, M_DEFAULT, M_NUMBER_OF_ENTRIES + M_TYPE_MIL_INT, &NbEntries);
            vector<MIL_STRING> PartBaseImgData = PartBaseData[BaseClsNames[i]];
            for (int j = 0; j < PartBaseImgData.size(); j++) {
                MclassControl(TagDataSet, M_DEFAULT, M_ENTRY_ADD, M_DEFAULT);
                MclassControlEntry(TagDataSet, NbEntries + j, M_DEFAULT_KEY, M_REGION_INDEX(0), M_CLASS_INDEX_GROUND_TRUTH, ClassIndex, M_NULL, M_DEFAULT);
                MclassControlEntry(TagDataSet, NbEntries + j, M_DEFAULT_KEY, M_DEFAULT, M_ENTRY_IMAGE_PATH, M_DEFAULT, PartBaseImgData[j], M_DEFAULT);
                MclassControlEntry(TagDataSet, NbEntries + j, M_DEFAULT_KEY, M_REGION_INDEX(0), M_AUTHOR_NAME, M_DEFAULT, AuthorName, M_DEFAULT);
            }
        }
        MIL_STRING UpdateDataSetPath = BaseDataDir + MIL_TEXT("UpdateDataSet.mclassd");
        MclassSave(UpdateDataSetPath, TagDataSet, M_DEFAULT);
        //生成BaseDataSet = AllData
        vector<MIL_STRING>NBaseClsNames;
        m_AIParse->getFoldersInFolder(strBaseImgDir, NBaseClsNames);
        //MIL_UNIQUE_CLASS_ID BaseDataSet = MclassAlloc(m_MilSystem, M_DATASET_IMAGES, M_DEFAULT, M_UNIQUE_ID);
        for (MIL_INT i = 0; i < NBaseClsNames.size(); i++)
        {
            //加入Icon
            MclassControl(BaseDataSet, M_DEFAULT, M_CLASS_ADD, NBaseClsNames[i]);
            MIL_STRING ClassIcon = BaseIconDir + NBaseClsNames[i] + L".mim";
            MIL_UNIQUE_BUF_ID IconImageId = MbufRestore(ClassIcon, m_MilSystem, M_UNIQUE_ID);
            MclassControl(BaseDataSet, M_CLASS_INDEX(i), M_CLASS_ICON_ID, IconImageId);
            //加入Img
            MIL_STRING BaseFolderDir = BaseImgDir + NBaseClsNames[i];
            string strFolderDir;
            m_AIParse->MIL_STRING2string(BaseFolderDir, strFolderDir);
            vector<MIL_STRING>BaseClsImg;
            m_AIParse->getFilesInFolder(strFolderDir, "bmp", BaseClsImg);
            MIL_INT ClassIndex = i;
            MIL_INT NbEntries;
            MclassInquire(BaseDataSet, M_DEFAULT, M_NUMBER_OF_ENTRIES + M_TYPE_MIL_INT, &NbEntries);
            for (int j = 0; j < BaseClsImg.size(); j++) {
                MIL_INT Cd = i;
                MclassControl(BaseDataSet, M_DEFAULT, M_ENTRY_ADD, M_DEFAULT);
                MclassControlEntry(BaseDataSet, NbEntries + j, M_DEFAULT_KEY, M_REGION_INDEX(0), M_CLASS_INDEX_GROUND_TRUTH, i, M_NULL, M_DEFAULT);
                MclassControlEntry(BaseDataSet, NbEntries + j, M_DEFAULT_KEY, M_DEFAULT, M_ENTRY_IMAGE_PATH, M_DEFAULT, BaseClsImg[j], M_DEFAULT);
                //MclassControlEntry(BaseDataSet, NbEntries + j, M_DEFAULT_KEY, M_REGION_INDEX(0), M_AUTHOR_NAME, M_DEFAULT, L"A", M_DEFAULT);
            }
        }
        MIL_STRING BaseDataSetPath = BaseDataDir + MIL_TEXT("BaseDataSet.mclassd");
        MclassSave(BaseDataSetPath, BaseDataSet, M_DEFAULT);
    }
    else {
        //生成 BaseDataSet = AllData
        vector<MIL_STRING>NBaseClsNames;
        m_AIParse->getFoldersInFolder(strBaseImgDir, NBaseClsNames);
        //MIL_UNIQUE_CLASS_ID BaseDataSet = MclassAlloc(m_MilSystem, M_DATASET_IMAGES, M_DEFAULT, M_UNIQUE_ID);
        for (MIL_INT i = 0; i < NBaseClsNames.size(); i++)
        {
            //加入Icon
            MclassControl(BaseDataSet, M_DEFAULT, M_CLASS_ADD, NBaseClsNames[i]);
            MIL_STRING ClassIcon = BaseIconDir + NBaseClsNames[i] + L".mim";
            MIL_UNIQUE_BUF_ID IconImageId = MbufRestore(ClassIcon, m_MilSystem, M_UNIQUE_ID);
            MclassControl(BaseDataSet, M_CLASS_INDEX(i), M_CLASS_ICON_ID, IconImageId);
            //加入Img
            MIL_STRING BaseFolderDir = BaseImgDir + NBaseClsNames[i];
            string strFolderDir;
            m_AIParse->MIL_STRING2string(BaseFolderDir, strFolderDir);
            vector<MIL_STRING>BaseClsImg;
            m_AIParse->getFilesInFolder(strFolderDir, "bmp", BaseClsImg);
            MIL_INT ClassIndex = i;
            MIL_INT NbEntries;
            MclassInquire(BaseDataSet, M_DEFAULT, M_NUMBER_OF_ENTRIES + M_TYPE_MIL_INT, &NbEntries);
            for (int j = 0; j < BaseClsImg.size(); j++) {
                MIL_INT Cd = i;
                MclassControl(BaseDataSet, M_DEFAULT, M_ENTRY_ADD, M_DEFAULT);
                MclassControlEntry(BaseDataSet, NbEntries + j, M_DEFAULT_KEY, M_REGION_INDEX(0), M_CLASS_INDEX_GROUND_TRUTH, i, M_NULL, M_DEFAULT);
                MclassControlEntry(BaseDataSet, NbEntries + j, M_DEFAULT_KEY, M_DEFAULT, M_ENTRY_IMAGE_PATH, M_DEFAULT, BaseClsImg[j], M_DEFAULT);
                //MclassControlEntry(BaseDataSet, NbEntries + j, M_DEFAULT_KEY, M_REGION_INDEX(0), M_AUTHOR_NAME, M_DEFAULT, L"A", M_DEFAULT);
            }
        }
        MIL_STRING BaseDataSetPath = BaseDataDir + MIL_TEXT("BaseDataSet.mclassd");
        MclassSave(BaseDataSetPath, BaseDataSet, M_DEFAULT);
        MIL_STRING UpdateDataSetPath = BaseDataDir + MIL_TEXT("UpdateDataSet.mclassd");
        MclassSave(UpdateDataSetPath, TagDataSet, M_DEFAULT);
    }
}

void CMLClassCNN::InitializeMergeRCDataset(MIL_STRING AuthorName, 
    MIL_STRING BaseDataDir, 
    MIL_STRING TagDataDir,
    vector<MIL_STRING>TagClassNames,
    MIL_UNIQUE_CLASS_ID& BaseDataSet, 
    MIL_UNIQUE_CLASS_ID& TagDataSet)
{  
    string strTagDataDir, strTagDataDir_C;
    MIL_STRING TagDataDir_C = TagDataDir + L"C/";
    MIL_STRING TagDataDir_R = TagDataDir + L"R/";

    m_AIParse->MIL_STRING2string(TagDataDir, strTagDataDir);
    m_AIParse->MIL_STRING2string(TagDataDir_C, strTagDataDir_C);

    vector<MIL_STRING > TagClassIcons;
    m_AIParse->getFilesInFolder(strTagDataDir, "mim", TagClassIcons);
    //m_AIParse->getFoldersInFolder(strTagDataDir_C, TagClassNames);

    MIL_UNIQUE_CLASS_ID BaseDataSet_C = MclassAlloc(m_MilSystem, M_DATASET_IMAGES, M_DEFAULT, M_UNIQUE_ID);
    ConstructDataset(TagClassNames, TagClassIcons, AuthorName, TagDataDir_C, BaseDataDir, BaseDataSet_C);
    MclassControl(BaseDataSet_C, M_CONTEXT, M_CONSOLIDATE_ENTRIES_INTO_FOLDER, BaseDataDir);
    MIL_STRING CropBaseDatasetPath = BaseDataDir + MIL_TEXT("COMPSetC.mclassd");
    MclassSave(CropBaseDatasetPath, BaseDataSet_C, M_DEFAULT);


    MIL_UNIQUE_CLASS_ID BaseDataSet_R = MclassAlloc(m_MilSystem, M_DATASET_IMAGES, M_DEFAULT, M_UNIQUE_ID);
    ConstructDataset(TagClassNames, TagClassIcons, AuthorName, TagDataDir_R, BaseDataDir, BaseDataSet_R);
    MclassControl(BaseDataSet_R, M_CONTEXT, M_CONSOLIDATE_ENTRIES_INTO_FOLDER, BaseDataDir);
    MIL_STRING BaseDataset_RPath = BaseDataDir + MIL_TEXT("COMPSetR.mclassd");
    MclassSave(BaseDataset_RPath, BaseDataSet_R, M_DEFAULT);

}



//void CMLClassCNN::ExpanDataset(
//    map<MIL_STRING, int>  mapClassName,
//    vector<MIL_STRING> ClassIcon,
//    MIL_STRING AuthorName,
//    MIL_STRING OriginalDataPath,
//    const MIL_STRING& WorkingDataPath,
//    MIL_UNIQUE_CLASS_ID& Dataset)
//{
//    MIL_INT  NumberOfClasses = mapClassName.size();
//
//    if (M_NULL == Dataset)
//    {
//        Dataset = MclassAlloc(m_MilSystem, M_DATASET_IMAGES, M_DEFAULT, M_UNIQUE_ID);
//    }
//
//    MclassControl(Dataset, M_DEFAULT, M_AUTHOR_ADD, AuthorName);
//
//    map<MIL_STRING, int> ::iterator it;
//    int i = 0;
//    for (it = mapClassName.begin(); it != mapClassName.end(); it++) {
//        MclassControl(Dataset, M_DEFAULT, M_CLASS_ADD, it->first);
//        MIL_UNIQUE_BUF_ID IconImageId = MbufRestore(ClassIcon[i], m_MilSystem, M_UNIQUE_ID);
//        MclassControl(Dataset, M_CLASS_INDEX(it->second), M_CLASS_ICON_ID, IconImageId);
//        AddClassToDataset(it->second, OriginalDataPath, it->first, AuthorName, Dataset);
//        i++;
//    }
// 
//
//    //for (MIL_INT ClassIdx = 0; ClassIdx < NumberOfClasses; ClassIdx++)
//    //{
//    //    AddClassToDataset(ClassIdx, OriginalDataPath, ClassName[ClassIdx], AuthorName, Dataset);
//    //}
//
//}

void CMLClassCNN::GeneralDataset(vector<MIL_STRING> ClassName,
    vector<MIL_STRING> ClassIcon,
    MIL_STRING AuthorName,
    MIL_STRING OriginalDataPath,
    MIL_STRING WorkingDataPath)
{
    MIL_UNIQUE_CLASS_ID  Dataset = MclassAlloc(m_MilSystem, M_DATASET_IMAGES, M_DEFAULT, M_UNIQUE_ID);

    ConstructDataset(ClassName,ClassIcon,AuthorName,OriginalDataPath,WorkingDataPath,Dataset);
 
    CreateFolder(WorkingDataPath);
    MclassControl(Dataset, M_CONTEXT, M_CONSOLIDATE_ENTRIES_INTO_FOLDER, WorkingDataPath);

    ////默认该数据已经由软件汇总，汇总数据，然后再保存
    //MIL_STRING OriginalAllDataPath = WorkingDataPath + MIL_TEXT("Images\\");
    //MIL_STRING OriginalAllIconsPath = WorkingDataPath + MIL_TEXT("Icons\\");
    //std::vector<MIL_STRING>ClassName_A;
    //std::vector<MIL_STRING>ClassIcon_A;
    //MIL_UNIQUE_CLASS_ID  AllDataset = MclassAlloc(m_MilSystem, M_DATASET_IMAGES, M_DEFAULT, M_UNIQUE_ID);

    //string OrgAIconsPath;
    //m_AIParse->MIL_STRING2string(OriginalAllIconsPath, OrgAIconsPath);
    //m_AIParse->getFilesInFolder(OrgAIconsPath,"mim", ClassIcon_A);
    //for (int i = 0; i < ClassIcon_A.size(); i++) {
    //    MIL_STRING ClassIconPath = ClassIcon_A[i];
    //    MIL_STRING::size_type iPos = ClassIconPath.find_last_of('\\') + 1;
    //    MIL_STRING filename = ClassIconPath.substr(iPos, ClassIconPath.length() - iPos);

    //    MIL_STRING::size_type iPosP = filename.find_last_of('.') + 1;
    //    MIL_STRING ClassName = filename.substr(0, iPosP - 1);

    //    ClassName_A.emplace_back(ClassName);
    //}
    //ConstructDataset(ClassName_A, ClassIcon_A, AuthorName[0], OriginalAllDataPath, WorkingDataPath, AllDataset);
    MIL_STRING WorkDatasetPath = WorkingDataPath + MIL_TEXT("DataSet.mclassd");
    MclassSave(WorkDatasetPath, Dataset, M_DEFAULT);


}

void CMLClassCNN::ConstructDataContext(DataContextParasStruct DataCtxParas, MIL_UNIQUE_CLASS_ID& DataContext)
{
    if (M_NULL == DataContext)
    {
        DataContext = MclassAlloc(m_MilSystem, M_PREPARE_IMAGES_CNN, M_DEFAULT, M_UNIQUE_ID);
    }
    //数据保存
    CreateFolder(DataCtxParas.PreparedDataFolder);
    MclassControl(DataContext, M_CONTEXT, M_PREPARED_DATA_FOLDER, DataCtxParas.PreparedDataFolder);
    MclassControl(DataContext, M_CONTEXT, M_AUGMENT_NUMBER_FACTOR, DataCtxParas.AugParas.AugmentationNumPerImage);
    if (DataCtxParas.ImageSizeX > 0 && DataCtxParas.ImageSizeY > 0)
    {
        MclassControl(DataContext, M_CONTEXT, M_SIZE_MODE, M_USER_DEFINED);
        MclassControl(DataContext, M_CONTEXT, M_SIZE_X, DataCtxParas.ImageSizeX);
        MclassControl(DataContext, M_CONTEXT, M_SIZE_Y, DataCtxParas.ImageSizeY);
        m_ImageSizeX = DataCtxParas.ImageSizeX;
        m_ImageSizeY = DataCtxParas.ImageSizeY;
    }
    MclassControl(DataContext, M_CONTEXT, M_DESTINATION_FOLDER_MODE, M_OVERWRITE);
    if (DataCtxParas.ResizeModel == 1) {
        MclassControl(DataContext, M_CONTEXT, M_RESIZE_SCALE_FACTOR, M_FILL_DESTINATION);
    }

    MIL_ID AugmentContext;
    MclassInquire(DataContext, M_CONTEXT, M_AUGMENT_CONTEXT_ID + M_TYPE_MIL_ID, &AugmentContext);

    if (DataCtxParas.AugParas.RotateAngleDelta > 0)
    {
        MimControl(AugmentContext, M_AUG_ROTATION_OP, M_ENABLE);
        MimControl(AugmentContext, M_AUG_ROTATION_OP_ANGLE_DELTA, DataCtxParas.AugParas.RotateAngleDelta);
    }

    if ((DataCtxParas.AugParas.ScaleFactorMin > 0 && DataCtxParas.AugParas.ScaleFactorMin != 1.0)
        || (DataCtxParas.AugParas.ScaleFactorMax > 0 && DataCtxParas.AugParas.ScaleFactorMax != 1.0))
    {
        MimControl(AugmentContext, M_AUG_SCALE_OP, M_ENABLE);
        MimControl(AugmentContext, M_AUG_SCALE_OP_FACTOR_MIN, DataCtxParas.AugParas.ScaleFactorMin);
        MimControl(AugmentContext, M_AUG_SCALE_OP_FACTOR_MAX, DataCtxParas.AugParas.ScaleFactorMax);
    }
    // Rotation augmentation and presets in the prepare data context.
// MclassControl(TrainPrepareDataCtx, M_CONTEXT, M_PRESET_ROTATION, M_ENABLE);

    if (DataCtxParas.AugParas.IntyDeltaAdd > 0) {
        MimControl(AugmentContext, M_AUG_INTENSITY_ADD_OP, M_ENABLE);
        MimControl(AugmentContext, M_AUG_INTENSITY_ADD_OP_DELTA, DataCtxParas.AugParas.IntyDeltaAdd);
    }

    if (DataCtxParas.AugParas.DirIntyMax > 0) {
        MimControl(AugmentContext, M_AUG_LIGHTING_DIRECTIONAL_OP, M_ENABLE);
        MimControl(AugmentContext, M_AUG_LIGHTING_DIRECTIONAL_OP_INTENSITY_MAX, DataCtxParas.AugParas.DirIntyMax);
        MimControl(AugmentContext, M_AUG_LIGHTING_DIRECTIONAL_OP_INTENSITY_MIN, DataCtxParas.AugParas.DirIntyMin);
    }
    // Noise augmentation and presets in the prepare data context.
    if (DataCtxParas.AugParas.GaussNoiseDelta > 0.0 || DataCtxParas.AugParas.GaussNoiseStdev > 0.0)
    {
        MimControl(AugmentContext, M_AUG_NOISE_GAUSSIAN_ADDITIVE_OP, M_ENABLE);
        MimControl(AugmentContext, M_AUG_NOISE_GAUSSIAN_ADDITIVE_OP_STDDEV, DataCtxParas.AugParas.GaussNoiseStdev);
        MimControl(AugmentContext, M_AUG_NOISE_GAUSSIAN_ADDITIVE_OP_STDDEV_DELTA, DataCtxParas.AugParas.GaussNoiseDelta);
    }
    //// Smoothness augmentation and presets in the prepare data context.
    if (DataCtxParas.AugParas.SmoothnessMin > 0.0 && DataCtxParas.AugParas.SmoothnessMax >= DataCtxParas.AugParas.SmoothnessMin)
    {
        MimControl(AugmentContext, M_AUG_SMOOTH_DERICHE_OP, M_ENABLE);
        MimControl(AugmentContext, M_AUG_SMOOTH_DERICHE_OP_FACTOR_MIN, DataCtxParas.AugParas.SmoothnessMin);
        MimControl(AugmentContext, M_AUG_SMOOTH_DERICHE_OP_FACTOR_MAX, DataCtxParas.AugParas.SmoothnessMax);
    }

    if (DataCtxParas.AugParas.GammaValue > 0)
    {
        MimControl(AugmentContext, M_AUG_GAMMA_OP, M_ENABLE);
        MimControl(AugmentContext, M_AUG_GAMMA_OP_VALUE, DataCtxParas.AugParas.GammaValue);
        MimControl(AugmentContext, M_AUG_GAMMA_OP_DELTA, DataCtxParas.AugParas.GammaDelta);
    }
}

void CMLClassCNN::PrepareDataset(MIL_UNIQUE_CLASS_ID& DatasetContext, 
    MIL_UNIQUE_CLASS_ID& PrepareDataset, 
    MIL_UNIQUE_CLASS_ID& PreparedDataset,
    MIL_STRING WorkingDataDir,
    MIL_STRING DatasetName)
{
    PreparedDataset = MclassAlloc(m_MilSystem, M_DATASET_IMAGES, M_DEFAULT, M_UNIQUE_ID);
    MclassPreprocess(DatasetContext, M_DEFAULT);
    MclassPrepareData(DatasetContext, PrepareDataset, PreparedDataset, M_NULL, M_DEFAULT);
    MclassSave(WorkingDataDir + DatasetName+ MIL_TEXT(".mclassd"), PreparedDataset, M_DEFAULT);
    MclassExport(WorkingDataDir + MIL_TEXT("class_definitions.csv"), M_FORMAT_CSV,PreparedDataset, M_DEFAULT, M_CLASS_DEFINITIONS, M_DEFAULT);
    MclassExport(WorkingDataDir + DatasetName+MIL_TEXT("_entries.csv"), M_FORMAT_CSV, PreparedDataset, M_DEFAULT, M_ENTRIES, M_DEFAULT);
}

void CMLClassCNN::ConstructTrainCtx(ClassifierParasStruct ClassifierParas, MIL_UNIQUE_CLASS_ID& TrainCtx)
{
    if (M_NULL == TrainCtx)
    {
        TrainCtx = MclassAlloc(m_MilSystem, M_TRAIN_CNN, M_DEFAULT, M_UNIQUE_ID);
    }

    CreateFolder(ClassifierParas.TrainDstFolder);
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

void CMLClassCNN::TrainClassifier(MIL_UNIQUE_CLASS_ID& Dataset, 
    MIL_UNIQUE_CLASS_ID& TrainCtx, 
    MIL_UNIQUE_CLASS_ID& PrevClassifierCtx,
    MIL_UNIQUE_CLASS_ID& TrainedClassifierCtx,
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

    MIL_UNIQUE_CLASS_ID TrainRes = MclassAllocResult(m_MilSystem, M_TRAIN_CNN_RESULT, M_DEFAULT, M_UNIQUE_ID);
    HookDataStruct TheHookData;
    TheHookData.MilSystem = m_MilSystem;
    TheHookData.MilDisplay = m_MilDisplay;
    TheHookData.DumpTmpRst = 1;
    TheHookData.ClassifierDumpFile = ClassifierDumpFile;

    TheHookData.DashboardPtr = CDashboardPtr(new CDashboard(
        m_MilSystem,
        0,
        MaxEpoch,
        MinibatchSize,
        LearningRate,
        m_ImageSizeX,
        m_ImageSizeY,
        TrainDatasetNbImages,
        DevDatasetNbImages,
        TrainEngineUsed,
        TrainEngineDescription));

    // Initialize the hook associated to the epoch trained event.
    MclassHookFunction(TrainCtx, M_EPOCH_TRAINED, HookFuncEpoch, &TheHookData);

    // Initialize the hook associated to the mini batch trained event.
    MclassHookFunction(TrainCtx, M_MINI_BATCH_TRAINED, HookFuncMiniBatch, &TheHookData);

    // Initialize the hook associated to the datasets prepared event.
    MclassHookFunction(TrainCtx, M_DATASETS_PREPARED, HookFuncDatasetsPrepared, &TheHookData);
    // Start the training process.
    double time = 0;
    MIL_TEXT_CHAR TheString[512];
    //timeStart();
    MclassTrain(TrainCtx, PrevClassifierCtx, Dataset, M_NULL, TrainRes, M_DEFAULT);
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
        TrainedClassifierCtx = MclassAlloc(m_MilSystem, M_CLASSIFIER_CNN_PREDEFINED, M_DEFAULT, M_UNIQUE_ID);
        MclassCopyResult(TrainRes, M_DEFAULT, TrainedClassifierCtx, M_DEFAULT, M_TRAINED_CLASSIFIER, M_DEFAULT);
    }
}

void CMLClassCNN::PredictBegin(MIL_UNIQUE_CLASS_ID& TrainedCCtx,MIL_ID Image,vector<MIL_DOUBLE>Class_Weights)
{  
    //获取模型输入尺寸
    MclassInquire(TrainedCCtx, M_DEFAULT_SOURCE_LAYER, M_SIZE_X + M_TYPE_MIL_INT, &m_InputSizeX);
    MclassInquire(TrainedCCtx, M_DEFAULT_SOURCE_LAYER, M_SIZE_Y + M_TYPE_MIL_INT, &m_InputSizeY);
    MclassInquire(TrainedCCtx, M_CONTEXT, M_NUMBER_OF_CLASSES + M_TYPE_MIL_INT, &m_ClassesNum);
    MIL_INT m_ImageSizeX = MbufInquire(Image, M_SIZE_X, M_NULL);
    MIL_INT m_ImageSizeY = MbufInquire(Image, M_SIZE_Y, M_NULL);
    if (Class_Weights.size()!= m_ClassesNum) {
        m_ClassWeights.resize(m_ClassesNum, 1.0);
    }
    else {
        m_ClassWeights.resize(m_ClassesNum, 1.0);
        for (int i = 0; i < m_ClassesNum; i++) {
            m_ClassWeights[i] = Class_Weights[i];
        }
    }
}

void CMLClassCNN::FolderImgsPredict(vector<MIL_STRING> FilesInFolder,
    vector<MIL_DOUBLE> Class_Weights, 
    MIL_UNIQUE_CLASS_ID& TrainedClassifierCtx, 
    vector<ClassificationResultStruct>& Result)
{
    int nFileNum = FilesInFolder.size();
    ClassificationResultStruct Result_i; 
    for (int i = 0; i < nFileNum; i++) {
        memset(&Result_i, 0, sizeof(Result_i));
        MIL_ID RawImage = MbufRestore(FilesInFolder[i], m_MilSystem, M_NULL);
        Predict(RawImage, Class_Weights,TrainedClassifierCtx, Result_i);
        Result.emplace_back(Result_i);
        MbufFree(RawImage);
    }

}

void CMLClassCNN::Predict(MIL_ID Image, 
    vector<MIL_DOUBLE> Class_Weights,
    MIL_UNIQUE_CLASS_ID& TrainedClassifierCtx, 
    ClassificationResultStruct& Result)
{
    PredictBegin(TrainedClassifierCtx, Image, Class_Weights);

    //中心crop模式
    //MIL_INT RawImage_X = MbufInquire(Image, M_SIZE_X, M_NULL);
    //MIL_INT RawImage_Y = MbufInquire(Image, M_SIZE_Y, M_NULL);
    //MIL_ID ImageReduce = MbufAlloc2d(m_MilSystem, m_InputSizeX, m_InputSizeY, 8 + M_UNSIGNED, M_IMAGE + M_PROC, M_NULL);
    //MbufClear(ImageReduce, 0);
    //MIL_INT SrcOffX = max(int((RawImage_X - m_InputSizeX) / 2), 0);
    //MIL_INT SrcOffY = max(int((RawImage_Y - m_InputSizeY) / 2), 0);
    //MIL_INT DstOffX = abs(min(int((RawImage_X - m_InputSizeX) / 2), 0));
    //MIL_INT DstOffY = abs(min(int((RawImage_Y - m_InputSizeY) / 2), 0));
    //MIL_INT SizeX = min(RawImage_X, m_InputSizeX);
    //MIL_INT SizeY = min(RawImage_Y, m_InputSizeY);
    //MbufCopyColor2d(Image, ImageReduce, M_ALL_BANDS, SrcOffX, SrcOffY, M_ALL_BANDS, DstOffX, DstOffY, SizeX, SizeY);

    //resize 模式
    MIL_ID ImageReduce = MbufAlloc2d(m_MilSystem, m_InputSizeX, m_InputSizeY, 8 + M_UNSIGNED, M_IMAGE + M_PROC, M_NULL);
    MimResize(Image, ImageReduce, M_FILL_DESTINATION, M_FILL_DESTINATION, M_BILINEAR);



    MIL_INT Status = M_FALSE;
    MclassInquire(TrainedClassifierCtx, M_DEFAULT, M_PREPROCESSED + M_TYPE_MIL_INT, &Status);
    if (M_FALSE == Status)
    {
        //LARGE_INTEGER t1, t2, tc;
        //QueryPerformanceFrequency(&tc);
        //QueryPerformanceCounter(&t1);
        MclassPreprocess(TrainedClassifierCtx, M_DEFAULT);
        //QueryPerformanceCounter(&t2);
        //double time = (double)(t2.QuadPart - t1.QuadPart) / (double)tc.QuadPart;
        //cout << "\nPreprocess_" << m_index << "time is = " << time << "\n" << endl;
    }
    MIL_UNIQUE_CLASS_ID ClassRes = MclassAllocResult(m_MilSystem, M_PREDICT_CNN_RESULT, M_DEFAULT, M_UNIQUE_ID);
    MclassPredict(TrainedClassifierCtx, ImageReduce, ClassRes, M_DEFAULT);
    MbufFree(ImageReduce);

    MclassGetResult(ClassRes, M_GENERAL, M_BEST_CLASS_SCORE + M_TYPE_MIL_DOUBLE, &Result.PredictScore);
    MclassGetResult(ClassRes, M_GENERAL, M_BEST_CLASS_INDEX + M_TYPE_MIL_INT, &Result.PredictClass);
    vector<MIL_DOUBLE>WeightScores;
    for (int i = 0; i < m_ClassWeights.size(); i++) {
        MIL_DOUBLE Score;
        MclassGetResult(ClassRes, M_CLASS_INDEX(i), M_CLASS_SCORES + M_TYPE_MIL_DOUBLE, &Score);
        MIL_DOUBLE WeightScore = Score * m_ClassWeights[i];
        WeightScores.emplace_back(WeightScore);
        Result.AllScores.emplace_back(Score);
    }
    vector<MIL_DOUBLE>::iterator biggest = max_element(begin(WeightScores), end(WeightScores));
    Result.PredictClass = distance(begin(WeightScores), biggest);
    MclassInquire(TrainedClassifierCtx, M_CLASS_INDEX(Result.PredictClass), M_CLASS_NAME, Result.PredictClassName);
}

void CMLClassCNN::Predict(MIL_ID Image, 
    vector<int> blob_px, 
    vector<int> blob_py,
    std::vector<MIL_DOUBLE> Class_Weights,
    MIL_UNIQUE_CLASS_ID& TrainedClassifierCtx,
    ClassificationResultStruct& Result)
{
    vector<Box> vecCropedBoxes;
    m_AIParse->cropBox(blob_px,blob_py,
        m_InputSizeX,
        m_InputSizeY,
        m_ImageSizeX,
        m_ImageSizeY,
        vecCropedBoxes);
    int CropedBN = vecCropedBoxes.size();
    for (int i = 0; i < CropedBN; i++) {
        ClassificationResultStruct tmpRst;

        MIL_ID InImg = MbufAlloc2d(m_MilSystem,m_InputSizeX,m_InputSizeY,
            8 + M_UNSIGNED,M_IMAGE + M_PROC, M_NULL);
        MbufClear(InImg, 0);
        auto CropBoxes = vecCropedBoxes[i];
        MbufCopyColor2d(Image,InImg,M_ALL_BANDS,CropBoxes.x1, CropBoxes.y1,
            M_ALL_BANDS, 0, 0,m_InputSizeX, m_InputSizeY);
        Predict(InImg, Class_Weights, TrainedClassifierCtx, tmpRst);
        for (int nCNum = 0; nCNum < m_ClassesNum; nCNum++) {
            Result.AllScores[nCNum] += tmpRst.AllScores[nCNum];
        }    
    }

    vector<MIL_DOUBLE>::iterator biggest = max_element(std::begin(Result.AllScores), std::end(Result.AllScores));
    Result.PredictClass = distance(begin(Result.AllScores), biggest);
    Result.PredictScore = Result.AllScores[Result.PredictClass]/(double)m_ClassesNum;
    MclassInquire(TrainedClassifierCtx, M_CLASS_INDEX(Result.PredictClass), M_CLASS_NAME, Result.PredictClassName);
}

void CMLClassCNN::AddClassDescription(MIL_ID Dataset, const MIL_STRING& AuthorName, std::vector<MIL_STRING> ClassName, std::vector<MIL_STRING> ClassIcon, MIL_INT NumberOfClasses)
{
    MclassControl(Dataset, M_DEFAULT, M_AUTHOR_ADD, AuthorName);

    for (MIL_INT i = 0; i < NumberOfClasses; i++)
    {
        MclassControl(Dataset, M_DEFAULT, M_CLASS_ADD, ClassName[i]);
        MIL_UNIQUE_BUF_ID IconImageId = MbufRestore(ClassIcon[i], m_MilSystem, M_UNIQUE_ID);
        MclassControl(Dataset, M_CLASS_INDEX(i), M_CLASS_ICON_ID, IconImageId);
    }

}

void CMLClassCNN::AddClassToDataset(MIL_INT ClassIndex, const MIL_STRING& DataPath, 
    const MIL_STRING& ClassName, const MIL_STRING& AuthorName, MIL_ID Dataset)
{
    MclassControl(Dataset, M_DEFAULT, M_AUTHOR_ADD, AuthorName);
    MIL_INT NbEntries;
    MclassInquire(Dataset, M_DEFAULT, M_NUMBER_OF_ENTRIES + M_TYPE_MIL_INT, &NbEntries);
    MIL_STRING FolderName = DataPath + ClassName + MIL_TEXT("\\");
    std::vector<MIL_STRING> FilesInFolder;
    string strFolderName;
    m_AIParse->MIL_STRING2string(FolderName, strFolderName);
    m_AIParse->getFilesInFolder(strFolderName, "bmp", FilesInFolder);

    MIL_INT CurImageIndex = 0;

    for (const auto& File : FilesInFolder)
    {
        MclassControl(Dataset, M_DEFAULT, M_ENTRY_ADD, M_DEFAULT);
        MclassControlEntry(Dataset, NbEntries + CurImageIndex, M_DEFAULT_KEY, M_REGION_INDEX(0), M_CLASS_INDEX_GROUND_TRUTH, ClassIndex, M_NULL, M_DEFAULT);
        MclassControlEntry(Dataset, NbEntries + CurImageIndex, M_DEFAULT_KEY, M_DEFAULT, M_ENTRY_IMAGE_PATH, M_DEFAULT, File, M_DEFAULT);
        MclassControlEntry(Dataset, NbEntries + CurImageIndex, M_DEFAULT_KEY, M_REGION_INDEX(0), M_AUTHOR_NAME, M_DEFAULT, AuthorName, M_DEFAULT);
        CurImageIndex++;
    }

}

void CMLClassCNN::AddClassToDataset(MIL_INT ClassIndex, 
    const MIL_STRING& DataPath, 
    const MIL_STRING& ClassName, 
    const MIL_STRING& AuthorName, 
    MIL_UNIQUE_CLASS_ID& PartialDataset,
    MIL_UNIQUE_CLASS_ID& Dataset,
    MIL_DOUBLE dSampleRatio)
{

    MIL_INT NbEntries;
    MclassControl(Dataset, M_DEFAULT, M_AUTHOR_ADD, AuthorName);
    MclassInquire(Dataset, M_DEFAULT, M_NUMBER_OF_ENTRIES + M_TYPE_MIL_INT, &NbEntries);
    MIL_STRING FolderName = DataPath + ClassName + MIL_TEXT("\\");
    std::vector<MIL_STRING> FilesInFolder;
    string strFolderName;
    m_AIParse->MIL_STRING2string(FolderName, strFolderName);
    m_AIParse->getFilesInFolder(strFolderName, "bmp", FilesInFolder);

    
    int nFileNum = ceil(FilesInFolder.size() * dSampleRatio);
    random_shuffle(FilesInFolder.begin(), FilesInFolder.end());
    for (int CurImageIndex =0; CurImageIndex < FilesInFolder.size(); CurImageIndex++)
    {
        //选择比例
        auto& File = FilesInFolder[CurImageIndex];
        if (CurImageIndex< nFileNum) {
           
            MclassControl(PartialDataset, M_DEFAULT, M_ENTRY_ADD, M_DEFAULT);
            MclassControlEntry(PartialDataset, NbEntries + CurImageIndex, M_DEFAULT_KEY, M_REGION_INDEX(0), M_CLASS_INDEX_GROUND_TRUTH, ClassIndex, M_NULL, M_DEFAULT);
            MclassControlEntry(PartialDataset, NbEntries + CurImageIndex, M_DEFAULT_KEY, M_DEFAULT, M_ENTRY_IMAGE_PATH, M_DEFAULT, File, M_DEFAULT);
            MclassControlEntry(PartialDataset, NbEntries + CurImageIndex, M_DEFAULT_KEY, M_REGION_INDEX(0), M_AUTHOR_NAME, M_DEFAULT, AuthorName, M_DEFAULT);
        }
        //所有图片
        MclassControl(Dataset, M_DEFAULT, M_ENTRY_ADD, M_DEFAULT);
        MclassControlEntry(Dataset, NbEntries + CurImageIndex, M_DEFAULT_KEY, M_REGION_INDEX(0), M_CLASS_INDEX_GROUND_TRUTH, ClassIndex, M_NULL, M_DEFAULT);
        MclassControlEntry(Dataset, NbEntries + CurImageIndex, M_DEFAULT_KEY, M_DEFAULT, M_ENTRY_IMAGE_PATH, M_DEFAULT, File, M_DEFAULT);
        MclassControlEntry(Dataset, NbEntries + CurImageIndex, M_DEFAULT_KEY, M_REGION_INDEX(0), M_AUTHOR_NAME, M_DEFAULT, AuthorName, M_DEFAULT);
     
    }
}
