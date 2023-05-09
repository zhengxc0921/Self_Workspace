#include "MLDetCNN.h"



CMLDetCNN::CMLDetCNN(MIL_ID MilSystem, MIL_ID MilDisplay):
	m_MilSystem(MilSystem),
	m_MilDisplay(MilDisplay)
{
	m_AIParse = CAIParsePtr(new CAIParse(MilSystem));
}

CMLDetCNN::~CMLDetCNN()
{
}

MIL_INT CMLDetCNN::CnnTrainEngineDLLInstalled(MIL_ID MilSystem)
{
	MIL_INT IsInstalled = M_FALSE;
	MIL_UNIQUE_CLASS_ID TrainCtx = MclassAlloc(MilSystem, M_TRAIN_DET, M_DEFAULT, M_UNIQUE_ID);
	MclassInquire(TrainCtx, M_DEFAULT, M_TRAIN_ENGINE_IS_INSTALLED + M_TYPE_MIL_INT, &IsInstalled);
	return IsInstalled;
}

void CMLDetCNN::ConstructDataset(string ClassesInfo,
	//MIL_STRING IconDir, 
	string ImgDataInfo, 
	const MIL_STRING& WorkingDataPath)
{
	MIL_UNIQUE_CLASS_ID  Dataset = MclassAlloc(m_MilSystem, M_DATASET_IMAGES, M_DEFAULT, M_UNIQUE_ID);
	MclassControl(Dataset, M_DEFAULT, M_AUTHOR_ADD, MIL_TEXT("ZXC"));
	//step1:txt-->IconDataInfo
	vector<MIL_STRING>vecClasses;
	m_AIParse->readClasses2Vector(ClassesInfo, vecClasses);
	for (int i = 0; i < vecClasses.size(); i++) {
		//MIL_STRING ClassIcon = IconDir + vecClasses[i] + L".mim";
		MclassControl(Dataset, M_DEFAULT, M_CLASS_ADD, vecClasses[i]);
		//MIL_UNIQUE_BUF_ID IconImageId = MbufRestore(ClassIcon, m_MilSystem, M_UNIQUE_ID);
		//MclassControl(Dataset, M_CLASS_INDEX(i), M_CLASS_ICON_ID, IconImageId);
	}

	//step2:txt-->ImgDataInfo
	vector<MIL_STRING> vecImgPaths;
	vector<vector<Box>> vec2Boxes;
	vector<vector<int>> veclabels;
	m_AIParse->readDataSet2Vector(ImgDataInfo, vecImgPaths, vec2Boxes, veclabels);
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


	CreateFolder(WorkingDataPath);
	MclassControl(Dataset, M_CONTEXT, M_CONSOLIDATE_ENTRIES_INTO_FOLDER, WorkingDataPath);
	MIL_STRING WorkDatasetPath = WorkingDataPath + MIL_TEXT("DataSet.mclassd");
	MclassSave(WorkDatasetPath, Dataset, M_DEFAULT);
}

void CMLDetCNN::ConstructDataset(string ClassesInfo,
    string IconDir,
    string ImgDataInfo,
    string WorkingDataPath,
    string DataSetName)
{

    MIL_STRING MStrIconDir = m_AIParse->string2MIL_STRING(IconDir);
    MIL_STRING MStrWorkingDataPath = m_AIParse->string2MIL_STRING(WorkingDataPath);
    MIL_STRING MStrDataSetName = m_AIParse->string2MIL_STRING(DataSetName);
    //m_AIParse->MIL_STRING2string(MStrIconDir, IconDir);
    //m_AIParse->MIL_STRING2string(MStrWorkingDataPath, WorkingDataPath);


    MIL_UNIQUE_CLASS_ID  Dataset = MclassAlloc(m_MilSystem, M_DATASET_IMAGES, M_DEFAULT, M_UNIQUE_ID);
    MclassControl(Dataset, M_DEFAULT, M_AUTHOR_ADD, MIL_TEXT("ZXC"));
    //step1:txt-->IconDataInfo
    vector<MIL_STRING>vecClasses;
    m_AIParse->readClasses2Vector(ClassesInfo, vecClasses);
    for (int i = 0; i < vecClasses.size(); i++) {
        MIL_STRING ClassIcon = MStrIconDir + vecClasses[i] + L".bmp";
        MclassControl(Dataset, M_DEFAULT, M_CLASS_ADD, vecClasses[i]);
        MIL_UNIQUE_BUF_ID IconImageId = MbufRestore(ClassIcon, m_MilSystem, M_UNIQUE_ID);
        MclassControl(Dataset, M_CLASS_INDEX(i), M_CLASS_ICON_ID, IconImageId);
    }

    //step2:txt-->ImgDataInfo
    vector<MIL_STRING> vecImgPaths;
    vector<vector<Box>> vec2Boxes;
    vector<vector<int>> veclabels;
    m_AIParse->readDataSet2Vector(ImgDataInfo, vecImgPaths, vec2Boxes, veclabels);
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


    CreateFolder(MStrWorkingDataPath);
    MclassControl(Dataset, M_CONTEXT, M_CONSOLIDATE_ENTRIES_INTO_FOLDER, MStrWorkingDataPath);
    MIL_STRING WorkDatasetPath = MStrWorkingDataPath + MStrDataSetName;
    MclassSave(WorkDatasetPath, Dataset, M_DEFAULT);
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

void CMLDetCNN::PrepareDataset(MIL_UNIQUE_CLASS_ID& DatasetContext, MIL_UNIQUE_CLASS_ID& PrepareDataset, MIL_UNIQUE_CLASS_ID& PreparedDataset)
{
    MclassPreprocess(DatasetContext, M_DEFAULT);
    MclassPrepareData(DatasetContext, PrepareDataset, PreparedDataset, M_NULL, M_DEFAULT);

    MIL_STRING PreparedDatasetPath =  MIL_TEXT("I:/MIL_Detection_Dataset/lslm_all/PreparedDataSet.mclassd");
    MclassSave(PreparedDatasetPath, PreparedDataset, M_DEFAULT);
    MclassExport(MIL_TEXT("I:/MIL_Detection_Dataset/lslm_all/TrainDatasetFeatures.csv"), M_FORMAT_CSV, PreparedDataset, M_DEFAULT, M_ENTRIES, M_DEFAULT);
  
}

void CMLDetCNN::ConstructTrainCtx(DetParas ClassifierParas, MIL_UNIQUE_CLASS_ID& TrainCtx)
{
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
    MIL_UNIQUE_CLASS_ID& DatasetContext,
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

void CMLDetCNN::PredictBegin(MIL_UNIQUE_CLASS_ID& TrainedCCtx, MIL_ID Image)
{

    //获取模型输入尺寸
    MclassInquire(TrainedCCtx, M_DEFAULT_SOURCE_LAYER, M_SIZE_X + M_TYPE_MIL_INT, &m_InputSizeX);
    MclassInquire(TrainedCCtx, M_DEFAULT_SOURCE_LAYER, M_SIZE_Y + M_TYPE_MIL_INT, &m_InputSizeY);
    MclassInquire(TrainedCCtx, M_CONTEXT, M_NUMBER_OF_CLASSES + M_TYPE_MIL_INT, &m_ClassesNum);
    MIL_INT m_ImageSizeX = MbufInquire(Image, M_SIZE_X, M_NULL);
    MIL_INT m_ImageSizeY = MbufInquire(Image, M_SIZE_Y, M_NULL);

}

void CMLDetCNN::Predict(MIL_ID Image, MIL_UNIQUE_CLASS_ID& TrainedDetCtx, DetResult& Result)
{
    PredictBegin(TrainedDetCtx, Image);
   
    //MIL_ID ImageReduce = MbufAllocColor(m_MilSystem, 3, m_InputSizeX, m_InputSizeY, 8 + M_UNSIGNED, M_IMAGE + M_PROC , M_NULL);
    MIL_ID ImageReduce = MbufAlloc2d(m_MilSystem, m_InputSizeX, m_InputSizeY, 8 + M_UNSIGNED, M_IMAGE + M_PROC, M_NULL);

    MimResize(Image, ImageReduce, M_FILL_DESTINATION, M_FILL_DESTINATION, M_BILINEAR);

    MIL_INT Status = M_FALSE;
    MclassInquire(TrainedDetCtx, M_DEFAULT, M_PREPROCESSED + M_TYPE_MIL_INT, &Status);
    if (M_FALSE == Status)
    {
        MclassPreprocess(TrainedDetCtx, M_DEFAULT);
    }
    MIL_UNIQUE_CLASS_ID ClassRes = MclassAllocResult(m_MilSystem, M_PREDICT_DET_RESULT, M_DEFAULT, M_UNIQUE_ID);
    
    LARGE_INTEGER t1, t2, tc;
    QueryPerformanceFrequency(&tc);
    QueryPerformanceCounter(&t1);
    int CN = 1;
    for (int i = 0; i < CN; i++) {
        MclassPredict(TrainedDetCtx, ImageReduce, ClassRes, M_DEFAULT);
    }
    QueryPerformanceCounter(&t2);
    double time = (double)(t2.QuadPart - t1.QuadPart) / (double)tc.QuadPart/(double)CN;
    cout << "\nMclassPredict_" << "time is = " << time  << endl;
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
       
        MclassGetResult(ClassRes, M_INSTANCE_INDEX(i), M_BEST_CLASS_INDEX + M_TYPE_MIL_INT, &Result.ClassIndex[i]);
        MclassGetResult(ClassRes, M_INSTANCE_INDEX(i), M_BEST_CLASS_SCORE + M_TYPE_MIL_DOUBLE, &Result.Score[i]);
        MclassInquire(TrainedDetCtx, M_CLASS_INDEX(Result.ClassIndex[i]), M_CLASS_NAME, Result.ClassName[i]);
    }
}

void CMLDetCNN::FolderImgsPredict(vector<MIL_STRING> FilesInFolder,
    MIL_UNIQUE_CLASS_ID& TrainedDetCtx,
    vector<DetResult>& Result)
{
    int nFileNum = FilesInFolder.size();
    DetResult Result_i;
    for (int i = 0; i < nFileNum; i++) {
        memset(&Result_i, 0, sizeof(Result_i));
        MIL_ID RawImage = MbufRestore(FilesInFolder[i], m_MilSystem, M_NULL);
        Predict(RawImage, TrainedDetCtx, Result_i);
        Result.emplace_back(Result_i);
        MbufFree(RawImage);
    }
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