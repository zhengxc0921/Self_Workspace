#include "Test.h"

Test::Test(MIL_ID MilSystem, MIL_ID MilDisplay, MIL_STRING strProject) :
	m_MilSystem(MilSystem),
	m_MilDisplay(MilDisplay),
	m_strProject(strProject)
{
	//m_MLClassCNN = CMLClassCNNPtr(new CMLClassCNN(MilSystem, MilDisplay));
	m_MLDetCNN = CMLDetCNNPtr(new CMLDetCNN(MilSystem, MilDisplay));
}

Test::~Test()
{

}

void Test::getIcon(vector<MIL_STRING> OriginalDataPath,
	vector<vector<MIL_STRING>> ClassName,
	vector<vector<MIL_STRING>>& ClassIcon)
{
	for (int f_n = 0; f_n < OriginalDataPath.size(); f_n++) {
		MIL_STRING tmp_OriginalDataPath = OriginalDataPath[f_n];
		vector<MIL_STRING> tmp_ClassName = ClassName[f_n];
		vector<MIL_STRING> tmp_ClassIcons;
		MIL_STRING tmp_ClassIcon;
		for (int c_n = 0; c_n < ClassName[f_n].size(); c_n++) {
			tmp_ClassIcon = tmp_OriginalDataPath + tmp_ClassName[c_n] + MIL_TEXT(".mim");
			tmp_ClassIcons.push_back(tmp_ClassIcon);
		}
		ClassIcon.emplace_back(tmp_ClassIcons);
	}
}

void Test::getModelInfo(MIL_UNIQUE_CLASS_ID& Model)
{
	MclassInquire(Model, M_CONTEXT, M_NUMBER_OF_CLASSES + M_TYPE_MIL_INT, &m_ClassesNum);
	m_ClassNames.resize(m_ClassesNum);
	MIL_STRING ClassName;
	for (int i = 0; i < m_ClassesNum; i++) {
		MclassInquire(Model, M_CLASS_INDEX(i), M_CLASS_NAME, ClassName);
		m_ClassNames[i] = ClassName;
	}
	MclassInquire(Model, M_DEFAULT_SOURCE_LAYER, M_SIZE_X + M_TYPE_MIL_INT, &m_InputSizeX);
	MclassInquire(Model, M_DEFAULT_SOURCE_LAYER, M_SIZE_Y + M_TYPE_MIL_INT, &m_InputSizeY);

}

//void Test::savePredictedImg()
//{
//	//创建目的class的文件夹
//	string strDstDir;
//	m_MLClassCNN->m_AIParse->MIL_STRING2string(m_DstImgDir, strDstDir);
//	m_MLClassCNN->CreateFolder(m_DstImgDir);
//
//	for (std::vector<MIL_STRING>::iterator it = m_ClassNames.begin(); it != m_ClassNames.end(); ++it) {
//		m_MLClassCNN->CreateFolder(m_DstImgDir + (*it));
//	}
//	//遍历所有文件及结果，并将图片保存到相应的Index
//	MIL_INT Img_index = 0;
//	for (std::vector<MIL_STRING>::iterator it = m_FilesInFolder.begin(); it != m_FilesInFolder.end(); it++) {
//		MIL_STRING RawImagePath = (*it);
//		MIL_ID Image = MbufRestore(RawImagePath, m_MilSystem, M_NULL);
//		string::size_type iPos = RawImagePath.find_last_of('/') + 1;
//		MIL_STRING ImageRawName = RawImagePath.substr(iPos, RawImagePath.length() - iPos);
//		MIL_STRING DstRootPath = m_DstImgDir + m_ClassNames[m_vecResults[Img_index].PredictClass] + MIL_TEXT("//") + ImageRawName;
//		MbufExport(DstRootPath, M_BMP, Image);
//		MbufFree(Image);
//		Img_index++;
//	}
//}

//void MILTest::predictBegin()
//{
//
//	//生成模型的参数
//	string strTdDetCtxName = "G:/DefectDataCenter/ParseData/Detection/" + m_strProject + "/MIL_Data/PreparedData/" + m_strProject + ".mclass";
//	MIL_STRING TdDetCtxName = m_MLDetCNN->m_AIParse->string2MIL_STRING(strTdDetCtxName);
//	m_TrainedCtx = MclassRestore(TdDetCtxName, m_MilSystem, M_DEFAULT, M_UNIQUE_ID);
//	MclassInquire(m_TrainedCtx, M_DEFAULT_SOURCE_LAYER, M_SIZE_X + M_TYPE_MIL_INT, &m_InputSizeX);
//	MclassInquire(m_TrainedCtx, M_DEFAULT_SOURCE_LAYER, M_SIZE_Y + M_TYPE_MIL_INT, &m_InputSizeY);
//	MclassInquire(m_TrainedCtx, M_CONTEXT, M_NUMBER_OF_CLASSES + M_TYPE_MIL_INT, &m_ClassesNum);
//	//设置ENGINE
//	MIL_INT engine_index = 2;
//	MIL_STRING Description;
//	MclassControl(m_TrainedCtx, M_DEFAULT, M_PREDICT_ENGINE, engine_index);
//	MclassInquire(m_TrainedCtx, M_PREDICT_ENGINE_INDEX(engine_index), M_PREDICT_ENGINE_DESCRIPTION, Description);
//	MosPrintf(MIL_TEXT("\nM_PREDICT_ENGINE_DESCRIPTION: %s \n"), Description.c_str());
//	//生成待测图片的参数
//
//	string FileType = "bmp";
//	vector<MIL_STRING>Files;
//	string	SrcDir = "G:/DefectDataCenter/ParseData/Detection/" + m_strProject + "/raw_data/";
//	string	SrcImgDir = SrcDir + "MutiThread_TImg";
//	m_MLDetCNN->m_AIParse->getFilesInFolder(SrcImgDir, FileType, m_FilesInFolder);
//	for (int i = 0; i < m_FilesInFolder.size(); i++) {
//		MIL_ID RawImage = MbufRestore(m_FilesInFolder[i], m_MilSystem, M_NULL);
//		string RawImgPath;
//		m_MLDetCNN->m_AIParse->MIL_STRING2string(m_FilesInFolder[i], RawImgPath);
//		m_PathRawImageMap.insert(pair<string, MIL_ID>(RawImgPath, RawImage));
//	}
//}

void Test::InitClassWeights()
{
	m_ClassWeights.resize(m_ClassesNum, 1.0);
}

//void Test::CropImgs()
//{
//	int CropWH = 64;
//	string SrcImgDir = "G:/DefectDataCenter/李巨欣/ShenZhen/Lot_Gather/ALL";
//	string DstImgDir = "G:/DefectDataCenter/李巨欣/ShenZhen/Lot_Gather/ALL_Crop";
//
//	MIL_STRING MDstImgDir = m_MLClassCNN->m_AIParse->string2MIL_STRING(DstImgDir);
//	m_MLClassCNN->CreateFolder(MDstImgDir);
//	vector<string>folders;
//	m_MLClassCNN->m_AIParse->getFoldersInFolder(SrcImgDir, folders);
//	if (folders.size() > 0) {
//		for (int i = 0; i < folders.size(); i++) {
//			string SrcSubDir = SrcImgDir + "//" + folders[i];
//			string DstSubDir = DstImgDir + "//" + folders[i];
//
//			MIL_STRING MDstSubDir = m_MLClassCNN->m_AIParse->string2MIL_STRING(DstSubDir);
//			m_MLClassCNN->CreateFolder(MDstSubDir);
//			vector<string>FilesInFolder;
//			m_MLClassCNN->m_AIParse->getFilesInFolder(SrcSubDir, "bmp", FilesInFolder);
//			for (int j = 0; j < FilesInFolder.size(); j++) {
//				MIL_ID ImgOut;
//				string srcImgPath = FilesInFolder[j];
//
//				vector<string>ListFilesInFolder;
//				m_MLClassCNN->m_AIParse->Split(FilesInFolder[j], ListFilesInFolder, "//");
//				string dstImgPath = DstSubDir + "//" + ListFilesInFolder.back();
//
//				MIL_STRING MsrcImgPath = m_MLClassCNN->m_AIParse->string2MIL_STRING(srcImgPath);
//				MIL_STRING MdstImgPath = m_MLClassCNN->m_AIParse->string2MIL_STRING(dstImgPath);
//				MIL_ID ImgIn = MbufRestore(MsrcImgPath, m_MilSystem, M_NULL);
//				m_MLClassCNN->m_AIParse->ImgCenterCrop(ImgIn, CropWH, ImgOut);
//				MbufExport(MdstImgPath, M_BMP, ImgOut);
//				MbufFree(ImgOut);
//			}
//		}
//	}
//	else
//	{
//		vector<string>FilesInFolder;
//		m_MLClassCNN->m_AIParse->getFilesInFolder(SrcImgDir, "bmp", FilesInFolder);
//		for (int j = 0; j < FilesInFolder.size(); j++) {
//			MIL_ID ImgOut;
//			vector<string>ListFilesInFolder;
//			m_MLClassCNN->m_AIParse->Split(FilesInFolder[j], ListFilesInFolder, "//");
//			string dstImgPath = DstImgDir + "//" + ListFilesInFolder.back();
//			MIL_STRING MsrcImgPath = m_MLClassCNN->m_AIParse->string2MIL_STRING(FilesInFolder[j]);
//			MIL_STRING MdstImgPath = m_MLClassCNN->m_AIParse->string2MIL_STRING(dstImgPath);
//			MIL_ID ImgIn = MbufRestore(MsrcImgPath, m_MilSystem, M_NULL);
//			m_MLClassCNN->m_AIParse->ImgCenterCrop(ImgIn, CropWH, ImgOut);
//			MbufExport(MdstImgPath, M_BMP, ImgOut);
//			MbufFree(ImgOut);
//		}
//	}
//}

void Test::FillImgs()
{
	MIL_ID RawImage = MbufRestore(L"G:/DefectDataCenter/TImg/C.jpg", m_MilSystem, M_NULL);
	MIL_STRING DstRootPath = L"G:/DefectDataCenter/TImg/C_Crop.jpg";
	const MIL_INT SizeX = MbufInquire(RawImage, M_SIZE_X, M_NULL);
	const MIL_INT SizeY = MbufInquire(RawImage, M_SIZE_Y, M_NULL);
	const MIL_INT SizeBAND = MbufInquire(RawImage, M_SIZE_BAND, M_NULL);
	MIL_INT SImgW = max(SizeX, SizeY);
	MIL_INT SImgH = max(SizeX, SizeY);
	if (SizeBAND == 1) {

		MIL_ID MonoImage = MbufAlloc2d(m_MilSystem, SImgW, SImgH, 8 + M_UNSIGNED, M_IMAGE + M_PROC, M_NULL);
		MIL_UINT8 Value1 = 0;
		std::unique_ptr<BYTE[]> ScaledImage = std::make_unique<BYTE[]>(SImgW * SImgH);
		if (SizeY >= SizeX) {
			for (int y = 0; y < SImgH; ++y)
			{
				MbufGet2d(RawImage, 0, y, 1, 1, &Value1);
				for (int x = 0; x < SImgW; ++x)
				{
					MIL_INT DstIndex = y * SImgW + x;
					ScaledImage[DstIndex] = Value1;
				}
			}
			MbufPut2d(MonoImage, 0, 0, SImgW, SImgH, ScaledImage.get());
			MbufCopyColor2d(RawImage, MonoImage, M_ALL_BANDS, 0, 0, M_ALL_BANDS, SizeY - SizeX, 0, SizeX, SizeY);
		}
		else {
			for (int x = 0; x < SImgW; ++x)
			{
				MbufGet2d(RawImage, x, 0, 1, 1, &Value1);
				for (int y = 0; y < SImgH; ++y)
				{
					MIL_INT DstIndex = y * SImgW + x;
					ScaledImage[DstIndex] = Value1;
				}
			}
			MbufPut2d(MonoImage, 0, 0, SImgW, SImgH, ScaledImage.get());
			MbufCopyColor2d(RawImage, MonoImage, M_ALL_BANDS, 0, 0, M_ALL_BANDS, 0, SizeX - SizeY, SizeX, SizeY);
		}

		MbufExport(DstRootPath, M_BMP, MonoImage);
	}
	else if (SizeBAND == 3) {

		MIL_ID MonoImage = MbufAllocColor(m_MilSystem, SizeBAND, SImgW, SImgH, 8 + M_UNSIGNED, M_IMAGE + M_PROC, M_NULL);
		MIL_UINT8 Value1 = 0;
		std::unique_ptr<BYTE[]> ScaledImage = std::make_unique<BYTE[]>(SizeBAND * SImgW * SImgH);
		for (int b = 0; b < SizeBAND; b++) {
			for (int y = 0; y < SImgH; ++y)
			{
				MbufGetColor2d(RawImage, M_SINGLE_BAND, b, 0, y, 1, 1, &Value1);

				cout << "SizeBAND:" << SizeBAND << "Value1" << Value1 << endl;
				for (int x = 0; x < SImgW; ++x)
				{
					MIL_INT DstIndex = b * SImgH * SImgW + y * SImgW + x;
					ScaledImage[DstIndex] = Value1;
				}
			}
		}

		MbufPutColor2d(MonoImage, M_PACKED + M_BGR24, M_ALL_BANDS, 0, 0, SImgW, SImgH, ScaledImage.get());
		MbufCopyColor2d(RawImage, MonoImage, M_ALL_BANDS, 0, 0, M_ALL_BANDS, SizeY - SizeX, 0, SizeX, SizeY);
		MbufExport(DstRootPath, M_BMP, MonoImage);
	}

}

//void Test::MILTestGenDataset()
//{
//	MIL_STRING AuthorName = MIL_TEXT("AA");
//	MIL_STRING OriginalDataPath = m_ClassifierSrcDataDir + m_strProject + L"//";
//	MIL_STRING WorkingDataPath = m_ClassifierWorkSpace + m_strProject + L"//";
//
//	vector<MIL_STRING>ClassName = { MIL_TEXT("2"),MIL_TEXT("11"),MIL_TEXT("12") ,
//		MIL_TEXT("30"),MIL_TEXT("45"),MIL_TEXT("90") };
//	vector<MIL_STRING > ClassIcon;
//	for (int i = 0; i < ClassName.size(); i++) {
//		ClassIcon.emplace_back(m_ClassifierSrcDataDir + m_strProject + L"//" + ClassName[i] + L".mim");
//	}
//	MIL_UNIQUE_CLASS_ID  Dataset = MclassAlloc(m_MilSystem, M_DATASET_IMAGES, M_DEFAULT, M_UNIQUE_ID);
//	m_MLClassCNN->ConstructDataset(ClassName, ClassIcon, AuthorName, OriginalDataPath, WorkingDataPath, Dataset);
//	m_MLClassCNN->CreateFolder(WorkingDataPath);
//	MclassControl(Dataset, M_CONTEXT, M_CONSOLIDATE_ENTRIES_INTO_FOLDER, WorkingDataPath);
//
//	////*******************************必须参数*******************************//
//	MIL_UNIQUE_CLASS_ID DataContext = MclassAlloc(m_MilSystem, M_PREPARE_IMAGES_CNN, M_DEFAULT, M_UNIQUE_ID);
//	DataContextParasStruct DataCtxParas;
//	MIL_DOUBLE TestDatasetPercentage = 10;
//	DataCtxParas.ImageSizeX = 128;
//	DataCtxParas.ImageSizeY = 128;
//	DataCtxParas.PreparedDataFolder = WorkingDataPath;
//	memset(&DataCtxParas.AugParas, 0, sizeof(AugmentationParasStruct));
//	DataCtxParas.AugParas.AugmentationNumPerImage = 0;
//	DataCtxParas.ResizeModel = 1;
//	DataCtxParas.AugParas.ScaleFactorMax = 1.03; //1.03
//	DataCtxParas.AugParas.ScaleFactorMin = 0.97; //0.97
//	DataCtxParas.AugParas.RotateAngleDelta = 20; //10
//	DataCtxParas.AugParas.IntyDeltaAdd = 32;  //32
//	DataCtxParas.AugParas.DirIntyMax = 1.2; //1.2
//	DataCtxParas.AugParas.DirIntyMin = 0.8; //0.8
//	DataCtxParas.AugParas.SmoothnessMax = 50; //50 {0<x<100}
//	DataCtxParas.AugParas.SmoothnessMin = 0.5; //0.5 {0<x<100}
//	DataCtxParas.AugParas.GaussNoiseStdev = 25; //25
//	DataCtxParas.AugParas.GaussNoiseDelta = 25; //25
//
//	m_MLClassCNN->ConstructDataContext(DataCtxParas, DataContext);
//	//MIL_STRING WorkingDataPath = SrcImgDir + MIL_TEXT("DataSet\\DataSet.mclassd");				//原始数据根文件下的 存放中间数据的文件夹
//	//MIL_UNIQUE_CLASS_ID WorkingDataset = MclassRestore(WorkingDataPath, m_MilSystem, M_DEFAULT, M_UNIQUE_ID);
//	MIL_UNIQUE_CLASS_ID PreparedDataset = MclassAlloc(m_MilSystem, M_DATASET_IMAGES, M_DEFAULT, M_UNIQUE_ID);
//	m_MLClassCNN->PrepareDataset(DataContext, Dataset, PreparedDataset, WorkingDataPath, TestDatasetPercentage);
//
//}
//void Test::MILTestTrain()
//{
//
//	int MaxNumberOfEpoch = 25;			//模型训练次数
//	int MiniBatchSize = 64;				//模型训练单次迭代的张数
//	//////*******************************必须参数*******************************//
//	MIL_STRING PreparedPath = m_ClassifierWorkSpace + m_strProject + MIL_TEXT("/WorkingDataset.mclassd");
//	MIL_UNIQUE_CLASS_ID PreparedDataset = MclassRestore(PreparedPath, m_MilSystem, M_DEFAULT, M_UNIQUE_ID);
//	MIL_UNIQUE_CLASS_ID TrainCtx = MclassAlloc(m_MilSystem, M_TRAIN_CNN, M_DEFAULT, M_UNIQUE_ID);
//	ClassifierParasStruct ClassifierParas;
//	ClassifierParas.TrainMode = 0;
//	ClassifierParas.TrainEngineUsed = 0;
//	ClassifierParas.MaxNumberOfEpoch = MaxNumberOfEpoch;
//	ClassifierParas.MiniBatchSize = MiniBatchSize;
//	ClassifierParas.SchedulerType = 0;
//	ClassifierParas.LearningRate = 0.0001;
//	ClassifierParas.LearningRateDecay = 0;
//	ClassifierParas.SplitPercent = 90.0;
//	ClassifierParas.TrainDstFolder = m_ClassifierWorkSpace + L"//" + m_strProject + MIL_TEXT("/PreparedData/");
//	m_MLClassCNN->ConstructTrainCtx(ClassifierParas, TrainCtx);
//	MIL_UNIQUE_CLASS_ID TrainedClassifierCtx;
//	MIL_UNIQUE_CLASS_ID PrevClassifierCtx;
//	MIL_STRING ClassifierDumpFile = ClassifierParas.TrainDstFolder + m_strProject + L".mclass";
//	m_MLClassCNN->TrainClassifier(PreparedDataset, TrainCtx, PrevClassifierCtx, TrainedClassifierCtx, ClassifierDumpFile);
//	return;
//}
//
//void Test::MILTestPredict() {
//
//	MIL_STRING PreClassifierName = m_ClassifierWorkSpace + m_strProject + MIL_TEXT("/PreparedData/") + m_strProject + L".mclass";
//
//	string strProj;
//	m_MLClassCNN->m_AIParse->MIL_STRING2string(m_strProject, strProj);
//	string	SrcImgDir = "G:/DefectDataCenter/原始_现场分类数据/LJX/TestData/" + strProj;
//	m_DstImgDir = MIL_TEXT("G:/DefectDataCenter/原始_现场分类数据/LJX/TestResult/") + m_strProject + L"//";
//	LARGE_INTEGER t1, t2, tc;
//	QueryPerformanceFrequency(&tc);
//	QueryPerformanceCounter(&t1);
//	MIL_UNIQUE_CLASS_ID TestCtx = MclassRestore(PreClassifierName, m_MilSystem, M_DEFAULT, M_UNIQUE_ID);
//	getModelInfo(TestCtx);
//	//设置ENGINE
//	MIL_STRING Description;
//	MclassInquire(TestCtx, M_PREDICT_ENGINE_INDEX(0), M_PREDICT_ENGINE_DESCRIPTION, Description);
//	MosPrintf(MIL_TEXT("\nM_PREDICT_ENGINE_DESCRIPTION: %s \n"), Description.c_str());
//
//	m_MLClassCNN->m_AIParse->getFilesInFolder(SrcImgDir, "bmp", m_FilesInFolder);
//	int nFileNum = m_FilesInFolder.size();
//	vector<MIL_DOUBLE> ClassWeights(m_ClassesNum, 1.0);
//	m_MLClassCNN->FolderImgsPredict(m_FilesInFolder, ClassWeights, TestCtx, m_vecResults);
//
//	QueryPerformanceCounter(&t2);
//	double calc_time = (double)(t2.QuadPart - t1.QuadPart) / (double)tc.QuadPart;
//	if (m_SavePredictedImg) {
//		savePredictedImg();
//	}
//}

//void Test::MILTestPredictAP()
//{
//	MIL_STRING PreClassifierName = m_ClassifierWorkSpace + m_strProject + MIL_TEXT("/PreparedData/") + m_strProject + L".mclass";
//	string strProj;
//	m_MLClassCNN->m_AIParse->MIL_STRING2string(m_strProject, strProj);
//	string	SrcImgDir = "G:/DefectDataCenter/原始_现场分类数据/LJX/TestData/" + strProj;
//
//	map<string, vector<MIL_STRING>> mapLabelFiles;
//	m_MLClassCNN->m_AIParse->getLabelFilesInFolder(SrcImgDir, "",
//		"bmp", mapLabelFiles);
//
//	m_DstImgDir = MIL_TEXT("G:/DefectDataCenter/原始_现场分类数据/LJX/TestResult/") + m_strProject + L"//";
//	LARGE_INTEGER t1, t2, tc;
//	QueryPerformanceFrequency(&tc);
//	QueryPerformanceCounter(&t1);
//	MIL_UNIQUE_CLASS_ID TestCtx = MclassRestore(PreClassifierName, m_MilSystem, M_DEFAULT, M_UNIQUE_ID);
//	getModelInfo(TestCtx);
//	//设置ENGINE
//	MIL_STRING Description;
//	MclassInquire(TestCtx, M_PREDICT_ENGINE_INDEX(0), M_PREDICT_ENGINE_DESCRIPTION, Description);
//	MosPrintf(MIL_TEXT("\nM_PREDICT_ENGINE_DESCRIPTION: %s \n"), Description.c_str());
//	map<string, float>mapPrecdictAP;
//	for (auto iter = mapLabelFiles.begin(); iter != mapLabelFiles.end(); iter++) {
//		string GtLabel = (*iter).first;
//		MIL_STRING MGtLabel = m_MLClassCNN->m_AIParse->string2MIL_STRING(GtLabel);
//		int GtLabelNum = (*iter).second.size();
//
//		vector<MIL_DOUBLE> ClassWeights(m_ClassesNum, 1.0);
//		m_MLClassCNN->FolderImgsPredict((*iter).second, ClassWeights, TestCtx, m_vecResults);
//		int k = 0;
//		for (int i = 0; i < GtLabelNum; i++) {
//			if (MGtLabel == m_vecResults[i].PredictClassName) {
//				k++;
//			};
//		}
//		float precision = float(k) / float(GtLabelNum);
//		mapPrecdictAP.insert(pair<string, float>(GtLabel, precision));
//		if (m_SavePredictedImg) {
//			savePredictedImg();
//		}
//	}
//}

//void Test::MILTestPredictWithBlob()
//{
//	//MIL_ID RawIma = MbufRestore(L"D:/LeetCode/Img/AugImg/L1C14 (2).bmp", m_MilSystem, M_NULL);
//	string BlobTxt = "D:/LeetCode/Img/BlobTxt";
//	string AugImg = "D:/LeetCode/Img/AugImg";
//	string CropedImg = "D:/LeetCode/Img/CropImg";
//	vector<string> Files;
//	m_MLClassCNN->m_AIParse->getFilesInFolder(BlobTxt, "txt", Files);
//
//	for (auto& file : Files) {
//		//读取blob数据
//		vector<int> blob_px;
//		vector<int> blob_py;
//		m_MLClassCNN->m_AIParse->readBlob2Vector(file, blob_px, blob_py);
//		//读取img
//		auto pos1 = file.find_last_of("//") + 1;
//		auto pos2 = file.find_last_of(".");
//		auto count = pos2 - pos1;
//		string img_name = file.substr(pos1, count);
//		cout << "img_name: " << img_name << endl;
//		vector<MIL_ID> ClipImgs;
//		m_MLClassCNN->m_AIParse->blobClip(AugImg, img_name, blob_px, blob_py, ClipImgs, CropedImg);
//	}
//}

//void Test::MILTestPredictEngine()
//{
//	string SrcImgDir = "G:/DefectDataCenter/lianglichuang/SXX_big_defect/SXX_big_defect_0";
//	m_DstImgDir = MIL_TEXT("G:/DefectDataCenter/lianglichuang/SXX_big_defect/SXX_big_defect_0_R/");
//	MIL_STRING PreClassifierName = MIL_TEXT("G:/DefectDataCenter/zhjuzhiqiang_2023/2023/SXX/PreparedData/SXX.mclass");
//	MIL_UNIQUE_CLASS_ID TestCtx = MclassRestore(PreClassifierName, m_MilSystem, M_DEFAULT, M_UNIQUE_ID);
//	getModelInfo(TestCtx);
//	MIL_INT NbPredEngines = 0;
//	MclassInquire(TestCtx, M_DEFAULT, M_NUMBER_OF_PREDICT_ENGINES + M_TYPE_MIL_INT, &NbPredEngines);
//	for (int engine_index = 0; engine_index < NbPredEngines; engine_index++) {
//		LARGE_INTEGER t1, t2, tc;
//		QueryPerformanceFrequency(&tc);
//		QueryPerformanceCounter(&t1);
//		//设置ENGINE
//		MIL_STRING Description;
//		MclassControl(TestCtx, M_DEFAULT, M_PREDICT_ENGINE, engine_index);
//		MclassInquire(TestCtx, M_PREDICT_ENGINE_INDEX(engine_index), M_PREDICT_ENGINE_DESCRIPTION, Description);
//		MosPrintf(MIL_TEXT("\nM_PREDICT_ENGINE_DESCRIPTION: %s \n"), Description.c_str());
//
//		m_MLDetCNN->m_AIParse->getFilesInFolder(SrcImgDir, "bmp", m_FilesInFolder);
//		int nFileNum = m_FilesInFolder.size();
//		vector<MIL_DOUBLE> ClassWeights(m_ClassesNum, 1.0);
//		m_MLClassCNN->FolderImgsPredict(m_FilesInFolder, ClassWeights, TestCtx, m_vecResults);
//		QueryPerformanceCounter(&t2);
//		double calc_time = (double)(t2.QuadPart - t1.QuadPart) / (double)tc.QuadPart;
//		MosPrintf(MIL_TEXT("\nM_PREDICT_ENGINE_DESCRIPTION: %s  calc time is %f\n"), Description.c_str(), calc_time);
//	}
//
//}

//void MILTest::MILTestPredictShareMem(
//	string strShareMame,
//	string index,
//	size_t filesize) {
//
//	//std::cout << "Infunc strShareMame is  " << strShareMame<< std::endl;
//	//std::cout << "Infunc index is  " << index << std::endl;
//	//std::cout << "Infunc filesize is  " << filesize << std::endl;
//
//	////读取共享内存到模型
//	auto m_hMap = ::OpenFileMappingA(FILE_MAP_READ, FALSE, strShareMame.c_str());
//	if (m_hMap == NULL) { std::cout << "m_hMap is Null " << std::endl; return; }
//	DWORD nSize = filesize + sizeof(ULONGLONG);
//	BYTE* pShareBuf = (BYTE*)MapViewOfFile(m_hMap, FILE_MAP_READ, 0, 0, nSize);
//	auto model_Buff = (MIL_UINT8*)(pShareBuf + sizeof(ULONGLONG));
//	MIL_UNIQUE_CLASS_ID TrainedCtx;
//	MclassStream(model_Buff, m_MilSystem, M_RESTORE, M_MEMORY, M_DEFAULT, M_DEFAULT, &TrainedCtx, M_NULL);
//	//读取模型引擎
//	MIL_STRING Description{};
//	MIL_STRING* pDescription = &Description;
//	MclassInquire(TrainedCtx, M_PREDICT_ENGINE_INDEX(0), M_PREDICT_ENGINE_DESCRIPTION, *pDescription);
//	MosPrintf(MIL_TEXT("\nEngine of Process_%s is %s\n"), index.c_str(), (*pDescription).c_str());
//
//	//图片来源、保存路径
//	string strSrcImgDir = "G:/DefectDataCenter/lianglichuang/SXX_big_defect/SXX_big_defect_" + index;
//	string strDstImgDir = strSrcImgDir + "_ShareMem_result";
//
//
//	m_DstImgDir = m_MLDetCNN->m_AIParse->string2MIL_STRING(strDstImgDir);
//
//	//设置预测引擎
//	MclassControl(TrainedCtx, M_DEFAULT, M_PREDICT_ENGINE, 2);
//	double calc_time = 0;
//	MILTestPredictCore(strSrcImgDir, TrainedCtx, calc_time);
//	cout << "\nOpen Process_" << index << " calc_time time(s) is = " << calc_time << "\n" << endl; //输出时间（单位：ｓ）
//}

//void MILTest::MILTestGenDetDataset()
//{
//	////CASE1:
//	//string IconInfo = "I:/MIL_Detection_Dataset/DSW/classes.txt";
//	//MIL_STRING IconDir = L"I:/MIL_Detection_Dataset/DSW/Dataset/Icons/";
//	//string ImgDataInfo = "I:/MIL_Detection_Dataset/DSW/train_box2.txt";
//	//const MIL_STRING WorkingDataPath = L"I:/MIL_Detection_Dataset/DSW_/";
//
//	//CASE2:
//	string strProject = "COT_Resize";
//	string strSrcDir = "G:/DefectDataCenter/ParseData/Detection/"+ strProject + "/raw_data/" ;
//
//	string IconInfo = strSrcDir +"Classes.txt";
//	string IconDir = strSrcDir + "ClassesIcon/";
//	string TrainImgDataInfo = strSrcDir + "ImgBoxes_train.txt";
//	string ValImgDataInfo = strSrcDir + "ImgBoxes_val.txt";
//	MIL_STRING WorkingDataPath = m_DetectionWorkSpace + m_strProject + L"//";
//
////string strProject = "VOC/";
////	string strSrcDir = "I:/MIL_Detection_Dataset/" + strProject;
////	string IconInfo = strSrcDir +"Classes.txt";
////	string IconDir = strSrcDir + "ClassesIcon/";
////	string ImgDataInfo = strSrcDir + "ImgBoxes.txt";
//
//	MIL_UNIQUE_CLASS_ID  WorkingDataset = MclassAlloc(m_MilSystem, M_DATASET_IMAGES, M_DEFAULT, M_UNIQUE_ID);
//	m_MLDetCNN->ConstructDataset(IconInfo, IconDir, TrainImgDataInfo, WorkingDataPath,"TrainDataSet.mclassd", WorkingDataset);
//
//	//int ImageSizeX = 1440;				//进入模型训练的图片的尺寸宽360*2 ;704
//	//int ImageSizeY = 1080;				//进入模型训练的图片的尺寸高270*2  ;512
//	//int ImageSizeX = 896;				//进入模型训练的图片的尺寸宽360*2 ;704  HW
//	//int ImageSizeY = 288;				//进入模型训练的图片的尺寸高270*2  ;512 HW
//	int ImageSizeX = 1120;				//进入模型训练的图片的尺寸宽360*2 ;704  HW
//	int ImageSizeY = 224;				//进入模型训练的图片的尺寸高270*2  ;512 HW
//
//	int AugmentationNumPerImage = 0;	//进入模型训练的图片的扩充倍数
//	MIL_DOUBLE TestDatasetPercentage = 10;
//	string strSrcImgDir = "G:/DefectDataCenter/ParseData/Detection/" + strProject + "/MIL_Data/";
//	MIL_STRING SrcImgDir = m_MLDetCNN->m_AIParse->string2MIL_STRING(strSrcImgDir);
//	////*******************************必须参数*******************************//
//	MIL_UNIQUE_CLASS_ID DataContext = MclassAlloc(m_MilSystem, M_PREPARE_IMAGES_DET, M_DEFAULT, M_UNIQUE_ID);
//
//	DataContextParasStruct DataCtxParas;
//	DataCtxParas.ImageSizeX = ImageSizeX;
//	DataCtxParas.ImageSizeY = ImageSizeY;
//	DataCtxParas.DstFolderMode = 1;
//	DataCtxParas.PreparedDataFolder = SrcImgDir + MIL_TEXT("PreparedData\\");
//	memset(&DataCtxParas.AugParas, 0, sizeof(AugmentationParasStruct));
//	DataCtxParas.AugParas.AugmentationNumPerImage = AugmentationNumPerImage;
//	m_MLDetCNN->ConstructDataContext(DataCtxParas, DataContext);
//	MIL_UNIQUE_CLASS_ID PreparedDataset = MclassAlloc(m_MilSystem, M_DATASET_IMAGES, M_DEFAULT, M_UNIQUE_ID);
//	m_MLDetCNN->PrepareDataset(DataContext, WorkingDataset, PreparedDataset, WorkingDataPath, TestDatasetPercentage);
//}

void Test::MILTestGenDetDataset()
{
	//string proj_n = "COT_Resize"; //COT_Resize  DSW
	//string proj_n = "HW";
	string proj_n = "COT_Raw"; //COT_Resize
	string DetDataSetConfigPath = "G:/DefectDataCenter/ParseData/Detection/" + proj_n + "/raw_data/Config/" + proj_n + "_yolo4tiny_Para.ini";

	m_MLDetCNN->GenDataSet(DetDataSetConfigPath, proj_n);
	//DET_DATASET_PARAS_STRUCT DetDataSetPara;
	//DetDataSetPara.ClassesPath = "G:/DefectDataCenter/ParseData/Detection/COT_Resize/raw_data/Config/Classes.txt";
	//DetDataSetPara.IconDir = "G:/DefectDataCenter/ParseData/Detection/COT_Resize/raw_data/ClassesIcon/";
	//DetDataSetPara.TrainDataInfoPath = "G:/DefectDataCenter/ParseData/Detection/COT_Resize/raw_data/Config/ImgBoxes_MIL_train.txt";
	//DetDataSetPara.ValDataInfoPath = "G:/DefectDataCenter/ParseData/Detection/COT_Resize/raw_data/Config/ImgBoxes_val.txt";
	//DetDataSetPara.WorkingDataDir = "G:/DefectDataCenter/ParseData/Detection/COT_Resize/MIL_Data/";
	//DetDataSetPara.PreparedDataDir = "G:/DefectDataCenter/ParseData/Detection/COT_Resize/MIL_Data/PreparedData";
	//DetDataSetPara.ImageSizeX = 1120;
	//DetDataSetPara.ImageSizeY = 224;
	//DetDataSetPara.TestDataRatio = 10;
	//DetDataSetPara.AugFreq = 0;
	//m_MLDetCNN->GenDataSet(DetDataSetPara);
}

void Test::MILTestDetTrain()
{
	DET_TRAIN_STRUCT DtParas;
	DtParas.TrainMode = 0;
	DtParas.TrainEngineUsed = 0;
	DtParas.MaxNumberOfEpoch = 25;
	DtParas.MiniBatchSize = 16;
	DtParas.SchedulerType = 0;
	DtParas.LearningRate = 0.001;
	DtParas.LearningRateDecay = 0.1;
	DtParas.SplitPercent = 90.0;
	DtParas.WorkSpaceDir = L"G:/DefectDataCenter/ParseData/Detection";
	DtParas.DataSetName = L"COT_Raw";
	m_MLDetCNN->TrainModel(DtParas);
	return;
}

void Test::MILTestDetPredict()
{
	string proj = "COT_Raw";
	MIL_STRING Mproj = L"COT_Raw";
	string	SrcImgDir = "G:/DefectDataCenter/ParseData/Detection/" + proj + "/raw_data/TImg";
	m_MLDetCNN->m_DetDataSetPara.WorkingDataDir = "G:/DefectDataCenter/ParseData/Detection/" + proj + "/MIL_Data/";
	MIL_STRING TdDetCtxPath = L"G:/DefectDataCenter/ParseData/Detection/" + Mproj + L"/MIL_Data/" + Mproj + L".mclass";
	vector<DET_RESULT_STRUCT> vecDetResults;
	bool SaveRst = TRUE;
	m_MLDetCNN->PredictFolderImgs(SrcImgDir, TdDetCtxPath, vecDetResults, SaveRst);
}

void Test::MILTestValDetModel()
{
	string proj = "COT_Raw"; //COT_Raw  //DSW  //DSW_random  // HW //COT_Resize
	string ValDataInfoPath = "G:/DefectDataCenter/ParseData/Detection/" + proj + "/raw_data/Config/ImgBoxes_val.txt";
	MIL_STRING Mproj = L"COT_Raw";
	MIL_STRING TdDetCtxPath = L"G:/DefectDataCenter/ParseData/Detection/" + Mproj + L"/MIL_Data/" + Mproj + L".mclass";
	string strPRResultPath = "G:/DefectDataCenter/ParseData/Detection/" + proj + "/MIL_Data/PresionRecall.txt";
	string strODNetResultPath = "G:/DefectDataCenter/ParseData/Detection/" + proj + "/MIL_Data/ODNetResult.txt";
	m_MLDetCNN->ValModel_AP_50(ValDataInfoPath, TdDetCtxPath, strPRResultPath, strODNetResultPath);
}

void Test::MILTestValTxtAP50() {

	string ValDataInfoPath = "G:/DefectDataCenter/ParseData/Detection/COT_Raw/raw_data/Config/ImgBoxes_val.txt";
	string PreDataInfoPath = "G:/DefectDataCenter/ParseData/Detection/COT_Raw/Pytorch_Data/ImgBoxes_val_pd_result.txt";
	string strPRResultPath = "G:/DefectDataCenter/ParseData/Detection/COT_Raw/Pytorch_Data/PresionRecall.txt";
	m_MLDetCNN->Val_Txt_AP_50(ValDataInfoPath, PreDataInfoPath, strPRResultPath);
}

//void MILTest::MILTestDetPredict()
//{
//	string ImgType = "bmp";
//	string strProject = "COT_Resize";
//	//string ImgType = "jpg";
//	//string strProject = "VOC";
//	string	SrcDir = "G:/DefectDataCenter/ParseData/Detection/" + strProject + "/raw_data/";
//	string	SrcImgDir = SrcDir  + "TImg";
//	MIL_STRING TdDetCtxName = m_DetectionWorkSpace + m_strProject + MIL_TEXT("/PreparedData/") + m_strProject + L".mclass";
//	LARGE_INTEGER t1, t2, tc;
//	QueryPerformanceFrequency(&tc);
//	QueryPerformanceCounter(&t1);
//	MIL_UNIQUE_CLASS_ID TestCtx = MclassRestore(TdDetCtxName, m_MilSystem, M_DEFAULT, M_UNIQUE_ID);
//
//	MclassInquire(TestCtx, M_DEFAULT_SOURCE_LAYER, M_SIZE_X + M_TYPE_MIL_INT, &m_InputSizeX);
//	MclassInquire(TestCtx, M_DEFAULT_SOURCE_LAYER, M_SIZE_Y + M_TYPE_MIL_INT, &m_InputSizeY);
//	MclassInquire(TestCtx, M_CONTEXT, M_NUMBER_OF_CLASSES + M_TYPE_MIL_INT, &m_ClassesNum);
//
//	//MclassControl(TestCtx, M_CONTEXT, M_DEFAULT_PREDICT_ENGINE_PRECISION, M_FP16);
//	//设置ENGINE
//	MIL_INT engine_index = 2;
//	MIL_STRING Description;
//	MclassControl(TestCtx, M_DEFAULT, M_PREDICT_ENGINE, engine_index);
//	MclassInquire(TestCtx, M_PREDICT_ENGINE_INDEX(engine_index), M_PREDICT_ENGINE_DESCRIPTION, Description);
//	MosPrintf(MIL_TEXT("\nM_PREDICT_ENGINE_DESCRIPTION: %s \n"), Description.c_str());
//	vector<string>FilesInFolder;
//	m_MLDetCNN->m_AIParse->getFilesInFolder(SrcImgDir, ImgType, m_FilesInFolder);
//	m_MLDetCNN->m_AIParse->getFilesInFolder(SrcImgDir, ImgType, FilesInFolder);
//	//读取到内存时间不计入
//	int nFileNum = m_FilesInFolder.size();
//	vector<DET_RESULT_STRUCT> vecDetResults;
//	vector<MIL_ID>RawImageS;
//	for (int i = 0; i < nFileNum; i++) {
//		MIL_ID RawImage = MbufRestore(m_FilesInFolder[i], m_MilSystem, M_NULL);
//		RawImageS.emplace_back(RawImage);
//	}
//	m_MLDetCNN->FolderImgsPredict(RawImageS, TestCtx, vecDetResults);
//
//	//释放内存
//	for (int i = 0; i < RawImageS.size();i++) {
//		MbufFree(RawImageS[i]);
//	}
//	//将结果保存到txt文件
//	ofstream ODNetResult;
//	ODNetResult.open(SrcDir+"ODNetResult.txt", ios::out);
//	for (int i = 0; i < nFileNum; i++) {
//		string ImgInfo;
//		ImgInfo = FilesInFolder[i];
//		//写入图片路径、box、conf、classname
//		DET_RESULT_STRUCT R_i = vecDetResults[i];
//		for (int j = 0; j < R_i.Boxes.size(); j++) {
//			string strClassName;
//			m_MLDetCNN->m_AIParse->MIL_STRING2string(R_i.ClassName[j], strClassName);
//			ImgInfo = ImgInfo + " " + to_string(R_i.Boxes[j].CX)
//				+ " " + to_string(R_i.Boxes[j].CY)
//				+ " " + to_string(R_i.Boxes[j].W)
//				+ " " + to_string(R_i.Boxes[j].H)
//				+ " " + to_string(R_i.Score[j])
//				+ " " + strClassName
//				;
//		}
//		ODNetResult << ImgInfo<<endl;
//	}
//	ODNetResult.close();
//}

void Test::MILTestDetPredictMutiProcessSingle()
{
	string Index = to_string(0);
	string ImgType = "bmp";
	string strProject = "lslm_bmp";
	string strTdDetCtxName = "G:/DefectDataCenter/ParseData/Detection/" + strProject + "/MIL_Data/PreparedData/" + strProject + ".mclass";
	MIL_STRING TdDetCtxName = m_MLDetCNN->m_AIParse->string2MIL_STRING(strTdDetCtxName);
	MIL_UNIQUE_CLASS_ID TrainedCtx = MclassRestore(TdDetCtxName, m_MilSystem, M_DEFAULT, M_UNIQUE_ID);

	////设置ENGINE
	MIL_INT engine_index = 2;
	MIL_STRING Description;

	//获取模型输入尺寸
	//MIL_INT BAND;
	//MclassInquire(TrainedCtx, M_DEFAULT_SOURCE_LAYER, M_SIZE_X + M_TYPE_MIL_INT, &m_InputSizeX);
	//MclassInquire(TrainedCtx, M_DEFAULT_SOURCE_LAYER, M_SIZE_Y + M_TYPE_MIL_INT, &m_InputSizeY);
	//MclassInquire(TrainedCtx, M_CONTEXT, M_NUMBER_OF_CLASSES + M_TYPE_MIL_INT, &m_ClassesNum);
	//MclassInquire(TrainedCtx, M_DEFAULT_SOURCE_LAYER, M_SIZE_BAND + M_TYPE_MIL_INT, &BAND);

	MclassControl(TrainedCtx, M_DEFAULT, M_PREDICT_ENGINE, engine_index);
	MclassInquire(TrainedCtx, M_PREDICT_ENGINE_INDEX(engine_index), M_PREDICT_ENGINE_DESCRIPTION, Description);
	MosPrintf(MIL_TEXT("\nEngine of Process_%s is %s\n"), Index.c_str(), (Description).c_str());

	//MIL_DOUBLE  MPF1, MPF2, MPF3;
	//MclassControl(TrainedCtx, M_CONTEXT, M_DEFAULT_PREDICT_ENGINE_PRECISION, 5486L);
	//MclassInquire(TrainedCtx, M_PREDICT_ENGINE_INDEX(0), M_PREDICT_ENGINE_PRECISION, &MPF1);
	//MclassInquire(TrainedCtx, M_PREDICT_ENGINE_INDEX(1), M_PREDICT_ENGINE_PRECISION, &MPF2);
	//MclassInquire(TrainedCtx, M_PREDICT_ENGINE_INDEX(2), M_PREDICT_ENGINE_PRECISION, &MPF3);

	//MIL_DOUBLE  MPF;
	//MclassInquire(TrainedCtx, M_CONTEXT, M_PREDICT_ENGINE_PRECISION, &MPF);
	//if (MPF == M_FP16) {
	//	cout << "MPF == M_FP16" << endl;
	//}
	//else if (MPF == M_FP32) {
	//	cout << "MPF == M_FP32" << endl;
	//}
	//else
	//{
	//	cout << "error MPF" << endl;
	//}

	//MclassControl(TrainedCtx, M_CONTEXT, M_PREDICT_ENGINE_PRECISION, M_FP16);

	double calc_time = 0;
	string	SrcDir = "G:/DefectDataCenter/ParseData/Detection/" + strProject + "/raw_data/";
	string	SrcImgDir = SrcDir + "TImg_" + Index;
	m_MLDetCNN->m_AIParse->getFilesInFolder(SrcImgDir, ImgType, m_FilesInFolder);
	string DstRst = SrcDir + Index + "_ODNetResult.txt";
	MILTestDetPredictCore(TrainedCtx, SrcDir, DstRst, calc_time);
	cout << "\nOpen Process_" << Index << " calc_time time(s) is = " << calc_time << "\n" << endl; //输出时间（单位：ｓ）

}

void Test::MILTestDetPredictMutiProcess(
	string strShareMame,
	size_t ShareMameSize,
	string Index,
	string ImgType,
	string strProject
)
{
	//std::cout << "Infunc strShareMame is  " << strShareMame<< std::endl;
	//std::cout << "Infunc index is  " << index << std::endl;
	//std::cout << "Infunc filesize is  " << filesize << std::endl;

	////读取共享内存到模型
	auto m_hMap = ::OpenFileMappingA(FILE_MAP_READ, FALSE, strShareMame.c_str());
	if (m_hMap == NULL) { std::cout << "m_hMap is Null " << std::endl; return; }
	DWORD nSize = ShareMameSize + sizeof(ULONGLONG);
	BYTE* pShareBuf = (BYTE*)MapViewOfFile(m_hMap, FILE_MAP_READ, 0, 0, nSize);
	auto model_Buff = (MIL_UINT8*)(pShareBuf + sizeof(ULONGLONG));
	MIL_UNIQUE_CLASS_ID TrainedCtx;
	MclassStream(model_Buff, m_MilSystem, M_RESTORE, M_MEMORY, M_DEFAULT, M_DEFAULT, &TrainedCtx, M_NULL);
	//读取模型引擎
	//MIL_STRING Description{};
	//MIL_STRING* pDescription = &Description;
	//MclassInquire(TrainedCtx, M_PREDICT_ENGINE_INDEX(0), M_PREDICT_ENGINE_DESCRIPTION, *pDescription);
	//MosPrintf(MIL_TEXT("\nEngine of Process_%s is %s\n"), Index.c_str(), (*pDescription).c_str());

	//设置ENGINE
	MIL_INT engine_index = 2;
	MIL_STRING Description;
	MclassControl(TrainedCtx, M_DEFAULT, M_PREDICT_ENGINE, engine_index);
	MclassInquire(TrainedCtx, M_PREDICT_ENGINE_INDEX(engine_index), M_PREDICT_ENGINE_DESCRIPTION, Description);
	MosPrintf(MIL_TEXT("\nEngine of Process_%s is %s\n"), Index.c_str(), (Description).c_str());

	ofstream ODNetResult;
	string StartFile = "G:/DefectDataCenter/ParseData/Detection/lslm_bmp/raw_data/" + Index + "_StartPredictFile.txt";
	ODNetResult.open(StartFile, ios::out);
	ODNetResult << "Prepare Predict" << endl;
	ODNetResult.close();

	double calc_time = 0;
	string	SrcDir = "G:/DefectDataCenter/ParseData/Detection/" + strProject + "/raw_data/";
	string	SrcImgDir = SrcDir + "TImg_" + Index;
	m_MLDetCNN->m_AIParse->getFilesInFolder(SrcImgDir, ImgType, m_FilesInFolder);
	string DstRst = SrcDir + Index + "_ODNetResult.txt";
	MILTestDetPredictCore(TrainedCtx, SrcDir, DstRst, calc_time);
	int nFileNum = m_FilesInFolder.size();
	cout << "\nOpen Process_" << Index << " nFileNum: " << nFileNum << " average calc_time time(s) is = " << calc_time << "\n" << endl; //输出时间（单位：ｓ）
}


void Test::MILTestDetPredictCore(MIL_UNIQUE_CLASS_ID& TestCtx,
	string SrcDir,
	string DstRst,
	double& calc_time)
{
	//读取到内存时间不计入
	int nFileNum = m_FilesInFolder.size();
	vector<DET_RESULT_STRUCT> vecDetResults;
	vector<MIL_ID>RawImageS;

	LARGE_INTEGER t1, t2, tc;
	QueryPerformanceFrequency(&tc);
	QueryPerformanceCounter(&t1);

	for (int i = 0; i < nFileNum; i++) {
		MIL_ID RawImage = MbufRestore(m_FilesInFolder[i], m_MilSystem, M_NULL);
		m_MLDetCNN->Predict(RawImage, TestCtx, vecDetResults[i]);
		//RawImageS.emplace_back(RawImage);
	}
	//int CirN = 1;
	//for (int j = 0; j < CirN; j++) {
	//	m_MLDetCNN->FolderImgsPredict(RawImageS, TestCtx, vecDetResults);
	//}

	QueryPerformanceCounter(&t2);
	calc_time = (double)(t2.QuadPart - t1.QuadPart) / (double)tc.QuadPart / (double)nFileNum;

	for (int i = 0; i < nFileNum; i++) {
		MbufFree(RawImageS[i]);
	}

	//将结果保存到txt文件
	ofstream ODNetResult;
	ODNetResult.open(DstRst, ios::out);
	for (int i = 0; i < nFileNum; i++) {
		string ImgInfo;
		m_MLDetCNN->m_AIParse->MIL_STRING2string(m_FilesInFolder[i], ImgInfo);
		//写入图片路径、box、conf、classname
		DET_RESULT_STRUCT R_i = vecDetResults[i];
		for (int j = 0; j < R_i.Boxes.size(); j++) {
			string strClassName;
			m_MLDetCNN->m_AIParse->MIL_STRING2string(R_i.ClassName[j], strClassName);
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


void Test::mutiThreadPrepare()
{
	//读取待图片路径测路径



}

void Test::MILTestDetPredictMutiThreadCore()
{
	LARGE_INTEGER t1, t2, tc;
	QueryPerformanceFrequency(&tc);
	QueryPerformanceCounter(&t1);

	//读取到内存时间不计入
	DET_RESULT_STRUCT Result_i;
	map<string, MIL_ID >::iterator it;
	for (it = m_PathRawImageMap.begin(); it != m_PathRawImageMap.end();) {
		Result_i.ImgPath = it->first;
		m_MLDetCNN->Predict(it->second, m_TrainedCtx, Result_i);
		m_vecDetResults.emplace_back(Result_i);

		MbufFree(it->second);  //释放已经预测过的图片内存
		m_PathRawImageMap.erase(it++);
	}
	//for (auto it : m_PathRawImageMap) {

	//	Result_i.ImgPath = it.first;
	//	m_MLDetCNN->Predict(it.second, m_TrainedCtx, Result_i);
	//	m_vecDetResults.emplace_back(Result_i);
	//	m_PathRawImageMap.erase(it);
	//}


}

//void MILTest::MILTestDetPredictMutiThread()
//{
//	predictBegin();
//	string SrcDir = "G:/DefectDataCenter/ParseData/Detection/" + m_strProject + "/raw_data/";
//	string	SrcImgDir = "G:/DefectDataCenter/ParseData/Detection/" + m_strProject + "/raw_data/MutiThread_TImg";
//	
//	ofstream StMutiThread;
//	string StDstRst = SrcDir+ "StMutiThread.txt";
//	StMutiThread.open(StDstRst, ios::out);
//	StMutiThread.close();
//	
//	thread tt1(&MILTest::MILTestDetPredictMutiThreadCore, this);
//	tt1.join();
//	thread t2(&MILTest::MILTestDetPredictMutiThreadCore, this);
//	t2.join();
//
//	MILTestDetPredictMutiThreadCore();
//
//	//将结果保存到txt文件
//	ofstream ODNetResult;
//	ODNetResult.open(SrcDir + "ODNetResult_MutiThread.txt", ios::out);
//	for (vector<DetResult>::iterator it = m_vecDetResults.begin(); it != m_vecDetResults.end();it++) {
//		string ImgInfo = it->ImgPath;
//		//写入图片路径、box、conf、classname
//		DetResult R_i =*it;
//		for (int j = 0; j < R_i.Boxes.size(); j++) {
//			string strClassName;
//			m_MLDetCNN->m_AIParse->MIL_STRING2string(R_i.ClassName[j], strClassName);
//			ImgInfo = ImgInfo + " " + to_string(R_i.Boxes[j].CX)
//				+ " " + to_string(R_i.Boxes[j].CY)
//				+ " " + to_string(R_i.Boxes[j].W)
//				+ " " + to_string(R_i.Boxes[j].H)
//				+ " " + to_string(R_i.Score[j])
//				+ " " + strClassName;
//		}
//		ODNetResult << ImgInfo << endl;
//
//	}
//
//
//	ODNetResult.close();
//
//}

void Test::MILTestONNXPredict()
{
	MIL_STRING TdONNXCtxName = MIL_TEXT("G:/DefectDataCenter/ParseData/Detection/COT_Raw/Pytorch_Data/yolo4tiny_COT_Raw.onnx");
	MIL_UNIQUE_CLASS_ID TestONNXCtx = MclassAlloc(m_MilSystem, M_CLASSIFIER_ONNX, M_DEFAULT, M_UNIQUE_ID);
	MclassImport(TdONNXCtxName, M_ONNX_FILE, TestONNXCtx, M_DEFAULT, M_DEFAULT, M_DEFAULT);
	MIL_INT engine_index = 2;
	MIL_STRING Description;
	MclassControl(TestONNXCtx, M_DEFAULT, M_PREDICT_ENGINE, engine_index);
	MclassInquire(TestONNXCtx, M_PREDICT_ENGINE_INDEX(engine_index), M_PREDICT_ENGINE_DESCRIPTION, Description);
	MosPrintf(MIL_TEXT("\nM_PREDICT_ENGINE_DESCRIPTION: %s \n"), Description.c_str());

	MclassInquire(TestONNXCtx, M_DEFAULT_SOURCE_LAYER, M_SIZE_X + M_TYPE_MIL_INT, &m_InputSizeX);
	MclassInquire(TestONNXCtx, M_DEFAULT_SOURCE_LAYER, M_SIZE_Y + M_TYPE_MIL_INT, &m_InputSizeY);
	//MclassInquire(TestONNXCtx, M_CONTEXT, M_NUMBER_OF_CLASSES + M_TYPE_MIL_INT, &m_ClassesNum);
	MclassControl(TestONNXCtx, M_DEFAULT, M_TARGET_IMAGE_SIZE_X, m_InputSizeX);
	MclassControl(TestONNXCtx, M_DEFAULT, M_TARGET_IMAGE_SIZE_Y, m_InputSizeY);
	MclassControl(TestONNXCtx, M_DEFAULT, M_TARGET_IMAGE_SIZE_BAND, 3);
	MclassPreprocess(TestONNXCtx, M_DEFAULT);

	string ImgDir = "G:/DefectDataCenter/ParseData/Detection/COT_Raw/raw_data/TImg";
	string strODNetResultPath = "G:/DefectDataCenter/ParseData/Detection/COT_Raw/raw_data/TImg_ONNX_Result.txt";
	vector<MIL_STRING> vecImgPaths;
	vector<MIL_STRING> Files;
	m_MLDetCNN->m_AIParse->getFilesInFolder(ImgDir, "bmp", Files);

	vector<DET_RESULT_STRUCT> vecDetResults;
	for (int i = 0; i < Files.size(); i++) {
		MIL_STRING ImagepATH = Files[i];
		MIL_ID Image = MbufRestore(ImagepATH, m_MilSystem, M_NULL);
		MIL_INT m_ImageSizeX = MbufInquire(Image, M_SIZE_X, M_NULL);
		MIL_INT m_ImageSizeY = MbufInquire(Image, M_SIZE_Y, M_NULL);

		MIL_ID ImageReduce = MbufAllocColor(m_MilSystem, 3, m_InputSizeX, m_InputSizeY, M_FLOAT + 32, M_IMAGE + M_PROC, M_NULL);
		MimResize(Image, ImageReduce, M_FILL_DESTINATION, M_FILL_DESTINATION, M_BICUBIC);
		MimArith(ImageReduce, 255.0, ImageReduce, M_DIV_CONST);
		MIL_UNIQUE_CLASS_ID ClassRes = MclassAllocResult(m_MilSystem, M_PREDICT_ONNX_RESULT, M_DEFAULT, M_UNIQUE_ID);

		MclassPredict(TestONNXCtx, ImageReduce, ClassRes, M_DEFAULT);
		clock_t  t1 = clock();
		int CircleNum = 100;
		for (int i = 0; i < CircleNum; i++) {
			MclassPredict(TestONNXCtx, ImageReduce, ClassRes, M_DEFAULT);
		}
		clock_t  t2 = clock();
		cout << "FPS: " << CircleNum * 1.0 / (double(t2 - t1) / CLOCKS_PER_SEC) << endl;
		MbufFree(ImageReduce);
		MbufFree(Image);
		MIL_INT NO = 0;
		MclassGetResult(ClassRes, M_GENERAL, M_NUMBER_OF_OUTPUTS + M_TYPE_MIL_INT, &NO);
		if (NO == 0) {
			continue;
		}
		vecImgPaths.emplace_back(ImagepATH);
		vector<MIL_UINT8>ROut;
		vector<MIL_DOUBLE>Out;
		vector<MIL_INT>OutSp;
		for (int i = 0; i < NO; i++) {
			MclassGetResult(ClassRes, M_OUTPUT_INDEX(i), M_OUTPUT_RAW, ROut);
			MclassGetResult(ClassRes, M_OUTPUT_INDEX(i), M_OUTPUT_SHAPE, OutSp);
			MclassGetResult(ClassRes, M_OUTPUT_INDEX(i), M_OUTPUT_DATA, Out);
		}
		//解析Result
		DET_RESULT_STRUCT tmpRst;
		for (int i = 0; i < OutSp[0]; i++) {
			Box tmp_box;
			//vector<float>tmp_target;
			vector<float>tmp_target;
			int box_S = i * OutSp[1];
			//int box_E = box_S+4;
			//tmp_box.assign(Out.begin() + box_S, Out.begin() + box_E);
			tmp_box.CX = Out[box_S + 0];
			tmp_box.CY = Out[box_S + 1];
			tmp_box.W = Out[box_S + 2] - Out[box_S + 0];
			tmp_box.H = Out[box_S + 3] - Out[box_S + 1];
			float tmp_score = Out[box_S + 4] * Out[box_S + 5];
			int ClassIndex = Out[box_S + 6];
			tmpRst.ClassName.emplace_back(to_wstring(ClassIndex));
			tmpRst.Boxes.emplace_back(tmp_box);
			tmpRst.Score.emplace_back(tmp_score);
			tmpRst.ClassIndex.emplace_back(ClassIndex);
		}
		vecDetResults.push_back(tmpRst);
	}
	m_MLDetCNN->saveResult2File(strODNetResultPath, vecImgPaths, vecDetResults);
}

//void MILTest::OpencvONNXPredict()
//{
//	//yolo7tiny_COT_Raw.onnx 被onnxruntime、MIL调用都正常。opencv报错
//	string onnx_path = "G:/DefectDataCenter/ParseData/Detection/COT_Raw/Pytorch_Data/yolo7tiny_COT_Raw.onnx";
//	dnn::Net net = dnn::readNetFromONNX(onnx_path);
//	//opencv test
//	string img_name = "G:/DefectDataCenter/ParseData/Detection/COT_Raw/raw_data/TImg/0_7_15.bmp";
//	cv::Mat image = imread(img_name, -1);
//	//image = imread(fn[i], IMREAD_GRAYSCALE);
//	// 2. convert color space, opencv read the image in BGR
//	Mat img_float;
//	// convert to float format
//	image.convertTo(img_float, CV_32F, 1.0 / 255);
//	// 3. resize the image for resnet101 model
//	Mat img_resize;
//	resize(img_float, img_resize, Size(2688, 448), INTER_CUBIC);
//	//cv::Mat blob = cv::dnn::blobFromImage(img_resize,  CV_32F);  // 由图片加载数据 还可以进行缩放、归一化等预处理操作
//	net.setInput(img_resize);  // 设置模型输入
//	Mat detections = net.forward();
//	Mat detectionMat(detections.size[2], detections.size[3], CV_32F, detections.ptr<float>());
//
//	for (int i = 0; i < detectionMat.rows; i++)
//	{
//		//自定义阈值
//		if (detectionMat.at<float>(i, 2) >= 0.14)
//		{
//			int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * image.cols);
//			int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * image.rows);
//			int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * image.cols);
//			int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * image.rows);
//
//			Rect object((int)xLeftBottom, (int)yLeftBottom,
//				(int)(xRightTop - xLeftBottom),
//				(int)(yRightTop - yLeftBottom));
//
//			rectangle(image, object, Scalar(0, 255, 0));
//		}
//
//
//		//int width = img_resize.cols;//获取图像宽度
//		//int height = img_resize.rows;//获取图像高度
//		//int channel = img_resize.channels();//获取通道数
//
//
//
//	}
//}

void Test::MILTestKTtreedbscan()
{
	//function: 对背景图片进行聚类，并按比例精简
	CDBSCANPtr m_CDBSCANPtr = CDBSCANPtr(new CDBSCAN(m_MilSystem));
	double radius = 1.2;
	int minPoints = 60;
	string ImgDir = "G:/DefectDataCenter/Test/Src/90";
	double AspectRatioTHD = 3;
	double RemovalRatio = 0.5;
	bool REMOVEIMG = false;


	vector<MIL_STRING> efftImgPaths;
	vector<vector<int>> Labels;
	vector<MIL_STRING> unefftImgPaths;
	m_CDBSCANPtr->ImgCluster(radius, minPoints, ImgDir, AspectRatioTHD,
		efftImgPaths, Labels, unefftImgPaths);
	//对聚类好的Labels按比例抽取，保存或者删除
	//Labels[0]为噪声，全保留
	vector<vector<int>>unNecLabels;
	unNecLabels.resize(Labels.size());
	unNecLabels[0] = Labels[0];
	for (int i = 1; i < Labels.size(); i++) {
		random_shuffle(Labels[i].begin(), Labels[i].end());
		int nSelectNum = int(Labels[i].size() * RemovalRatio);
		unNecLabels[i].assign(Labels[i].begin(), Labels[i].begin() + nSelectNum);
	}
	if (REMOVEIMG) {
		//case2：删除去除的图片
		m_CDBSCANPtr->removalImg(efftImgPaths, unNecLabels, unefftImgPaths);
	}
	else {
		//case1：保存,查看去除的图
		MIL_STRING DstImgDir = L"G:/DefectDataCenter/Test/ImgCluster/Cpp_SPA90/";
		m_CDBSCANPtr->saveClusterRst(DstImgDir, efftImgPaths, unNecLabels, unefftImgPaths);
	}
}

//void MILTest::OpencvTest(MIL_ID& ImageReshape)
//{
// //opencv_img-->MIL_ID_img
//	//opencv test
//	string img_name = "G:/DefectDataCenter/ParseData/Detection/lslm/raw_data/TImg/lslm.bmp";
//	cv::Mat image = imread(img_name, CV_LOAD_IMAGE_UNCHANGED);
//	//image = imread(fn[i], IMREAD_GRAYSCALE);
//	// 2. convert color space, opencv read the image in BGR
//	Mat img_float;
//	// convert to float format
//	image.convertTo(img_float, CV_32F, 1.0 / 255);
//	// 3. resize the image for resnet101 model
//	Mat img_resize;
//	resize(img_float, img_resize, Size(704, 512), INTER_CUBIC);
//
//	int width = img_resize.cols;//获取图像宽度
//	int height = img_resize.rows;//获取图像高度
//	int channel = img_resize.channels();//获取通道数
//	float* pNorlzBuffer = new float[(int)(width * height * 3)];
//	//数组方法遍历
//	for (int h = 0; h < height; h++) //height
//	{
//		for (int w = 0; w < width; w++) //width
//		{
//			if (channel == 3)//彩色图像
//			{
//				//bgr 是一个vector，包含三个通道的值
//				Vec3f bgr = img_resize.at<Vec3f>(h, w);
//				/*				bgr[0] = 255 - bgr[0];
//								bgr[1] = 255 - bgr[1];
//								bgr[2] = 255 - bgr[2];
//								img_resize.at<Vec3f>(h, w) = bgr;*/
//								//cout << " b: " << bgr[0] << " g: " << bgr[1] << " r: " << bgr[2] << endl;
//
//				pNorlzBuffer[h + w + 0] = bgr[0];
//				pNorlzBuffer[h + w + 1] = bgr[1];
//				pNorlzBuffer[h + w + 2] = bgr[2];
//
//			}
//
//		}
//	}
//
//	ImageReshape = MbufAllocColor(m_MilSystem, 3, width, height, M_FLOAT + 32, M_IMAGE + M_PROC, M_NULL);
//	MbufPut(ImageReshape, pNorlzBuffer);
//
//	float* m_pResizeBufferOrgRGB = new float[(int)(width * height * 3)];
//	MbufGetColor(ImageReshape, M_PLANAR, M_ALL_BANDS, m_pResizeBufferOrgRGB);
//	int nPixelIndex = 0;
//	long nResizeCount = width * height;
//	for (int i = 0; i < nResizeCount; i++) {
//		//M_PLANAR的图片存放格式为：RRRR...GGGG...BBBB
//		//OpenCV的图片存放格式为：BGRBGR...BGR
//		//AI模型使用OPENCV图像形式训练所得，故需要将MIL格式转换
//		for (int j = 2; j >= 0; j--) {
//			//pNorlzBuffer[nPixelIndex] = m_pResizeBufferOrgRGB[i + nResizeCount * j];    //R_OpenCv<--R_MIL
//			//cout << "m_pResizeBufferOrgRGB[nPixelIndex]: " << m_pResizeBufferOrgRGB[nPixelIndex] << endl;
//			//cout << "pNorlzBuffer[nPixelIndex]: " << pNorlzBuffer[nPixelIndex] << endl;
//
//			nPixelIndex++;
//		}
//	}
//
//}


//void MILTest::Pytest()
//{
//	//Py_SetPythonHome(L"D:/Anaconda3/envs/AI_gpu/include");
//	Py_Initialize(); //初始化python解释器
//	if (!Py_IsInitialized()) {
//		std::system("pause");
//		//return -99;
//	} //查看python解释器是否成功初始化
//
//	PyRun_SimpleString("import sys");
//	PyRun_SimpleString("sys.path.append('I:/MIL_AI/testMILAI')");
//	PyRun_SimpleString("sys.path.append('I:/MIL_AI/testMILAI/site-packages')");
//	PyObject* pModule = PyImport_Import(PyUnicode_FromString("Img_Cluster"));
//	if (!pModule) {
//		cout << "Can't find  Img_Cluster" << endl;
//		std::system("pause");
//	}
//
//	////调用pt1函数
//	PyObject* pFunc = PyObject_GetAttrString(pModule, "pt1");//这里是要调用的函数名
//	PyObject* pyParams = PyTuple_New(4); //定义两个变量
//	string Csrc_dir = "G:/DefectDataCenter/Test/Src/90";
//	string dst_dir = "G:/DefectDataCenter/Test/ImgCluster";
//	float	Eeps = 1.2;
//	//const char* pn = "SPA90";
//	string pn = "SPA90";
//	PyTuple_SetItem(pyParams, 0, Py_BuildValue("s", Csrc_dir.c_str()));// 变量格式转换成python格式
//	PyTuple_SetItem(pyParams, 1, Py_BuildValue("s", dst_dir.c_str()));// 变量格式转换成python格式
//	PyTuple_SetItem(pyParams, 2, Py_BuildValue("f", Eeps));// 变量格式转换成python格式
//	PyTuple_SetItem(pyParams, 3, Py_BuildValue("s", pn.c_str()));// 变量格式转换成python格式
//	PyObject_CallObject(pFunc, pyParams);//调用函数
//
//	//PyObject* pFunc = PyObject_GetAttrString(pModule, "pt2");//这里是要调用的函数名
//	//PyEval_CallObject(pFunc, NULL);//调用函数
//	//销毁python相关
//	//Py_DECREF(pyParams);
//	//Py_DECREF(pFunc);
//	Py_DECREF(pModule);
//	Py_Finalize();
//}