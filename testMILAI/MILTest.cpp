#include "MILTest.h"

MILTest::MILTest(MIL_ID MilSystem, MIL_ID MilDisplay, MIL_STRING strProject):
	m_MilSystem(MilSystem),
	m_MilDisplay(MilDisplay),
	m_strProject(strProject)
{
	m_MLClassCNN = CMLClassCNNPtr(new CMLClassCNN( MilSystem,  MilDisplay));
	m_MLDetCNN = CMLDetCNNPtr(new CMLDetCNN(MilSystem, MilDisplay));
}

MILTest::~MILTest()
{

}

void MILTest::getIcon(vector<MIL_STRING> OriginalDataPath,
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

void MILTest::getModelInfo(MIL_UNIQUE_CLASS_ID& Model)
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

void MILTest::savePredictedImg()
{
	//创建目的class的文件夹
	string strDstDir;
	m_MLClassCNN->m_AIParse->MIL_STRING2string(m_DstImgDir, strDstDir);
	m_MLClassCNN->CreateFolder(m_DstImgDir);

	for (std::vector<MIL_STRING>::iterator it = m_ClassNames.begin(); it != m_ClassNames.end(); ++it) {
		m_MLClassCNN->CreateFolder(m_DstImgDir + (*it));
	}
	//遍历所有文件及结果，并将图片保存到相应的Index
	MIL_INT Img_index = 0;
	for (std::vector<MIL_STRING>::iterator it = m_FilesInFolder.begin(); it != m_FilesInFolder.end(); it++) {
		MIL_STRING RawImagePath = (*it);
		MIL_ID Image = MbufRestore(RawImagePath, m_MilSystem, M_NULL);
		string::size_type iPos = RawImagePath.find_last_of('/') + 1;
		MIL_STRING ImageRawName = RawImagePath.substr(iPos, RawImagePath.length() - iPos);
		MIL_STRING DstRootPath = m_DstImgDir + m_ClassNames[m_vecResults[Img_index].PredictClass] + MIL_TEXT("//") + ImageRawName;
		MbufExport(DstRootPath, M_BMP, Image);
		MbufFree(Image);
		Img_index++;
	}
}

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

void MILTest::InitClassWeights()
{
	m_ClassWeights.resize(m_ClassesNum, 1.0);
}

void MILTest::CropImgs()
{
	int CropWH = 64;
	string SrcImgDir = "G:/DefectDataCenter/李巨欣/ShenZhen/Lot_Gather/ALL";
	string DstImgDir = "G:/DefectDataCenter/李巨欣/ShenZhen/Lot_Gather/ALL_Crop";

	MIL_STRING MDstImgDir = m_MLClassCNN->m_AIParse->string2MIL_STRING(DstImgDir);
	m_MLClassCNN->CreateFolder(MDstImgDir);


	vector<string>folders;
	m_MLClassCNN->m_AIParse->getFoldersInFolder(SrcImgDir, folders);
	if (folders.size() > 0) {
		for (int i = 0; i < folders.size(); i++) {
			string SrcSubDir = SrcImgDir + "//" + folders[i];
			string DstSubDir = DstImgDir + "//" + folders[i];

			MIL_STRING MDstSubDir = m_MLClassCNN->m_AIParse->string2MIL_STRING(DstSubDir);
			m_MLClassCNN->CreateFolder(MDstSubDir);
			vector<string>FilesInFolder;
			m_MLClassCNN->m_AIParse->getFilesInFolder(SrcSubDir, "bmp", FilesInFolder);
			for (int j = 0; j < FilesInFolder.size(); j++) {
				MIL_ID ImgOut;
				string srcImgPath = FilesInFolder[j];

				vector<string>ListFilesInFolder;
				m_MLClassCNN->m_AIParse->Split(FilesInFolder[j], ListFilesInFolder, "//");
				string dstImgPath = DstSubDir + "//" + ListFilesInFolder.back();

				MIL_STRING MsrcImgPath = m_MLClassCNN->m_AIParse->string2MIL_STRING(srcImgPath);
				MIL_STRING MdstImgPath = m_MLClassCNN->m_AIParse->string2MIL_STRING(dstImgPath);

				MIL_ID ImgIn = MbufRestore(MsrcImgPath, m_MilSystem, M_NULL);

				m_MLClassCNN->m_AIParse->ImgCenterCrop(ImgIn, CropWH, ImgOut);
				MbufExport(MdstImgPath, M_BMP, ImgOut);
				MbufFree(ImgOut);
			}
		}
	}
	else
	{
		vector<string>FilesInFolder;
		m_MLClassCNN->m_AIParse->getFilesInFolder(SrcImgDir, "bmp", FilesInFolder);
		for (int j = 0; j < FilesInFolder.size(); j++) {
			MIL_ID ImgOut;

			vector<string>ListFilesInFolder;
			m_MLClassCNN->m_AIParse->Split(FilesInFolder[j], ListFilesInFolder, "//");


			string dstImgPath = DstImgDir + "//" + ListFilesInFolder.back();

	
			MIL_STRING MsrcImgPath = m_MLClassCNN->m_AIParse->string2MIL_STRING(FilesInFolder[j]);
			MIL_STRING MdstImgPath = m_MLClassCNN->m_AIParse->string2MIL_STRING(dstImgPath);

			MIL_ID ImgIn = MbufRestore(MsrcImgPath, m_MilSystem, M_NULL);

			m_MLClassCNN->m_AIParse->ImgCenterCrop(ImgIn, CropWH, ImgOut);
			MbufExport(MdstImgPath, M_BMP, ImgOut);
			MbufFree(ImgOut);
		}
	}
}

void MILTest::FillImgs()
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
			MbufCopyColor2d(RawImage, MonoImage, M_ALL_BANDS, 0, 0, M_ALL_BANDS,0 , SizeX - SizeY, SizeX, SizeY);
		}

		MbufExport(DstRootPath, M_BMP, MonoImage);
	}
	else if(SizeBAND == 3) {

		MIL_ID MonoImage = MbufAllocColor(m_MilSystem, SizeBAND, SImgW, SImgH, 8 + M_UNSIGNED, M_IMAGE + M_PROC, M_NULL);
		MIL_UINT8 Value1 = 0;
		std::unique_ptr<BYTE[]> ScaledImage = std::make_unique<BYTE[]>(SizeBAND *SImgW * SImgH);
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
	
		MbufPutColor2d(MonoImage, M_PACKED + M_BGR24,M_ALL_BANDS, 0,0,SImgW,SImgH,ScaledImage.get());
		//MbufExport(L"G:/DefectDataCenter/TImg/A1.bmp", M_BMP, MonoImage);
		MbufCopyColor2d(RawImage, MonoImage, M_ALL_BANDS, 0, 0, M_ALL_BANDS, SizeY - SizeX, 0, SizeX, SizeY);
		MbufExport(DstRootPath, M_BMP, MonoImage);
	}
	
}

void MILTest::MILTestGenDataset()
{
	MIL_STRING AuthorName = MIL_TEXT("AA");
	MIL_STRING OriginalDataPath = m_ClassifierSrcDataDir + m_strProject+ L"//";
	MIL_STRING WorkingDataPath = m_ClassifierWorkSpace + m_strProject+ L"//";			

	vector<MIL_STRING>ClassName = { MIL_TEXT("91") ,MIL_TEXT("10") };
	vector<MIL_STRING > ClassIcon;
	for (int i = 0; i < ClassName.size(); i++) {
		 ClassIcon .emplace_back(m_ClassifierSrcDataDir + m_strProject + L"//"+ ClassName[i]+L".mim");
	}
	MIL_UNIQUE_CLASS_ID  Dataset = MclassAlloc(m_MilSystem, M_DATASET_IMAGES, M_DEFAULT, M_UNIQUE_ID);
	m_MLClassCNN->ConstructDataset(ClassName, ClassIcon, AuthorName, OriginalDataPath, WorkingDataPath, Dataset);
	m_MLClassCNN->CreateFolder(WorkingDataPath);
	MclassControl(Dataset, M_CONTEXT, M_CONSOLIDATE_ENTRIES_INTO_FOLDER, WorkingDataPath);

	////*******************************必须参数*******************************//
	MIL_UNIQUE_CLASS_ID DataContext = MclassAlloc(m_MilSystem, M_PREPARE_IMAGES_CNN, M_DEFAULT, M_UNIQUE_ID);
	DataContextParasStruct DataCtxParas;
	MIL_DOUBLE TestDatasetPercentage = 10;
	DataCtxParas.ImageSizeX = 128;
	DataCtxParas.ImageSizeY = 128;
	DataCtxParas.PreparedDataFolder = WorkingDataPath;
	memset(&DataCtxParas.AugParas, 0, sizeof(AugmentationParasStruct));
	DataCtxParas.AugParas.AugmentationNumPerImage = 0;
	DataCtxParas.ResizeModel = 1;
	DataCtxParas.AugParas.ScaleFactorMax = 1.03; //1.03
	DataCtxParas.AugParas.ScaleFactorMin = 0.97; //0.97
	DataCtxParas.AugParas.RotateAngleDelta = 20; //10
	DataCtxParas.AugParas.IntyDeltaAdd = 32;  //32
	DataCtxParas.AugParas.DirIntyMax = 1.2; //1.2
	DataCtxParas.AugParas.DirIntyMin = 0.8; //0.8
	DataCtxParas.AugParas.SmoothnessMax = 50; //50 {0<x<100}
	DataCtxParas.AugParas.SmoothnessMin = 0.5; //0.5 {0<x<100}
	DataCtxParas.AugParas.GaussNoiseStdev = 25; //25
	DataCtxParas.AugParas.GaussNoiseDelta = 25; //25

	m_MLClassCNN->ConstructDataContext(DataCtxParas, DataContext);
	//MIL_STRING WorkingDataPath = SrcImgDir + MIL_TEXT("DataSet\\DataSet.mclassd");				//原始数据根文件下的 存放中间数据的文件夹
	//MIL_UNIQUE_CLASS_ID WorkingDataset = MclassRestore(WorkingDataPath, m_MilSystem, M_DEFAULT, M_UNIQUE_ID);
	MIL_UNIQUE_CLASS_ID PreparedDataset = MclassAlloc(m_MilSystem, M_DATASET_IMAGES, M_DEFAULT, M_UNIQUE_ID);
	m_MLClassCNN->PrepareDataset(DataContext, Dataset, PreparedDataset, WorkingDataPath, TestDatasetPercentage);

}

void MILTest::MILTestTrain()
{

	int MaxNumberOfEpoch = 5;			//模型训练次数
	int MiniBatchSize = 64;				//模型训练单次迭代的张数
	//////*******************************必须参数*******************************//
	MIL_STRING PreparedPath = m_ClassifierWorkSpace +m_strProject+ MIL_TEXT("/PreparedDataset.mclassd");				
	MIL_UNIQUE_CLASS_ID PreparedDataset = MclassRestore(PreparedPath, m_MilSystem, M_DEFAULT, M_UNIQUE_ID);
	MIL_UNIQUE_CLASS_ID TrainCtx = MclassAlloc(m_MilSystem, M_TRAIN_CNN, M_DEFAULT, M_UNIQUE_ID);  
	ClassifierParasStruct ClassifierParas;
	ClassifierParas.TrainMode = 0;
	ClassifierParas.TrainEngineUsed = 0;
	ClassifierParas.MaxNumberOfEpoch = MaxNumberOfEpoch;
	ClassifierParas.MiniBatchSize = MiniBatchSize;
	ClassifierParas.SchedulerType = 0;
	ClassifierParas.LearningRate = 0.0001;
	ClassifierParas.LearningRateDecay = 0;
	ClassifierParas.SplitPercent = 90.0;
	ClassifierParas.TrainDstFolder = m_ClassifierWorkSpace+L"//" + m_strProject + MIL_TEXT("/PreparedData/");
	m_MLClassCNN->ConstructTrainCtx(ClassifierParas, TrainCtx);
	MIL_UNIQUE_CLASS_ID TrainedClassifierCtx;
	MIL_UNIQUE_CLASS_ID PrevClassifierCtx;
	MIL_STRING ClassifierDumpFile = ClassifierParas.TrainDstFolder + m_strProject+L".mclass";
	m_MLClassCNN->TrainClassifier(PreparedDataset, TrainCtx, PrevClassifierCtx, TrainedClassifierCtx, ClassifierDumpFile);
	return;
}

void MILTest::MILTestPredict() {

	MIL_STRING PreClassifierName = m_ClassifierWorkSpace + m_strProject + MIL_TEXT("/PreparedData/") + m_strProject + L".mclass";
	string	SrcImgDir = "G:/DefectDataCenter/ParseData/Classifier/SXX_GrayWave/Original_Gray3/91/";
	m_DstImgDir = MIL_TEXT("G:/DefectDataCenter/ParseData/Classifier/SXX_GrayWave/Original_Gray3/MIL/");
	LARGE_INTEGER t1, t2, tc;
	QueryPerformanceFrequency(&tc);
	QueryPerformanceCounter(&t1);
	MIL_UNIQUE_CLASS_ID TestCtx = MclassRestore(PreClassifierName, m_MilSystem, M_DEFAULT, M_UNIQUE_ID);
	getModelInfo(TestCtx);
	//设置ENGINE
	MIL_STRING Description;
	MclassInquire(TestCtx, M_PREDICT_ENGINE_INDEX(0), M_PREDICT_ENGINE_DESCRIPTION, Description);
	MosPrintf(MIL_TEXT("\nM_PREDICT_ENGINE_DESCRIPTION: %s \n"), Description.c_str());
	
	m_MLClassCNN->m_AIParse->getFilesInFolder(SrcImgDir, "bmp", m_FilesInFolder);
	int nFileNum = m_FilesInFolder.size();
	vector<MIL_DOUBLE> ClassWeights(m_ClassesNum, 1.0);
	m_MLClassCNN->FolderImgsPredict(m_FilesInFolder, ClassWeights, TestCtx, m_vecResults);

	QueryPerformanceCounter(&t2);
	double calc_time = (double)(t2.QuadPart - t1.QuadPart) / (double)tc.QuadPart;
	if (m_SavePredictedImg) {
		savePredictedImg();
	}
}

void MILTest::MILTestPredictWithBlob()
{
	//MIL_ID RawIma = MbufRestore(L"D:/LeetCode/Img/AugImg/L1C14 (2).bmp", m_MilSystem, M_NULL);
	string BlobTxt = "D:/LeetCode/Img/BlobTxt";
	string AugImg = "D:/LeetCode/Img/AugImg";
	string CropedImg = "D:/LeetCode/Img/CropImg";
	vector<string> Files;
	m_MLClassCNN->m_AIParse->getFilesInFolder(BlobTxt, "txt", Files);

	for (auto& file : Files) {
		//读取blob数据
		vector<int> blob_px;
		vector<int> blob_py;
		m_MLClassCNN->m_AIParse->readBlob2Vector(file, blob_px, blob_py);
		//读取img
		auto pos1 = file.find_last_of("//") + 1;
		auto pos2 = file.find_last_of(".");
		auto count = pos2 - pos1;
		string img_name = file.substr(pos1, count);
		cout << "img_name: " << img_name << endl;
		vector<MIL_ID> ClipImgs;
		m_MLClassCNN->m_AIParse->blobClip(AugImg, img_name, blob_px, blob_py, ClipImgs, CropedImg);
	}
}

void MILTest::MILTestPredictEngine()
{
	string SrcImgDir = "G:/DefectDataCenter/lianglichuang/SXX_big_defect/SXX_big_defect_0";
	m_DstImgDir = MIL_TEXT("G:/DefectDataCenter/lianglichuang/SXX_big_defect/SXX_big_defect_0_R/");
	MIL_STRING PreClassifierName = MIL_TEXT("G:/DefectDataCenter/zhjuzhiqiang_2023/2023/SXX/PreparedData/SXX.mclass");
	MIL_UNIQUE_CLASS_ID TestCtx = MclassRestore(PreClassifierName, m_MilSystem, M_DEFAULT, M_UNIQUE_ID);
	getModelInfo(TestCtx);
	MIL_INT NbPredEngines = 0;
	MclassInquire(TestCtx, M_DEFAULT, M_NUMBER_OF_PREDICT_ENGINES + M_TYPE_MIL_INT, &NbPredEngines);
	for (int engine_index=0; engine_index< NbPredEngines; engine_index++){
		LARGE_INTEGER t1, t2, tc;
		QueryPerformanceFrequency(&tc);
		QueryPerformanceCounter(&t1);
		//设置ENGINE
		MIL_STRING Description;
		MclassControl(TestCtx, M_DEFAULT, M_PREDICT_ENGINE, engine_index);
		MclassInquire(TestCtx, M_PREDICT_ENGINE_INDEX(engine_index), M_PREDICT_ENGINE_DESCRIPTION, Description);
		MosPrintf(MIL_TEXT("\nM_PREDICT_ENGINE_DESCRIPTION: %s \n"), Description.c_str());

		m_MLClassCNN->m_AIParse->getFilesInFolder(SrcImgDir, "bmp", m_FilesInFolder);
		int nFileNum = m_FilesInFolder.size();
		vector<MIL_DOUBLE> ClassWeights(m_ClassesNum, 1.0);
		m_MLClassCNN->FolderImgsPredict(m_FilesInFolder, ClassWeights, TestCtx, m_vecResults);
		QueryPerformanceCounter(&t2);
		double calc_time = (double)(t2.QuadPart - t1.QuadPart) / (double)tc.QuadPart;
		MosPrintf(MIL_TEXT("\nM_PREDICT_ENGINE_DESCRIPTION: %s  calc time is %f\n"), Description.c_str(), calc_time);
	}

}

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

void MILTest::MILTestGenDetDataset()
{
	string proj_n = "DSW";
	//string proj_n = "HW";
	//string proj_n = "COT_Raw";
	string DetDataSetConfigPath = "G:/DefectDataCenter/ParseData/Detection/"+proj_n+"/raw_data/Config/"+proj_n+"_Para.ini";
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

void MILTest::MILTestDetTrain()
{
	DET_TRAIN_STRUCT DtParas;
	DtParas.TrainMode = 0;
	DtParas.TrainEngineUsed = 0;
	DtParas.MaxNumberOfEpoch = 50;
	DtParas.MiniBatchSize = 16;
	DtParas.SchedulerType = 0;
	DtParas.LearningRate = 0.001;
	DtParas.LearningRateDecay = 0.1;
	DtParas.SplitPercent = 90.0;
	DtParas.WorkSpaceDir = L"G:/DefectDataCenter/ParseData/Detection";
	DtParas.DataSetName = L"DSW";
	m_MLDetCNN->TrainModel( DtParas);
	return;
}

void MILTest::MILTestDetPredict()
{
	string proj = "DSW";
	MIL_STRING Mproj = L"DSW";
	string	SrcImgDir = "G:/DefectDataCenter/ParseData/Detection/"+ proj+ "/raw_data/TImg";
	m_MLDetCNN->m_DetDataSetPara.WorkingDataDir = "G:/DefectDataCenter/ParseData/Detection/" + proj + "/MIL_Data/";
	MIL_STRING TdDetCtxPath = L"G:/DefectDataCenter/ParseData/Detection/"+ Mproj +L"/MIL_Data/"+ Mproj +L".mclass";
	vector<DET_RESULT_STRUCT> vecDetResults;
	bool SaveRst = TRUE;
	m_MLDetCNN->PredictFolderImgs(SrcImgDir,TdDetCtxPath,vecDetResults,SaveRst);
}

void MILTest::MILTestValDetModel()
{
	string proj = "DSW"; //COT_Raw
	string ValDataInfoPath = "G:/DefectDataCenter/ParseData/Detection/"+ proj+"/raw_data/Config/Val.txt";
	MIL_STRING Mproj = L"DSW";
	MIL_STRING TdDetCtxPath = L"G:/DefectDataCenter/ParseData/Detection/" + Mproj + L"/MIL_Data/" + Mproj + L".mclass";
	string strPRResultPath = "G:/DefectDataCenter/ParseData/Detection/" + proj + "/MIL_Data/PresionRecall.txt";
	m_MLDetCNN->ValModel_AP_50( ValDataInfoPath, TdDetCtxPath, strPRResultPath);
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
//
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
//
//	ODNetResult.close();
//}

void MILTest::MILTestDetPredictMutiProcessSingle()
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

void MILTest::MILTestDetPredictMutiProcess(
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
	string StartFile = "G:/DefectDataCenter/ParseData/Detection/lslm_bmp/raw_data/"+ Index+"_StartPredictFile.txt";
	ODNetResult.open(StartFile, ios::out);
	ODNetResult << "Prepare Predict" << endl;
	ODNetResult.close();

	double calc_time = 0;
	string	SrcDir = "G:/DefectDataCenter/ParseData/Detection/" + strProject + "/raw_data/";
	string	SrcImgDir = SrcDir + "TImg_" + Index;
	m_MLDetCNN->m_AIParse->getFilesInFolder(SrcImgDir, ImgType, m_FilesInFolder);
	string DstRst = SrcDir + Index + "_ODNetResult.txt";
	MILTestDetPredictCore( TrainedCtx, SrcDir, DstRst,calc_time);
	int nFileNum = m_FilesInFolder.size();
	cout << "\nOpen Process_" << Index <<" nFileNum: "<< nFileNum<< " average calc_time time(s) is = " << calc_time << "\n" << endl; //输出时间（单位：ｓ）
}


void MILTest::MILTestDetPredictCore(MIL_UNIQUE_CLASS_ID& TestCtx,
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
	calc_time = (double)(t2.QuadPart - t1.QuadPart) / (double)tc.QuadPart / (double)nFileNum ;

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



void MILTest::mutiThreadPrepare()
{
	//读取待图片路径测路径



}

void MILTest::MILTestDetPredictMutiThreadCore()
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

void MILTest::MILTestONNXPredict()
{
	MIL_STRING TdONNXCtxName = MIL_TEXT("I:/MIL_AI/testMILAI/yolov4_weights_LMK.onnx");

	MIL_UNIQUE_CLASS_ID TestONNXCtx = MclassAlloc(m_MilSystem,M_CLASSIFIER_ONNX, M_DEFAULT, M_UNIQUE_ID);
	MclassImport(TdONNXCtxName,M_ONNX_FILE, TestONNXCtx, M_DEFAULT, M_DEFAULT, M_DEFAULT);
	MclassInquire(TestONNXCtx, M_DEFAULT_SOURCE_LAYER, M_SIZE_X + M_TYPE_MIL_INT, &m_InputSizeX);
	MclassInquire(TestONNXCtx, M_DEFAULT_SOURCE_LAYER, M_SIZE_Y + M_TYPE_MIL_INT, &m_InputSizeY);
	MclassInquire(TestONNXCtx, M_CONTEXT, M_NUMBER_OF_CLASSES + M_TYPE_MIL_INT, &m_ClassesNum);
	
	MIL_STRING ImagepATH = MIL_TEXT("I:/MIL_AI/testMILAI/LMK1.bmp");
	MIL_ID Image = MbufRestore(ImagepATH, m_MilSystem, M_NULL);
	MIL_INT m_ImageSizeX = MbufInquire(Image, M_SIZE_X, M_NULL);
	MIL_INT m_ImageSizeY = MbufInquire(Image, M_SIZE_Y, M_NULL);

	MIL_ID ImageReduce = MbufAllocColor(m_MilSystem, 3, m_InputSizeX, m_InputSizeY, M_FLOAT + 32, M_IMAGE + M_PROC, M_NULL);
	MimResize(Image, ImageReduce, M_FILL_DESTINATION, M_FILL_DESTINATION, M_DEFAULT);

	MIL_UNIQUE_CLASS_ID ClassRes = MclassAllocResult(m_MilSystem, M_PREDICT_ONNX_RESULT, M_DEFAULT, M_UNIQUE_ID);

	MclassControl(TestONNXCtx, M_DEFAULT, M_TARGET_IMAGE_SIZE_X, m_InputSizeX);
	MclassControl(TestONNXCtx, M_DEFAULT, M_TARGET_IMAGE_SIZE_Y, m_InputSizeY);
	MclassControl(TestONNXCtx, M_DEFAULT, M_TARGET_IMAGE_SIZE_BAND, 3);

	MclassPreprocess(TestONNXCtx, M_DEFAULT);
	MclassPredict(TestONNXCtx, ImageReduce, ClassRes, M_DEFAULT);

	MIL_INT NO = 0;
	MclassGetResult(ClassRes, M_GENERAL, M_NUMBER_OF_OUTPUTS+ M_TYPE_MIL_INT, &NO);

	vector<MIL_UINT8>ROut;
	vector<MIL_DOUBLE>Out;
	vector<MIL_INT>OutSp;
	for (int i = 0; i < NO; i++) {
		MclassGetResult(ClassRes, M_OUTPUT_INDEX(i), M_OUTPUT_RAW, ROut);
		MclassGetResult(ClassRes, M_OUTPUT_INDEX(i), M_OUTPUT_SHAPE, OutSp);
		MclassGetResult(ClassRes, M_OUTPUT_INDEX(i), M_OUTPUT_DATA, Out);
	}

}

void MILTest::MILTestKTtreedbscan()
{
	clock_t start, finish;
	double  duration;
	start = clock();

	MulDimPointCloud<double> cloud;
	//针对黑白图的DBSCAN
	double AspectRatioTHD = 3;
	MIL_INT InSizeX = 16;
	MIL_INT InSizeY = 16;
	string ImgDir = "G:/DefectDataCenter/Test/Src/90";
	vector<MIL_STRING> ImgPaths;
	m_MLClassCNN->m_AIParse->getFilesInFolder(ImgDir,  "bmp", ImgPaths);
	//分割出长宽比过大的图为非法图片，不参与训练
	vector<MIL_STRING> unefftImgPaths;
	vector<MIL_STRING> efftImgPaths;
	for (auto iter = ImgPaths.begin(); iter != ImgPaths.end(); iter++) {

		MIL_ID Image = MbufRestore(*iter, m_MilSystem, M_NULL);
		MIL_INT ImageSizeX = MbufInquire(Image, M_SIZE_X, M_NULL);
		MIL_INT ImageSizeY = MbufInquire(Image, M_SIZE_Y, M_NULL);
		MIL_INT ImageBand = MbufInquire(Image, M_SIZE_BAND, M_NULL);

		bool illegalImg = (double)max(ImageSizeX, ImageSizeY) / (double)min(ImageSizeX, ImageSizeY) > AspectRatioTHD;
		if (illegalImg || ImageBand>2) {
			unefftImgPaths.emplace_back(*iter);
			continue;
		}
		efftImgPaths.emplace_back(*iter);
		MIL_ID ImageReduce = MbufAlloc2d(m_MilSystem, InSizeX, InSizeY, 8 + M_UNSIGNED, M_IMAGE + M_PROC, M_NULL);
		MimResize(Image, ImageReduce, M_FILL_DESTINATION, M_FILL_DESTINATION, M_BILINEAR);

		vector<MIL_UINT8>ImgPixels;
		ImgPixels.resize(InSizeX * InSizeY * ImageBand);
		MbufGet2d(ImageReduce, 0, 0, InSizeX, InSizeY, &ImgPixels[0]);
		MulDimPointCloud<double>::DBPoint TmpPts;
		for (int i = 0; i < ImgPixels.size(); i++) {
			TmpPts.Array[i] = ImgPixels[i] / 1.0;
		}
		cloud.pts.emplace_back(TmpPts);
	}

	double radius = 1.2;
	int m_minPoints = 60;
	int clusterNum;
	vector<vector<int>> Labels;
	kdtree_dbscan<double>(cloud,radius, m_minPoints, Labels, clusterNum);

	finish = clock();
	duration = (double)(finish - start) / CLOCKS_PER_SEC;
	cout << "duration: " << duration << endl;
	////创建目的class的文件夹
	MIL_STRING DstImgDir = L"G:/DefectDataCenter/Test/ImgCluster/Cpp_SPA90/";
	m_MLClassCNN->CreateFolder(DstImgDir);

	for (int i = 0; i < clusterNum+1; i++) {
		m_MLClassCNN->CreateFolder(DstImgDir + to_wstring(i-1));
	}
	////遍历所有文件及结果，并将图片保存到相应的Index
	for (int i = 0; i < Labels.size(); i++) {
		vector<int> clst = Labels[i];
		for (int j = 0; j < clst.size(); j++) {
			MIL_STRING RawImagePath = efftImgPaths[clst[j]];
			MIL_STRING ClassNames = to_wstring(i);
			MIL_ID Image = MbufRestore(RawImagePath, m_MilSystem, M_NULL);
			string pth;
			m_MLClassCNN->m_AIParse->MIL_STRING2string(RawImagePath, pth);
			string::size_type iPos = pth.find_last_of('/') + 1;
			MIL_STRING ImageRawName = RawImagePath.substr(iPos, RawImagePath.length() - iPos);
			MIL_STRING DstRootPath = DstImgDir + ClassNames + MIL_TEXT("//") + ImageRawName;
			MbufExport(DstRootPath, M_BMP, Image);
			MbufFree(Image);
		}	
	}
}


//void MILTest::OpencvTest(MIL_ID& ImageReshape)
//{
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