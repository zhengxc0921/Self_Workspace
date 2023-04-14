#include "MILTest.h"

MILTest::MILTest(MIL_ID MilSystem, MIL_ID MilDisplay):
	m_MilSystem(MilSystem),
	m_MilDisplay(MilDisplay)
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

void MILTest::MILTestGenDataset()
{
	MIL_STRING SrcImgDir = MIL_TEXT("G:/DefectDataCenter/李巨欣/ShenZhen/");  //原始数据根文件
	std::vector<MIL_STRING>OriginalDataPath = { SrcImgDir + MIL_TEXT("Original1_Crop\\")};
	std::vector<vector<MIL_STRING>>ClassName = { { MIL_TEXT("1"), MIL_TEXT("2") ,MIL_TEXT("3"),MIL_TEXT("99")} };
	vector<MIL_STRING>AuthorName = { MIL_TEXT("AA")  };
	MIL_STRING WorkingDataPath = SrcImgDir + MIL_TEXT("DataSet\\");				//原始数据根文件下的 存放中间数据的文件夹
	std::vector<vector<MIL_STRING>>ClassIcon;
	getIcon(OriginalDataPath, ClassName, ClassIcon);
	m_MLClassCNN->GeneralDataset(ClassName, ClassIcon, AuthorName, OriginalDataPath, WorkingDataPath);

}

void MILTest::MILTestTrain()
{
	int ImageSizeX = 64;				//进入模型训练的图片的尺寸宽
	int ImageSizeY = 64;				//进入模型训练的图片的尺寸高
	int AugmentationNumPerImage = 1;	//进入模型训练的图片的扩充倍数
	int MaxNumberOfEpoch = 50;			//模型训练次数
	int MiniBatchSize = 64;				//模型训练单次迭代的张数
	MIL_STRING ClassifierFileName = MIL_TEXT("SZCrop.mclass");
	MIL_STRING SrcImgDir = MIL_TEXT("G:/DefectDataCenter/李巨欣/ShenZhen/");

	////*******************************必须参数*******************************//
	MIL_UNIQUE_CLASS_ID DataContext = MclassAlloc(m_MilSystem, M_PREPARE_IMAGES_CNN, M_DEFAULT, M_UNIQUE_ID);

	DataContextParasStruct DataCtxParas;
	DataCtxParas.ImageSizeX = ImageSizeX;
	DataCtxParas.ImageSizeY = ImageSizeX;
	DataCtxParas.DstFolderMode = 1;
	DataCtxParas.PreparedDataFolder = SrcImgDir + MIL_TEXT("PreparedData\\");
	memset(&DataCtxParas.AugParas, 0, sizeof(AugmentationParasStruct));
	DataCtxParas.AugParas.AugmentationNumPerImage = AugmentationNumPerImage;


	//DataCtxParas.AugParas.GammaValue = 0.9;
	//DataCtxParas.AugParas.GammaDelta = 0.05;
	//DataCtxParas.AugParas.InAddValue = 30;
	//DataCtxParas.AugParas.InAddDelta = 5;
	DataCtxParas.AugParas.InMulValue = 1.1;
	DataCtxParas.AugParas.InMulDelta = 0.2;



	m_MLClassCNN->ConstructDataContext(DataCtxParas, DataContext);

	MIL_STRING WorkingDataPath = SrcImgDir + MIL_TEXT("DataSet\\DataSet.mclassd");				//原始数据根文件下的 存放中间数据的文件夹
	MIL_UNIQUE_CLASS_ID WorkingDataset = MclassRestore(WorkingDataPath, m_MilSystem, M_DEFAULT, M_UNIQUE_ID);
	MIL_UNIQUE_CLASS_ID PreparedDataset = MclassAlloc(m_MilSystem, M_DATASET_IMAGES, M_DEFAULT, M_UNIQUE_ID);
	m_MLClassCNN->PrepareDataset(DataContext, WorkingDataset, PreparedDataset);

	MIL_UNIQUE_CLASS_ID TrainCtx = MclassAlloc(m_MilSystem, M_TRAIN_CNN, M_DEFAULT, M_UNIQUE_ID);  //网络类型定义1，默认形式：M_TRAIN_CNN？
	ClassifierParasStruct ClassifierParas;

	ClassifierParas.TrainMode = 0;
	ClassifierParas.TrainEngineUsed = 0;
	ClassifierParas.MaxNumberOfEpoch = MaxNumberOfEpoch;
	ClassifierParas.MiniBatchSize = MiniBatchSize;
	ClassifierParas.SchedulerType = 0;
	ClassifierParas.LearningRate = 0.0001;
	ClassifierParas.LearningRateDecay = 0;
	ClassifierParas.SplitPercent = 90.0;
	ClassifierParas.TrainDstFolder = SrcImgDir + MIL_TEXT("PreparedData\\");
	m_MLClassCNN->ConstructTrainCtx(ClassifierParas, TrainCtx);

	MIL_UNIQUE_CLASS_ID TrainedClassifierCtx;
	MIL_UNIQUE_CLASS_ID PrevClassifierCtx;
	MIL_STRING ClassifierDumpFile = SrcImgDir + MIL_TEXT("PreparedData\\") + ClassifierFileName;
	m_MLClassCNN->TrainClassifier(PreparedDataset, DataContext, TrainCtx, PrevClassifierCtx, TrainedClassifierCtx, ClassifierDumpFile);
	return;

}

void MILTest::MILTestPredict() {

	//string	SrcImgDir = "G:/DefectDataCenter/武翔/WuX_Test/Lot_Gather/20_s";
	//m_DstImgDir = MIL_TEXT("G:/DefectDataCenter/武翔/WuX_Test/Lot_Gather/20_s_result/");
	//MIL_STRING PreClassifierName = MIL_TEXT("G:/DefectDataCenter/zhjuzhiqiang_2023/2023/FZ/PreparedData/FZ.mclass");

	string	SrcImgDir = "G:/DefectDataCenter/李巨欣/ShenZhen/Lot_Gather/ALL_Crop";
	m_DstImgDir = MIL_TEXT("G:/DefectDataCenter/李巨欣/ShenZhen/Lot_Gather/ALL_Crop_result/");
	MIL_STRING PreClassifierName = MIL_TEXT("G:/DefectDataCenter/李巨欣/ShenZhen/PreparedData/SZ.mclass");

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
//	m_DstImgDir = string2MIL_STRING(strDstImgDir);
//	m_CAIPrePtr = CAIProcPtr(new CAIProc(m_MilSystem, m_MilDisplay, TrainedCtx, index));
//
//	//设置预测引擎
//	MclassControl(TrainedCtx, M_DEFAULT, M_PREDICT_ENGINE, 2);
//	double calc_time = 0;
//	MILTestPredictCore(strSrcImgDir, TrainedCtx, calc_time);
//	cout << "\nOpen Process_" << index << " calc_time time(s) is = " << calc_time << "\n" << endl; //输出时间（单位：ｓ）
//}


void MILTest::MILTestGenDetDataset()
{
	////CASE1:
	//string IconInfo = "I:/MIL_Detection_Dataset/DSW/classes.txt";
	//MIL_STRING IconDir = L"I:/MIL_Detection_Dataset/DSW/Dataset/Icons/";
	//string ImgDataInfo = "I:/MIL_Detection_Dataset/DSW/train_box2.txt";
	//const MIL_STRING WorkingDataPath = L"I:/MIL_Detection_Dataset/DSW_/";

	//CASE2:
	string IconInfo = "I:/MIL_Detection_Dataset/VOC/raw_data/Classes.txt";
	MIL_STRING IconDir = L"I:/MIL_Detection_Dataset/VOC/raw_data/ClassesIcon/";
	string ImgDataInfo = "I:/MIL_Detection_Dataset/VOC/raw_data/ImgBoxes.txt";
	const MIL_STRING WorkingDataPath = L"I:/MIL_Detection_Dataset/VOC/";

	m_MLDetCNN->ConstructDataset(IconInfo, IconDir,  ImgDataInfo, WorkingDataPath);
}

void MILTest::MILTestDetTrain()
{
	int ImageSizeX = 512;				//进入模型训练的图片的尺寸宽
	int ImageSizeY = 512;				//进入模型训练的图片的尺寸高
	int AugmentationNumPerImage = 0;	//进入模型训练的图片的扩充倍数
	int MaxNumberOfEpoch = 50;			//模型训练次数
	int MiniBatchSize = 4;				//模型训练单次迭代的张数
	MIL_STRING DetFileName = MIL_TEXT("VOC_update.mclass");
	MIL_STRING SrcImgDir = MIL_TEXT("I:/MIL_Detection_Dataset/VOC/");

	////*******************************必须参数*******************************//
	MIL_UNIQUE_CLASS_ID DataContext = MclassAlloc(m_MilSystem, M_PREPARE_IMAGES_DET, M_DEFAULT, M_UNIQUE_ID);

	DataContextParasStruct DataCtxParas;
	DataCtxParas.ImageSizeX = ImageSizeX;
	DataCtxParas.ImageSizeY = ImageSizeY;
	DataCtxParas.DstFolderMode = 1;
	DataCtxParas.PreparedDataFolder = SrcImgDir + MIL_TEXT("PreparedData\\");
	memset(&DataCtxParas.AugParas, 0, sizeof(AugmentationParasStruct));
	DataCtxParas.AugParas.AugmentationNumPerImage = AugmentationNumPerImage;
	DataCtxParas.AugParas.InMulValue = 1.1;
	DataCtxParas.AugParas.InMulDelta = 0.2;

	m_MLDetCNN->ConstructDataContext(DataCtxParas, DataContext);

	MIL_STRING WorkingDataPath = SrcImgDir + MIL_TEXT("DataSet.mclassd");				//原始数据根文件下的 存放中间数据的文件夹
	MIL_UNIQUE_CLASS_ID WorkingDataset = MclassRestore(WorkingDataPath, m_MilSystem, M_DEFAULT, M_UNIQUE_ID);
	MIL_UNIQUE_CLASS_ID PreparedDataset = MclassAlloc(m_MilSystem, M_DATASET_IMAGES, M_DEFAULT, M_UNIQUE_ID);


	m_MLDetCNN->PrepareDataset(DataContext, WorkingDataset, PreparedDataset);

	MIL_UNIQUE_CLASS_ID TrainCtx = MclassAlloc(m_MilSystem, M_TRAIN_DET, M_DEFAULT, M_UNIQUE_ID);  //网络类型定义1，默认形式：M_TRAIN_CNN？
	DetParas DtParas;

	DtParas.TrainMode = 0;
	DtParas.TrainEngineUsed = 0;
	DtParas.MaxNumberOfEpoch = MaxNumberOfEpoch;
	DtParas.MiniBatchSize = MiniBatchSize;
	DtParas.SchedulerType = 0;
	DtParas.LearningRate = 0.0001;
	DtParas.LearningRateDecay = 0;
	DtParas.SplitPercent = 90.0;
	DtParas.TrainDstFolder = SrcImgDir + MIL_TEXT("PreparedData\\");
	m_MLDetCNN->ConstructTrainCtx(DtParas, TrainCtx);

	MIL_UNIQUE_CLASS_ID TrainedDetCtx;

	MIL_STRING PreDetCtxName = MIL_TEXT("I:/MIL_Detection_Dataset/VOC/PreparedData/VOC.mclass");
	MIL_UNIQUE_CLASS_ID PreDetCtx = MclassRestore(PreDetCtxName, m_MilSystem, M_DEFAULT, M_UNIQUE_ID);

	MIL_STRING DetDumpFile = SrcImgDir + MIL_TEXT("PreparedData\\") + DetFileName;
	m_MLDetCNN->TrainClassifier(PreparedDataset, DataContext, TrainCtx, PreDetCtx, TrainedDetCtx, DetDumpFile);
	return;


}

void MILTest::MILTestDetPredict()
{
	string	SrcDir = "I:/MIL_Detection_Dataset/VOC/";
	string	SrcImgDir = SrcDir+ "TImages/";
	MIL_STRING TdDetCtxName = MIL_TEXT("I:/MIL_Detection_Dataset/VOC/PreparedData/VOC.mclass");
	//string	SrcImgDir = "I:/MIL_Detection_Dataset/VOC/Images";
	//MIL_STRING TdDetCtxName = MIL_TEXT("I:/MIL_Detection_Dataset/VOC/PreparedData/VOC.mclass");
	LARGE_INTEGER t1, t2, tc;
	QueryPerformanceFrequency(&tc);
	QueryPerformanceCounter(&t1);
	MIL_UNIQUE_CLASS_ID TestCtx = MclassRestore(TdDetCtxName, m_MilSystem, M_DEFAULT, M_UNIQUE_ID);

	MclassInquire(TestCtx, M_DEFAULT_SOURCE_LAYER, M_SIZE_X + M_TYPE_MIL_INT, &m_InputSizeX);
	MclassInquire(TestCtx, M_DEFAULT_SOURCE_LAYER, M_SIZE_Y + M_TYPE_MIL_INT, &m_InputSizeY);
	MclassInquire(TestCtx, M_CONTEXT, M_NUMBER_OF_CLASSES + M_TYPE_MIL_INT, &m_ClassesNum);

	//设置ENGINE
	MIL_INT engine_index = 0;
	MIL_STRING Description;
	MclassControl(TestCtx, M_DEFAULT, M_PREDICT_ENGINE, engine_index);
	MclassInquire(TestCtx, M_PREDICT_ENGINE_INDEX(engine_index), M_PREDICT_ENGINE_DESCRIPTION, Description);
	MosPrintf(MIL_TEXT("\nM_PREDICT_ENGINE_DESCRIPTION: %s \n"), Description.c_str());

	vector<string>FilesInFolder;
	m_MLDetCNN->m_AIParse->getFilesInFolder(SrcImgDir, "jpg", m_FilesInFolder);
	m_MLDetCNN->m_AIParse->getFilesInFolder(SrcImgDir, "jpg", FilesInFolder);

	int nFileNum = m_FilesInFolder.size();
	vector<DetResult> vecDetResults;
	m_MLDetCNN->FolderImgsPredict(m_FilesInFolder, TestCtx, vecDetResults);

	//将结果保存到txt文件
	ofstream ODNetResult;
	ODNetResult.open(SrcDir+"ODNetResult.txt", ios::out);
	for (int i = 0; i < nFileNum; i++) {
		string ImgInfo;
		ImgInfo = FilesInFolder[i];
		//写入图片路径、box、conf、classname
		DetResult R_i = vecDetResults[i];
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
		ODNetResult << ImgInfo<<endl;
	}

	ODNetResult.close();
}

void MILTest::MILTestONNXPredict()
{
	MIL_STRING TdONNXCtxName = MIL_TEXT("I:/MIL_AI/testMILAI/Dsw_random.onnx");

	MIL_UNIQUE_CLASS_ID TestONNXCtx = MclassAlloc(m_MilSystem,M_CLASSIFIER_ONNX, M_DEFAULT, M_UNIQUE_ID);
	MclassImport(TdONNXCtxName,M_ONNX_FILE, TestONNXCtx, M_DEFAULT, M_DEFAULT, M_DEFAULT);
	MclassInquire(TestONNXCtx, M_DEFAULT_SOURCE_LAYER, M_SIZE_X + M_TYPE_MIL_INT, &m_InputSizeX);
	MclassInquire(TestONNXCtx, M_DEFAULT_SOURCE_LAYER, M_SIZE_Y + M_TYPE_MIL_INT, &m_InputSizeY);
	MclassInquire(TestONNXCtx, M_CONTEXT, M_NUMBER_OF_CLASSES + M_TYPE_MIL_INT, &m_ClassesNum);
	
	MIL_STRING ImagepATH = MIL_TEXT("I:/MIL_Detection_Dataset/DSW_random/raw_data/img/block_A_defect_A_cellcol_0.bmp");
	MIL_ID Image = MbufRestore(ImagepATH, m_MilSystem, M_NULL);
	MIL_INT m_ImageSizeX = MbufInquire(Image, M_SIZE_X, M_NULL);
	MIL_INT m_ImageSizeY = MbufInquire(Image, M_SIZE_Y, M_NULL);

	MIL_ID ImageReduce = MbufAllocColor(m_MilSystem, 3, m_InputSizeX, m_InputSizeY, 8 + M_UNSIGNED, M_IMAGE + M_PROC, M_NULL);
	MimResize(Image, ImageReduce, M_FILL_DESTINATION, M_FILL_DESTINATION, M_DEFAULT);

	MIL_UNIQUE_CLASS_ID ClassRes = MclassAllocResult(m_MilSystem, M_PREDICT_ONNX_RESULT, M_DEFAULT, M_UNIQUE_ID);

	MclassControl(TestONNXCtx, M_DEFAULT, M_TARGET_IMAGE_SIZE_X, m_InputSizeX);
	MclassControl(TestONNXCtx, M_DEFAULT, M_TARGET_IMAGE_SIZE_Y, m_InputSizeY);
	MclassControl(TestONNXCtx, M_DEFAULT, M_TARGET_IMAGE_SIZE_BAND, 3);

	MclassPreprocess(TestONNXCtx, M_DEFAULT);
	MclassPredict(TestONNXCtx, ImageReduce, ClassRes, M_DEFAULT);
	MIL_INT INStangs = 0;
	MclassGetResult(ClassRes, M_GENERAL, M_NUMBER_OF_INSTANCES + M_TYPE_MIL_INT, &INStangs);
}

