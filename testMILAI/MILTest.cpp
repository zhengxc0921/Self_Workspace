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

void MILTest::OpencvTest(MIL_ID& ImageReshape)
{
	//opencv test
	string img_name = "G:/DefectDataCenter/ParseData/Detection/lslm/raw_data/TImg/lslm.bmp";
	cv::Mat image = imread(img_name, CV_LOAD_IMAGE_UNCHANGED);
	//image = imread(fn[i], IMREAD_GRAYSCALE);
	// 2. convert color space, opencv read the image in BGR
	Mat img_float;
	// convert to float format
	image.convertTo(img_float, CV_32F, 1.0 / 255);
	// 3. resize the image for resnet101 model
	Mat img_resize;
	resize(img_float, img_resize, Size(704, 512), INTER_CUBIC);

	int width = img_resize.cols;//获取图像宽度
	int height = img_resize.rows;//获取图像高度
	int channel = img_resize.channels();//获取通道数
	float* pNorlzBuffer = new float[(int)(width * height * 3)];
	//数组方法遍历
		for (int h = 0; h < height; h++) //height
		{
			for (int w = 0; w < width; w++) //width
			{
				if (channel == 3)//彩色图像
				{
					//bgr 是一个vector，包含三个通道的值
					Vec3f bgr = img_resize.at<Vec3f>(h, w);
	/*				bgr[0] = 255 - bgr[0];
					bgr[1] = 255 - bgr[1];
					bgr[2] = 255 - bgr[2];
					img_resize.at<Vec3f>(h, w) = bgr;*/
					//cout << " b: " << bgr[0] << " g: " << bgr[1] << " r: " << bgr[2] << endl;

					pNorlzBuffer[h  + w+0] = bgr[0];
					pNorlzBuffer[h   + w  + 1] = bgr[1];
					pNorlzBuffer[h   + w  + 2] = bgr[2];

				}

			}
		}

		ImageReshape = MbufAllocColor(m_MilSystem, 3, width, height, M_FLOAT + 32, M_IMAGE + M_PROC, M_NULL);
		MbufPut(ImageReshape, pNorlzBuffer);

		float* m_pResizeBufferOrgRGB = new float[(int)(width * height * 3)];
		MbufGetColor(ImageReshape, M_PLANAR, M_ALL_BANDS, m_pResizeBufferOrgRGB);
		int nPixelIndex = 0;
		long nResizeCount = width * height;
		for (int i = 0; i < nResizeCount; i++) {
		//M_PLANAR的图片存放格式为：RRRR...GGGG...BBBB
		//OpenCV的图片存放格式为：BGRBGR...BGR
		//AI模型使用OPENCV图像形式训练所得，故需要将MIL格式转换
		for (int j = 2; j >= 0; j--) {
		//pNorlzBuffer[nPixelIndex] = m_pResizeBufferOrgRGB[i + nResizeCount * j];    //R_OpenCv<--R_MIL
		//cout << "m_pResizeBufferOrgRGB[nPixelIndex]: " << m_pResizeBufferOrgRGB[nPixelIndex] << endl;
		//cout << "pNorlzBuffer[nPixelIndex]: " << pNorlzBuffer[nPixelIndex] << endl;

		nPixelIndex++;
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

	
	MIL_STRING DatasetName = L"/BaseDataSet.mclassd";
	m_MLClassCNN->ConstructDataContext(DataCtxParas, DataContext);
	//MIL_UNIQUE_CLASS_ID PreparedDataset = MclassAlloc(m_MilSystem, M_DATASET_IMAGES, M_DEFAULT, M_UNIQUE_ID);
	m_MLClassCNN->PrepareDataset(DataContext, Dataset, WorkingDataPath, DatasetName);

}

/// MILTestWKSPDataset
bool isFileExists_ifstream(string& name) {
	ifstream f(name.c_str());
	return f.good();
}

//Check Tag_ClasssIcon == DataSet.mclassd ClassIcon

//void MILTest::isTagSameClass(MIL_UNIQUE_CLASS_ID& PreparedDataset,
//	const vector<MIL_STRING>& TagClassIcons,
//	map<MIL_STRING,int>& TagClassIconsIndex,
//	bool& isTagSameClass) {
//	isTagSameClass = TRUE;
//	MIL_INT nClassesNum;
//	MIL_STRING BaseClassIcon;
//	vector<MIL_STRING>BaseClassIcons;
//	MclassInquire(PreparedDataset, M_CONTEXT, M_NUMBER_OF_CLASSES + M_TYPE_MIL_INT, &nClassesNum);
//	for (int i = 0; i < nClassesNum; i++)
//	{	
//		MclassInquire(PreparedDataset, M_CLASS_INDEX(i), M_CLASS_NAME, BaseClassIcon);
//		BaseClassIcons.emplace_back(BaseClassIcon);
//	}
//	int Cont = 0;
//	for (int i = 0; i < TagClassIcons.size(); i++) {
//		if (std::find(BaseClassIcons.begin(), BaseClassIcons.end(), TagClassIcons[i]) == BaseClassIcons.end())
//		{
//			TagClassIconsIndex.insert(pair<MIL_STRING, int>(TagClassIcons[i], nClassesNum+ Cont));
//			isTagSameClass = FALSE;
//			Cont++;
//		}
//		else {
//			int index = distance(BaseClassIcons.begin(), find(BaseClassIcons.begin(), BaseClassIcons.end(), TagClassIcons[i]));
//			TagClassIconsIndex.insert(pair<MIL_STRING, int>(TagClassIcons[i],index));
//		}
//	}
//
//}


//void MILTest::isTagSameClass(
//	vector<MIL_STRING>BaseClassIcons,
//	const vector<MIL_STRING>& TagClassIcons,
//	map<MIL_STRING, int>& TagClassIconsIndex) {
//
//	MIL_INT nClassesNum;
//	MIL_STRING BaseClassIcon;
//	vector<MIL_STRING>BaseClassIcons;
//	
//	int Cont = 0;
//	for (int i = 0; i < TagClassIcons.size(); i++) {
//		if (std::find(BaseClassIcons.begin(), BaseClassIcons.end(), TagClassIcons[i]) == BaseClassIcons.end())
//		{
//			TagClassIconsIndex.insert(pair<MIL_STRING, int>(TagClassIcons[i], nClassesNum + Cont));
//			Cont++;
//		}
//		else {
//			int index = distance(BaseClassIcons.begin(), find(BaseClassIcons.begin(), BaseClassIcons.end(), TagClassIcons[i]));
//			TagClassIconsIndex.insert(pair<MIL_STRING, int>(TagClassIcons[i], index));
//		}
//	}
//
//}


void MILTest::MILTestWKSPDataset(MIL_STRING TagFolder)
{
	MIL_DOUBLE ValRatio = 0.1;

	MIL_STRING AuthorName = MIL_TEXT("AA");
	MIL_STRING BaseDataDir = m_ClassifierWorkSpace + m_strProject + L"/DataSet/" + L"/";
	MIL_STRING BaseDataPath = m_ClassifierWorkSpace + m_strProject + L"/DataSet/" + L"/BaseDataSet.mclassd";

	m_MLClassCNN->CreateFolder(m_ClassifierWorkSpace + m_strProject + L"/DataSet/");

	MIL_STRING TagDataDir = m_TagDataDir + TagFolder;
	string strBaseDataPath;
	m_MLClassCNN->m_AIParse->MIL_STRING2string(BaseDataPath,strBaseDataPath);
	bool DataSetExist = isFileExists_ifstream(strBaseDataPath);

	vector<MIL_STRING> TagClassNames;
	string strTagPath;
	m_MLClassCNN->m_AIParse->MIL_STRING2string(TagDataDir, strTagPath);
	m_MLClassCNN->m_AIParse->getFoldersInFolder(strTagPath, TagClassNames);

	//PrePared DataContext
		////*******************************必须参数*******************************//
	MIL_UNIQUE_CLASS_ID BaseDataContext = MclassAlloc(m_MilSystem, M_PREPARE_IMAGES_CNN, M_DEFAULT, M_UNIQUE_ID);
	MIL_UNIQUE_CLASS_ID UpdateDataContext = MclassAlloc(m_MilSystem, M_PREPARE_IMAGES_CNN, M_DEFAULT, M_UNIQUE_ID);
	DataContextParasStruct DataCtxParas;
	DataCtxParas.ImageSizeX = 128;
	DataCtxParas.ImageSizeY = 128;
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
	DataCtxParas.PreparedDataFolder = m_ClassifierWorkSpace  + m_strProject + L"/DataSet/preparedBaseData/";
	m_MLClassCNN->ConstructDataContext(DataCtxParas, BaseDataContext);

	DataCtxParas.PreparedDataFolder = m_ClassifierWorkSpace + m_strProject + L"/DataSet/preparedUpdateData/";
	m_MLClassCNN->ConstructDataContext(DataCtxParas, UpdateDataContext);
	//MIL_DOUBLE dSampleRatio;
	MIL_UNIQUE_CLASS_ID BaseDataSet = MclassAlloc(m_MilSystem, M_DATASET_IMAGES, M_DEFAULT, M_UNIQUE_ID);
	MIL_UNIQUE_CLASS_ID UpdateDataSet = MclassAlloc(m_MilSystem, M_DATASET_IMAGES, M_DEFAULT, M_UNIQUE_ID);
	if (DataSetExist) {	
		vector<MIL_DOUBLE>vecSampleRatio = {0.25,0.25};
		m_MLClassCNN->ConstructMergeDataset(AuthorName,BaseDataDir,TagDataDir,vecSampleRatio, BaseDataSet, UpdateDataSet);
		//生成PreParedUpdateDataSet、PreParedUpdateDataSet
		m_MLClassCNN->PrepareDataset(BaseDataContext, BaseDataSet , BaseDataDir, L"PreParedBaseDataSet");
		m_MLClassCNN->PrepareDataset(UpdateDataContext, UpdateDataSet, BaseDataDir, L"PreParedUpdateDataSet");
	}
	else {

		//以下内容生成BaseDataSet 、UpdateDataSet；UpdateDataSet==BaseDataSet
		vector<MIL_STRING > TagClassIcons;
		for (int i = 0; i < TagClassNames.size(); i++) {
			TagClassIcons.emplace_back(TagDataDir + TagClassNames[i] + L".mim");
		}
		m_MLClassCNN->ConstructDataset(TagClassNames, TagClassIcons, AuthorName, TagDataDir, BaseDataDir, BaseDataSet);
		MclassControl(BaseDataSet, M_CONTEXT, M_CONSOLIDATE_ENTRIES_INTO_FOLDER, BaseDataDir);
		MIL_STRING BaseDatasetPath = BaseDataDir + MIL_TEXT("BaseDataSet.mclassd");
		MIL_STRING UpdateDatasetPath = BaseDataDir + MIL_TEXT("UpdateDataSet.mclassd");
		MclassSave(BaseDatasetPath, BaseDataSet, M_DEFAULT);
		MclassSave(UpdateDatasetPath, BaseDataSet, M_DEFAULT);
		m_MLClassCNN->PrepareDataset(BaseDataContext, BaseDataSet, BaseDataDir, L"PreParedBaseDataSet");
	}

	
}

void MILTest::MILTestWKSPTrain()
{
	int MaxNumberOfEpoch = 20;			//模型训练次数
	int MiniBatchSize = 64;				//模型训练单次迭代的张数

	//////*******************************必须参数*******************************//
	MIL_STRING PreparedPath = m_ClassifierWorkSpace + m_strProject + MIL_TEXT("/DataSet/PreParedBaseDataSet.mclassd");
	MIL_UNIQUE_CLASS_ID PreparedDataset = MclassRestore(PreparedPath, m_MilSystem, M_DEFAULT, M_UNIQUE_ID);
	MIL_UNIQUE_CLASS_ID TrainCtx = MclassAlloc(m_MilSystem, M_TRAIN_CNN, M_DEFAULT, M_UNIQUE_ID);
	ClassifierParasStruct ClassifierParas;
	ClassifierParas.TrainMode = 0;
	ClassifierParas.TrainEngineUsed = 0;
	ClassifierParas.MaxNumberOfEpoch = MaxNumberOfEpoch;
	ClassifierParas.MiniBatchSize = MiniBatchSize;
	ClassifierParas.SchedulerType = 0;
	ClassifierParas.LearningRate = 0.0001 / 1; //normal:0.0001; 
	ClassifierParas.LearningRateDecay = 0;
	ClassifierParas.SplitPercent = 80.0;
	ClassifierParas.TrainDstFolder = m_ClassifierWorkSpace + m_strProject + MIL_TEXT("/PreparedData/");
	m_MLClassCNN->ConstructTrainCtx(ClassifierParas, TrainCtx);

	MIL_UNIQUE_CLASS_ID TrainedClassifierCtx;
	//MIL_STRING TrainedCtxFile = m_ClassifierWorkSpace + MIL_TEXT("FZ/PreparedData/") + L"FZ.mclass";
	//MIL_UNIQUE_CLASS_ID PrevClassifierCtx = MclassRestore(TrainedCtxFile, m_MilSystem, M_DEFAULT, M_UNIQUE_ID);
	MIL_UNIQUE_CLASS_ID PrevClassifierCtx;
	MIL_STRING ClassifierDumpFile = ClassifierParas.TrainDstFolder + m_strProject + L".mclass";
	m_MLClassCNN->TrainClassifier(PreparedDataset, TrainCtx, PrevClassifierCtx, TrainedClassifierCtx, ClassifierDumpFile);
	return;

}

void MILTest::MILTestWKSPUpdate()
{
	int MaxNumberOfEpoch = 20;			//模型训练次数
	int MiniBatchSize = 16;				//模型训练单次迭代的张数

	//////*******************************必须参数*******************************//
	MIL_STRING PreparedPath = m_ClassifierWorkSpace + m_strProject + MIL_TEXT("/DataSet/PreParedUpdateDataSet.mclassd");
	MIL_UNIQUE_CLASS_ID PreparedDataset = MclassRestore(PreparedPath, m_MilSystem, M_DEFAULT, M_UNIQUE_ID);
	MIL_UNIQUE_CLASS_ID TrainCtx = MclassAlloc(m_MilSystem, M_TRAIN_CNN, M_DEFAULT, M_UNIQUE_ID);
	ClassifierParasStruct ClassifierParas;
	ClassifierParas.TrainMode = 0;
	ClassifierParas.TrainEngineUsed = 0;
	ClassifierParas.MaxNumberOfEpoch = MaxNumberOfEpoch;
	ClassifierParas.MiniBatchSize = MiniBatchSize;
	ClassifierParas.SchedulerType = 0;
	ClassifierParas.LearningRate = 0.001; //normal:0.0001; 
	ClassifierParas.LearningRateDecay = 0;
	ClassifierParas.SplitPercent = 80.0;
	ClassifierParas.TrainDstFolder = m_ClassifierWorkSpace + m_strProject + MIL_TEXT("/PreparedData/");

	m_MLClassCNN->ConstructTrainCtx(ClassifierParas, TrainCtx);

	MIL_UNIQUE_CLASS_ID TrainedClassifierCtx;
	MIL_STRING TrainedCtxFile = m_ClassifierWorkSpace + MIL_TEXT("FZ/PreparedData/") + L"FZ.mclass";
	MIL_UNIQUE_CLASS_ID PrevClassifierCtx = MclassRestore(TrainedCtxFile, m_MilSystem, M_DEFAULT, M_UNIQUE_ID);
	//MIL_UNIQUE_CLASS_ID PrevClassifierCtx;
	MIL_STRING ClassifierDumpFile = ClassifierParas.TrainDstFolder + m_strProject + L".mclass";
	m_MLClassCNN->TrainClassifier(PreparedDataset, TrainCtx, PrevClassifierCtx, TrainedClassifierCtx, ClassifierDumpFile);
	return;

}

void MILTest::MILTestTrain()
{

	int MaxNumberOfEpoch = 15;			//模型训练次数
	int MiniBatchSize = 8;				//模型训练单次迭代的张数

	//////*******************************必须参数*******************************//
	MIL_STRING PreparedPath = m_ClassifierWorkSpace +m_strProject+ MIL_TEXT("/WorkingDataset.mclassd");				
	MIL_UNIQUE_CLASS_ID PreparedDataset = MclassRestore(PreparedPath, m_MilSystem, M_DEFAULT, M_UNIQUE_ID);
	MIL_UNIQUE_CLASS_ID TrainCtx = MclassAlloc(m_MilSystem, M_TRAIN_CNN, M_DEFAULT, M_UNIQUE_ID);  
	ClassifierParasStruct ClassifierParas;
	ClassifierParas.TrainMode = 0;
	ClassifierParas.TrainEngineUsed = 0;
	ClassifierParas.MaxNumberOfEpoch = MaxNumberOfEpoch;
	ClassifierParas.MiniBatchSize = MiniBatchSize;
	ClassifierParas.SchedulerType = 0;
	ClassifierParas.LearningRate = 0.0001/1; //normal:0.0001; 
	ClassifierParas.LearningRateDecay = 0;
	ClassifierParas.SplitPercent = 80.0;
	ClassifierParas.TrainDstFolder = m_ClassifierWorkSpace+ m_strProject + MIL_TEXT("/PreparedData/");
	m_MLClassCNN->ConstructTrainCtx(ClassifierParas, TrainCtx);

	MIL_UNIQUE_CLASS_ID TrainedClassifierCtx;
	MIL_STRING TrainedCtxFile = m_ClassifierWorkSpace  + MIL_TEXT("FZ/PreparedData/") + L"FZ.mclass";
	MIL_UNIQUE_CLASS_ID PrevClassifierCtx = MclassRestore(TrainedCtxFile, m_MilSystem, M_DEFAULT, M_UNIQUE_ID);
	//MIL_UNIQUE_CLASS_ID PrevClassifierCtx;
	MIL_STRING ClassifierDumpFile = ClassifierParas.TrainDstFolder + m_strProject+L".mclass";
	m_MLClassCNN->TrainClassifier(PreparedDataset, TrainCtx, PrevClassifierCtx, TrainedClassifierCtx, ClassifierDumpFile);
	return;
}

void MILTest::MILTestPredict(MIL_STRING TagFolder) {
	MIL_STRING PreClassifierName = m_ClassifierWorkSpace + m_strProject + MIL_TEXT("/PreparedData/") + m_strProject + L".mclass";
	MIL_STRING	SrcImgDir = L"G:/DefectDataCenter/WorkSpace/Test/"+m_strProject+L"/" + TagFolder;
	m_DstImgDir = L"G:/DefectDataCenter/WorkSpace/Test/" + m_strProject+L"/rst"+ TagFolder;


	LARGE_INTEGER t1, t2, tc;
	QueryPerformanceFrequency(&tc);
	QueryPerformanceCounter(&t1);
	MIL_UNIQUE_CLASS_ID TestCtx = MclassRestore(PreClassifierName, m_MilSystem, M_DEFAULT, M_UNIQUE_ID);
	getModelInfo(TestCtx);
	//设置ENGINE
	MIL_STRING Description;
	MclassInquire(TestCtx, M_PREDICT_ENGINE_INDEX(0), M_PREDICT_ENGINE_DESCRIPTION, Description);
	MosPrintf(MIL_TEXT("\nM_PREDICT_ENGINE_DESCRIPTION: %s \n"), Description.c_str());
	
	string strSrcImgDir;
	m_MLClassCNN->m_AIParse->MIL_STRING2string(SrcImgDir, strSrcImgDir);
	m_MLClassCNN->m_AIParse->getFilesInFolder(strSrcImgDir, "bmp", m_FilesInFolder);
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

void MILTest::MILTestGenDetDataset()
{
	////CASE1:
	//string IconInfo = "I:/MIL_Detection_Dataset/DSW/classes.txt";
	//MIL_STRING IconDir = L"I:/MIL_Detection_Dataset/DSW/Dataset/Icons/";
	//string ImgDataInfo = "I:/MIL_Detection_Dataset/DSW/train_box2.txt";
	//const MIL_STRING WorkingDataPath = L"I:/MIL_Detection_Dataset/DSW_/";

	//CASE2:
	string strProject = "DSW_random";

	string strSrcDir = "G:/DefectDataCenter/ParseData/Detection/"+ strProject + "/raw_data/" ;
	string IconInfo = strSrcDir +"Classes.txt";
	string IconDir = strSrcDir + "ClassesIcon/";
	string TrainImgDataInfo = strSrcDir + "ImgBoxes_train.txt";
	string ValImgDataInfo = strSrcDir + "ImgBoxes_val.txt";

	MIL_STRING WorkingDataPath = m_DetectionWorkSpace + m_strProject + L"//";

//string strProject = "VOC/";
//	string strSrcDir = "I:/MIL_Detection_Dataset/" + strProject;
//	string IconInfo = strSrcDir +"Classes.txt";
//	string IconDir = strSrcDir + "ClassesIcon/";
//	string ImgDataInfo = strSrcDir + "ImgBoxes.txt";

	MIL_UNIQUE_CLASS_ID  WorkingDataset = MclassAlloc(m_MilSystem, M_DATASET_IMAGES, M_DEFAULT, M_UNIQUE_ID);
	m_MLDetCNN->ConstructDataset(IconInfo, IconDir, TrainImgDataInfo, WorkingDataPath,"TrainDataSet.mclassd", WorkingDataset);

	int ImageSizeX = 1440;				//进入模型训练的图片的尺寸宽360*2 ;704
	int ImageSizeY = 1080;				//进入模型训练的图片的尺寸高270*2  ;512
	int AugmentationNumPerImage = 0;	//进入模型训练的图片的扩充倍数
	MIL_DOUBLE TestDatasetPercentage = 10;
	string strSrcImgDir = "G:/DefectDataCenter/ParseData/Detection/" + strProject + "/MIL_Data/";
	MIL_STRING SrcImgDir = m_MLDetCNN->m_AIParse->string2MIL_STRING(strSrcImgDir);
	////*******************************必须参数*******************************//
	MIL_UNIQUE_CLASS_ID DataContext = MclassAlloc(m_MilSystem, M_PREPARE_IMAGES_DET, M_DEFAULT, M_UNIQUE_ID);

	DataContextParasStruct DataCtxParas;
	DataCtxParas.ImageSizeX = ImageSizeX;
	DataCtxParas.ImageSizeY = ImageSizeY;
	DataCtxParas.DstFolderMode = 1;
	DataCtxParas.PreparedDataFolder = SrcImgDir + MIL_TEXT("PreparedData\\");
	memset(&DataCtxParas.AugParas, 0, sizeof(AugmentationParasStruct));
	DataCtxParas.AugParas.AugmentationNumPerImage = AugmentationNumPerImage;
	m_MLDetCNN->ConstructDataContext(DataCtxParas, DataContext);
	MIL_UNIQUE_CLASS_ID PreparedDataset = MclassAlloc(m_MilSystem, M_DATASET_IMAGES, M_DEFAULT, M_UNIQUE_ID);
	m_MLDetCNN->PrepareDataset(DataContext, WorkingDataset, PreparedDataset, WorkingDataPath, TestDatasetPercentage);
}

void MILTest::MILTestDetTrain()
{
	int MaxNumberOfEpoch = 3;			//模型训练次数
	int MiniBatchSize = 8;				//模型训练单次迭代的张数
	MIL_UNIQUE_CLASS_ID TrainCtx = MclassAlloc(m_MilSystem, M_TRAIN_DET, M_DEFAULT, M_UNIQUE_ID);  //网络类型定义1，默认形式：M_TRAIN_CNN？
	DetParas DtParas;
	DtParas.TrainMode = 0;
	DtParas.TrainEngineUsed = 0;
	DtParas.MaxNumberOfEpoch = MaxNumberOfEpoch;
	DtParas.MiniBatchSize = MiniBatchSize;
	DtParas.SchedulerType = 0;
	DtParas.LearningRate = 0.001;
	DtParas.LearningRateDecay = 0.1;
	DtParas.SplitPercent = 90.0;
	DtParas.TrainDstFolder = m_DetectionWorkSpace + L"//" + m_strProject + MIL_TEXT("/PreparedData/");
	m_MLDetCNN->CreateFolder(DtParas.TrainDstFolder);
	m_MLDetCNN->ConstructTrainCtx(DtParas, TrainCtx);
	MIL_UNIQUE_CLASS_ID TrainedDetCtx;
	MIL_STRING DetDumpFile = DtParas.TrainDstFolder + m_strProject + L".mclass";
	MIL_STRING PreparedPath = m_DetectionWorkSpace + m_strProject + MIL_TEXT("/WorkingDataset.mclassd");
	MIL_UNIQUE_CLASS_ID PreparedDataset = MclassRestore(PreparedPath, m_MilSystem, M_DEFAULT, M_UNIQUE_ID);
	m_MLDetCNN->TrainClassifier(PreparedDataset, TrainCtx, TrainedDetCtx, DetDumpFile);
	return;
}

void MILTest::MILTestDetPredict()
{
	string ImgType = "bmp";
	string strProject = "DSW_random";
	//string ImgType = "jpg";
	//string strProject = "VOC";
	string	SrcDir = "G:/DefectDataCenter/ParseData/Detection/" + strProject + "/raw_data/";
	string	SrcImgDir = SrcDir  + "TImg";
	MIL_STRING TdDetCtxName = m_DetectionWorkSpace + m_strProject + MIL_TEXT("/PreparedData/") + m_strProject + L".mclass";
	LARGE_INTEGER t1, t2, tc;
	QueryPerformanceFrequency(&tc);
	QueryPerformanceCounter(&t1);
	MIL_UNIQUE_CLASS_ID TestCtx = MclassRestore(TdDetCtxName, m_MilSystem, M_DEFAULT, M_UNIQUE_ID);

	MclassInquire(TestCtx, M_DEFAULT_SOURCE_LAYER, M_SIZE_X + M_TYPE_MIL_INT, &m_InputSizeX);
	MclassInquire(TestCtx, M_DEFAULT_SOURCE_LAYER, M_SIZE_Y + M_TYPE_MIL_INT, &m_InputSizeY);
	MclassInquire(TestCtx, M_CONTEXT, M_NUMBER_OF_CLASSES + M_TYPE_MIL_INT, &m_ClassesNum);

	//MclassControl(TestCtx, M_CONTEXT, M_DEFAULT_PREDICT_ENGINE_PRECISION, M_FP16);
	//设置ENGINE
	MIL_INT engine_index = 2;
	MIL_STRING Description;
	MclassControl(TestCtx, M_DEFAULT, M_PREDICT_ENGINE, engine_index);
	MclassInquire(TestCtx, M_PREDICT_ENGINE_INDEX(engine_index), M_PREDICT_ENGINE_DESCRIPTION, Description);
	MosPrintf(MIL_TEXT("\nM_PREDICT_ENGINE_DESCRIPTION: %s \n"), Description.c_str());
	vector<string>FilesInFolder;
	m_MLDetCNN->m_AIParse->getFilesInFolder(SrcImgDir, ImgType, m_FilesInFolder);
	m_MLDetCNN->m_AIParse->getFilesInFolder(SrcImgDir, ImgType, FilesInFolder);
	//读取到内存时间不计入
	int nFileNum = m_FilesInFolder.size();
	vector<DetResult> vecDetResults;
	vector<MIL_ID>RawImageS;
	for (int i = 0; i < nFileNum; i++) {
		MIL_ID RawImage = MbufRestore(m_FilesInFolder[i], m_MilSystem, M_NULL);
		RawImageS.emplace_back(RawImage);
	}
	m_MLDetCNN->FolderImgsPredict(RawImageS, TestCtx, vecDetResults);
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
	vector<DetResult> vecDetResults;
	vector<MIL_ID>RawImageS;
	for (int i = 0; i < nFileNum; i++) {
		MIL_ID RawImage = MbufRestore(m_FilesInFolder[i], m_MilSystem, M_NULL);
		RawImageS.emplace_back(RawImage);
	}


	LARGE_INTEGER t1, t2, tc;
	QueryPerformanceFrequency(&tc);
	QueryPerformanceCounter(&t1);

	int CirN = 1;
	for (int j = 0; j < CirN; j++) {
		m_MLDetCNN->FolderImgsPredict(RawImageS, TestCtx, vecDetResults);
	}

	QueryPerformanceCounter(&t2);
	calc_time = (double)(t2.QuadPart - t1.QuadPart) / (double)tc.QuadPart / (double)nFileNum / (double)CirN;

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
	DetResult Result_i;
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
	MIL_STRING ImagepATH = MIL_TEXT("I:/MIL_AI/testMILAI/lslm.bmp");
	MIL_ID Image = MbufRestore(ImagepATH, m_MilSystem, M_NULL);
	MIL_INT m_ImageSizeX = MbufInquire(Image, M_SIZE_X, M_NULL);
	MIL_INT m_ImageSizeY = MbufInquire(Image, M_SIZE_Y, M_NULL);
	//float* pNorlzBuffer = new float[(int)(m_ImageSizeX * m_ImageSizeY * 3)];
	MIL_STRING TdONNXCtxName = MIL_TEXT("I:/MIL_AI/testMILAI/yolov4_tiny_weights_lslm_b.onnx");
	MIL_UNIQUE_CLASS_ID TestONNXCtx = MclassAlloc(m_MilSystem,M_CLASSIFIER_ONNX, M_DEFAULT, M_UNIQUE_ID);
	MclassImport(TdONNXCtxName,M_ONNX_FILE, TestONNXCtx, M_DEFAULT, M_DEFAULT, M_DEFAULT);
	MclassInquire(TestONNXCtx, M_DEFAULT_SOURCE_LAYER, M_SIZE_X + M_TYPE_MIL_INT, &m_InputSizeX);
	MclassInquire(TestONNXCtx, M_DEFAULT_SOURCE_LAYER, M_SIZE_Y + M_TYPE_MIL_INT, &m_InputSizeY);
	MclassInquire(TestONNXCtx, M_CONTEXT, M_NUMBER_OF_CLASSES + M_TYPE_MIL_INT, &m_ClassesNum);
	MIL_ID ImageReduce = MbufAllocColor(m_MilSystem, 3, m_InputSizeX, m_InputSizeY, M_FLOAT + 32, M_IMAGE + M_PROC, M_NULL);
	//MimResize(Image, ImageReduce, M_FILL_DESTINATION, M_FILL_DESTINATION, M_BICUBIC);
	//MimArith(ImageReduce, 255.0, ImageReduce, M_DIV_CONST);
	MimArith(Image, 255.0, ImageReduce, M_DIV_CONST);

	MclassControl(TestONNXCtx, M_DEFAULT, M_TARGET_IMAGE_SIZE_X, m_InputSizeX);
	MclassControl(TestONNXCtx, M_DEFAULT, M_TARGET_IMAGE_SIZE_Y, m_InputSizeY);
	MclassControl(TestONNXCtx, M_DEFAULT, M_TARGET_IMAGE_SIZE_BAND, 3);
	MclassPreprocess(TestONNXCtx, M_DEFAULT);
	MIL_UNIQUE_CLASS_ID ClassRes = MclassAllocResult(m_MilSystem, M_PREDICT_ONNX_RESULT, M_DEFAULT, M_UNIQUE_ID);
	MclassPredict(TestONNXCtx, ImageReduce, ClassRes, M_DEFAULT);
	MIL_INT NO = 0;
	MclassGetResult(ClassRes, M_GENERAL, M_NUMBER_OF_OUTPUTS + M_TYPE_MIL_INT, &NO);
	vector<MIL_UINT8>ROut;
	vector<MIL_FLOAT>Out;
	vector<MIL_INT>OutSp;
	for (int i = 0; i < 1; i++) {
		MclassGetResult(ClassRes, M_OUTPUT_INDEX(i), M_OUTPUT_RAW, ROut);
		MclassGetResult(ClassRes, M_OUTPUT_INDEX(i), M_OUTPUT_SHAPE, OutSp);
		MclassGetResult(ClassRes, M_OUTPUT_INDEX(i), M_OUTPUT_DATA, Out);
		//cout << Out[0] << endl;
	}

	//vector<MIL_DOUBLE> m_pRGB;
	//MbufGetColor(ImageReduce, M_PLANAR, M_ALL_BANDS, m_pRGB);

	//MIL_ID OpencvImageReduce;
	//OpencvTest(OpencvImageReduce);

	//float* pNorlzBuffer = new float[(int)(m_InputSizeX * m_InputSizeY * 3)];
	//float* m_pResizeBufferOrgRGB = new float[(int)(m_InputSizeX * m_InputSizeY * 3)];
	//MbufGetColor(ImageReduce, M_PACKED + M_BGR24, M_ALL_BANDS, m_pResizeBufferOrgRGB);
	//MbufGetColor(ImageReduce, M_PLANAR , M_ALL_BANDS, m_pResizeBufferOrgRGB);
	//int nPixelIndex = 0;
	//long nResizeCount = m_InputSizeX * m_InputSizeY;
	//for (int i = 0; i < nResizeCount; i++) {
	//	//M_PLANAR的图片存放格式为：RRRR...GGGG...BBBB
	//	//OpenCV的图片存放格式为：BGRBGR...BGR
	//	//AI模型使用OPENCV图像形式训练所得，故需要将MIL格式转换
	//	for (int j = 2; j >= 0; j--) {
	//		pNorlzBuffer[nPixelIndex] = m_pResizeBufferOrgRGB[i + nResizeCount * j];    //R_OpenCv<--R_MIL

	//		cout << "pNorlzBuffer[nPixelIndex]: " 
	//			<<float(pNorlzBuffer[nPixelIndex]) << endl;
	//	nPixelIndex++;
	//	}

	//	//for (int j = 0; j <= 2; j++) {
	//	//	pNorlzBuffer[nPixelIndex] = m_pResizeBufferOrgRGB[i + nResizeCount * j];    //R_OpenCv<--R_MIL
	//	//	nPixelIndex++;
	//	//}
	//}



	//int* pNorlzBuffer = new int[(int)(m_InputSizeX * m_InputSizeY * 3)];
	//int* m_pResizeBufferOrgRGB = new int[(int)(m_InputSizeX * m_InputSizeY * 3)];
	//MbufGetColor(Image, M_PLANAR, M_ALL_BANDS, m_pResizeBufferOrgRGB);
	//int nPixelIndex = 0;
	//long nResizeCount = m_InputSizeX * m_InputSizeY;
	//for (int i = 0; i < nResizeCount; i++) {
	//	//M_PLANAR的图片存放格式为：RRRR...GGGG...BBBB
	//	//OpenCV的图片存放格式为：BGRBGR...BGR
	//	//AI模型使用OPENCV图像形式训练所得，故需要将MIL格式转换
	//	for (int j = 2; j >= 0; j--) {
	//		pNorlzBuffer[nPixelIndex] = m_pResizeBufferOrgRGB[i + nResizeCount * j];    //R_OpenCv<--R_MIL
	//		//cout <<"pNorlzBuffer[nPixelIndex]: "<< pNorlzBuffer[nPixelIndex]*255 << endl;
	//		cout << "[nPixelIndex]: " << m_pResizeBufferOrgRGB[nPixelIndex] << endl;
	//		nPixelIndex++;
	//	}
	//}
	//MIL_ID ImageReshape = MbufAllocColor(m_MilSystem, 3, m_InputSizeX, m_InputSizeY, M_FLOAT + 32, M_IMAGE + M_PROC, M_NULL);
	//MbufPut(ImageReshape, pNorlzBuffer);

	//MclassControl(TestONNXCtx, M_DEFAULT, M_TARGET_IMAGE_SIZE_X, m_InputSizeX);
	//MclassControl(TestONNXCtx, M_DEFAULT, M_TARGET_IMAGE_SIZE_Y, m_InputSizeY);
	//MclassControl(TestONNXCtx, M_DEFAULT, M_TARGET_IMAGE_SIZE_BAND, 3);
	//MclassPreprocess(TestONNXCtx, M_DEFAULT);
	//MIL_UNIQUE_CLASS_ID ClassRes = MclassAllocResult(m_MilSystem, M_PREDICT_ONNX_RESULT, M_DEFAULT, M_UNIQUE_ID);
	//MclassPredict(TestONNXCtx, ImageReduce, ClassRes, M_DEFAULT);
	//MIL_INT NO = 0;
	//MclassGetResult(ClassRes, M_GENERAL, M_NUMBER_OF_OUTPUTS+ M_TYPE_MIL_INT, &NO);
	//vector<MIL_UINT8>ROut;
	//vector<MIL_FLOAT>Out;
	//vector<MIL_INT>OutSp;
	//for (int i = 0; i < 1; i++) {
	//	MclassGetResult(ClassRes, M_OUTPUT_INDEX(i), M_OUTPUT_RAW, ROut);
	//	MclassGetResult(ClassRes, M_OUTPUT_INDEX(i), M_OUTPUT_SHAPE, OutSp);
	//	MclassGetResult(ClassRes, M_OUTPUT_INDEX(i), M_OUTPUT_DATA, Out);
	//	//cout << Out[0] << endl;
	//}

	
}

void MILTest::MILTestCNNONNXPredict()
{
	MIL_STRING ImagepATH = MIL_TEXT("I:/MIL_AI/testMILAI/Mg_grain.bmp");
	MIL_ID Image = MbufRestore(ImagepATH, m_MilSystem, M_NULL);
	MIL_INT m_ImageSizeX = MbufInquire(Image, M_SIZE_X, M_NULL);
	MIL_INT m_ImageSizeY = MbufInquire(Image, M_SIZE_Y, M_NULL);
	/*float* pNorlzBuffer = new float[(int)(m_ImageSizeX * m_ImageSizeY * 3)];*/
	MIL_STRING TdONNXCtxName = MIL_TEXT("I:/MIL_AI/testMILAI/cpu_Mg_grain.onnx");
	MIL_UNIQUE_CLASS_ID TestONNXCtx = MclassAlloc(m_MilSystem, M_CLASSIFIER_ONNX, M_DEFAULT, M_UNIQUE_ID);
	MclassImport(TdONNXCtxName, M_ONNX_FILE, TestONNXCtx, M_DEFAULT, M_DEFAULT, M_DEFAULT);
	MclassInquire(TestONNXCtx, M_DEFAULT_SOURCE_LAYER, M_SIZE_X + M_TYPE_MIL_INT, &m_InputSizeX);
	MclassInquire(TestONNXCtx, M_DEFAULT_SOURCE_LAYER, M_SIZE_Y + M_TYPE_MIL_INT, &m_InputSizeY);
	MclassInquire(TestONNXCtx, M_CONTEXT, M_NUMBER_OF_CLASSES + M_TYPE_MIL_INT, &m_ClassesNum);
	MIL_ID ImageReduce = MbufAllocColor(m_MilSystem, 3, m_InputSizeX, m_InputSizeY, M_FLOAT + 32, M_IMAGE + M_PROC, M_NULL);
	//灰度图：
	//原图插值后与Pytorch在INTER_LINEAR插值后计算结果不一致，
	//而直接使用Pytorch插值后的图计算，结果一致精确度小数点后5位
	//MimResize(Image, ImageReduce, M_FILL_DESTINATION, M_FILL_DESTINATION, M_BILINEAR);
	/*MimArith(ImageReduce, 255.0, ImageReduce, M_DIV_CONST);*/
	//彩图：而直接使用Pytorch插值后的图计算，结果同样不一致

	MimArith(Image, 255.0, ImageReduce, M_DIV_CONST);

	MclassControl(TestONNXCtx, M_DEFAULT, M_TARGET_IMAGE_SIZE_X, m_InputSizeX);
	MclassControl(TestONNXCtx, M_DEFAULT, M_TARGET_IMAGE_SIZE_Y, m_InputSizeY);
	MclassControl(TestONNXCtx, M_DEFAULT, M_TARGET_IMAGE_SIZE_BAND, 3);
	MclassPreprocess(TestONNXCtx, M_DEFAULT);
	MIL_UNIQUE_CLASS_ID ClassRes = MclassAllocResult(m_MilSystem, M_PREDICT_ONNX_RESULT, M_DEFAULT, M_UNIQUE_ID);
	MclassPredict(TestONNXCtx, ImageReduce, ClassRes, M_DEFAULT);
	MIL_INT NO = 0;
	MclassGetResult(ClassRes, M_GENERAL, M_NUMBER_OF_OUTPUTS + M_TYPE_MIL_INT, &NO);
	vector<MIL_UINT8>ROut;
	vector<MIL_FLOAT>Out;
	vector<MIL_INT>OutSp;
	for (int i = 0; i < 1; i++) {
		MclassGetResult(ClassRes, M_OUTPUT_INDEX(i), M_OUTPUT_RAW, ROut);
		MclassGetResult(ClassRes, M_OUTPUT_INDEX(i), M_OUTPUT_SHAPE, OutSp);
		MclassGetResult(ClassRes, M_OUTPUT_INDEX(i), M_OUTPUT_DATA, Out);
		//cout << Out[0] << endl;
	}

}

