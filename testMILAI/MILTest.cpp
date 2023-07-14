#include "MILTest.h"

MILTest::MILTest(MIL_ID MilSystem, MIL_ID MilDisplay, MIL_STRING strProject):
	m_MilSystem(MilSystem),
	m_MilDisplay(MilDisplay),
	m_strProject(strProject)
{
	m_MLClassCNN = CMLClassCNNPtr(new CMLClassCNN( MilSystem,  MilDisplay));
	//m_MLDetCNN = CMLDetCNNPtr(new CMLDetCNN(MilSystem, MilDisplay));
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
	MIL_UNIQUE_CLASS_ID PreparedDataset = MclassAlloc(m_MilSystem, M_DATASET_IMAGES, M_DEFAULT, M_UNIQUE_ID);
	m_MLClassCNN->PrepareDataset(DataContext, Dataset, PreparedDataset, WorkingDataPath, DatasetName);

}



void MILTest::ReduceSimilarityImg()
{
	clock_t start, end, end1;
	start = clock();
	//数据源文件
	MIL_STRING BaseImgDir = L"G:/DefectDataCenter/原始_现场分类数据/LJX/TrainingDatsSet/SPA_ASI_Reclass_DataSet/Images/45";
	//MIL_STRING BaseImgDir = L"G:/DefectDataCenter/原始_现场分类数据/LJX/TrainingDatsSet/PO1_ASI_ReClass_DataSet/Images/91";
	MIL_STRING DstDir = L"G:/DefectDataCenter/Test/MIL_ImgCluster/SPA_ASI_Reclass_DataSet45/";

	vector<MIL_STRING>vecBaseImg;
	m_MLClassCNN->m_AIParse->getFilesInFolder(BaseImgDir, "bmp", vecBaseImg);
	
		MIL_INT ImgSizeX = 16;
		MIL_INT ImgSizeY = 16;
		MIL_INT SizeBAND = 1;
		MIL_ID BaseImage = MbufAlloc2d(m_MilSystem, ImgSizeX, ImgSizeY,8 + M_UNSIGNED,M_IMAGE + M_PROC,M_NULL);
		MbufClear(BaseImage, 0);
		vector<DBPoint>vecImgPixels;
		//vecBaseImg.size()
		int TNum = vecBaseImg.size();
		for (int j = 0; j < TNum; j++) {
			DBPoint DBpoint;
			MIL_ID baseImg = MbufRestore(vecBaseImg[j], m_MilSystem, M_NULL);
			MIL_INT bSizeX = MbufInquire(baseImg, M_SIZE_X, M_NULL);
			MIL_INT bSizeY = MbufInquire(baseImg, M_SIZE_Y, M_NULL);

			MimResize(baseImg, BaseImage, M_FILL_DESTINATION, M_FILL_DESTINATION, M_DEFAULT);

			//MIL_STRING DstRootPath = L"G:/DefectDataCenter/Test/Resize/a.bmp";
			//MbufExport(DstRootPath, M_BMP, BaseImage);

			vector<MIL_UINT8>ImgPixels;
			ImgPixels.resize(ImgSizeX* ImgSizeY* SizeBAND);
			MbufGet2d(BaseImage, 0, 0, ImgSizeX, ImgSizeY, &ImgPixels[0]);
			for (int i = 0; i < ImgPixels.size(); i++) {
				DBpoint.vecImgPixel.emplace_back(ImgPixels[i] / 255.0);
			}
			vecImgPixels.emplace_back(DBpoint);
		}
		unsigned int minPts = 15;
		float eps = 13;
		DBSCAN ds(minPts, eps, vecImgPixels);
		ds.run2();

		end1 = clock();
		cout << "TIME(SEC) " << static_cast<double>(end1 - start) / CLOCKS_PER_SEC << "\n";

		vector<MIL_STRING>DstFolder;
		m_MLClassCNN->CreateFolder(DstDir);
		for (int i = -1; i < ds.m_ClassNum; i++) {
			MIL_STRING Dst_folder = DstDir + m_MLClassCNN->m_AIParse->string2MIL_STRING(to_string(i))+L"//";
			m_MLClassCNN->CreateFolder(Dst_folder);
			DstFolder.emplace_back(Dst_folder);
		}


		for (int i = 0; i < TNum; i++) {
			int fi = ds.m_points[i].clusterID+2;
			MIL_STRING dst_path = DstFolder[fi]+ m_MLClassCNN->m_AIParse->string2MIL_STRING(to_string(i))+L".BMP";

			ifstream source(vecBaseImg[i], ios::binary);
			ofstream dest(dst_path, ios::binary);

			dest << source.rdbuf();

			source.close();
			dest.close();

		}

		end = clock();
		cout << "TIME(SEC) " << static_cast<double>(end - start) / CLOCKS_PER_SEC << "\n";

}

void MILTest::Pytest()
{
		Py_Initialize(); //初始化python解释器
		if (!Py_IsInitialized()) {
			std::system("pause");
			//return -99;
		} //查看python解释器是否成功初始化

		PyRun_SimpleString("import sys");
		PyRun_SimpleString("sys.path.append('D:/Anaconda3/envs/AI_cpu/Lib/site-packages')");
		PyRun_SimpleString("sys.path.append('I:/MIL_AI/Python/auxiliary')");
		PyObject* pModule = PyImport_Import(PyUnicode_FromString("Img_Cluster"));
		if (!pModule) {
			cout << "Can't find  Img_Cluster" << endl;
			std::system("pause");
		}

		////调用pt1函数
		PyObject* pFunc = PyObject_GetAttrString(pModule, "pt1");//这里是要调用的函数名
		PyObject* pyParams = PyTuple_New(4); //定义两个变量
		const char* Csrc_dir = "G:/DefectDataCenter/Test/Src/90";
		const char* dst_dir = "G:/DefectDataCenter/Test/ImgCluster";
		float	Eeps = 1.2;
		const char* pn = "SPA90";
		PyTuple_SetItem(pyParams, 0, Py_BuildValue("s", Csrc_dir));// 变量格式转换成python格式
		PyTuple_SetItem(pyParams, 1, Py_BuildValue("s", dst_dir));// 变量格式转换成python格式
		PyTuple_SetItem(pyParams, 2, Py_BuildValue("f", Eeps));// 变量格式转换成python格式
		PyTuple_SetItem(pyParams, 3, Py_BuildValue("s", pn));// 变量格式转换成python格式
		PyEval_CallObject(pFunc, pyParams);//调用函数
		


		//PyObject* pFunc = PyObject_GetAttrString(pModule, "pt2");//这里是要调用的函数名
		//PyEval_CallObject(pFunc, NULL);//调用函数
		//销毁python相关
		//Py_DECREF(pyParams);
		//Py_DECREF(pFunc);
		Py_DECREF(pModule);
		Py_Finalize();
}

/// MILTestWKSPDataset
bool isFileExists_ifstream(string& name) {
	ifstream f(name.c_str());
	return f.good();
}


void MILTest::MILTestWKSPRCDataset(MIL_STRING TagFolder)
{
	MIL_DOUBLE ValRatio = 0.1;
	MIL_STRING PPdCOMPSet = L"PPdCOMPSet";
	MIL_STRING PPdSimplSet = L"PPdSimplSet";

	MIL_STRING AuthorName = MIL_TEXT("AA");
	MIL_STRING BaseDataDir = m_ClassifierWorkSpace + m_strProject + L"/DataSet/" + L"/";
	m_MLClassCNN->CreateFolder(m_ClassifierWorkSpace + m_strProject + L"/DataSet/");
	MIL_STRING TagDataDir = m_TagDataDir + TagFolder;
	MIL_STRING BaseData_RPath = m_ClassifierWorkSpace + m_strProject + L"/DataSet/" + L"/COMPSetR.mclassd";
	string strBaseData_RPath;
	m_MLClassCNN->m_AIParse->MIL_STRING2string(BaseData_RPath, strBaseData_RPath);
	MIL_STRING BaseData_CPath = m_ClassifierWorkSpace + m_strProject + L"/DataSet/" + L"/COMPSetC.mclassd";
	string strBaseData_CPath;
	m_MLClassCNN->m_AIParse->MIL_STRING2string(BaseData_CPath, strBaseData_CPath);
	bool DataSetExist = isFileExists_ifstream(strBaseData_RPath)|| isFileExists_ifstream(strBaseData_CPath);

	//获取BaseClsNames
	vector<MIL_STRING> BaseClsNames;
	string strBaseImgDir;
	m_MLClassCNN->m_AIParse->MIL_STRING2string(BaseDataDir+L"Images", strBaseImgDir);
	m_MLClassCNN->m_AIParse->getFoldersInFolder(strBaseImgDir, BaseClsNames);

	//获取TagClassNames
	vector<MIL_STRING> TagClassNames;
	string  strTagImgDir;
	m_MLClassCNN->m_AIParse->MIL_STRING2string(TagDataDir+L"/R", strTagImgDir);
	m_MLClassCNN->m_AIParse->getFoldersInFolder(strTagImgDir, TagClassNames);

	//PrePared DataContext
	//*******************************必须参数*******************************//
	MIL_UNIQUE_CLASS_ID BaseSetRContext = MclassAlloc(m_MilSystem, M_PREPARE_IMAGES_CNN, M_DEFAULT, M_UNIQUE_ID);
	MIL_UNIQUE_CLASS_ID BaseSetCContext = MclassAlloc(m_MilSystem, M_PREPARE_IMAGES_CNN, M_DEFAULT, M_UNIQUE_ID);
	MIL_UNIQUE_CLASS_ID UpdateSetRContext = MclassAlloc(m_MilSystem, M_PREPARE_IMAGES_CNN, M_DEFAULT, M_UNIQUE_ID);
	MIL_UNIQUE_CLASS_ID UpdateSetCContext = MclassAlloc(m_MilSystem, M_PREPARE_IMAGES_CNN, M_DEFAULT, M_UNIQUE_ID);
	DataContextParasStruct DataCtxParas;
	DataCtxParas.ImageSizeX = 128;
	DataCtxParas.ImageSizeY = 128;
	memset(&DataCtxParas.AugParas, 0, sizeof(AugmentationParasStruct));
	DataCtxParas.AugParas.AugmentationNumPerImage = 1;
	
	DataCtxParas.AugParas.ScaleFactorMax = 1.03; //1.03
	DataCtxParas.AugParas.ScaleFactorMin = 0.97; //0.97
	DataCtxParas.AugParas.RotateAngleDelta = 10; //10
	DataCtxParas.AugParas.IntyDeltaAdd = 32;  //32
	DataCtxParas.AugParas.DirIntyMax = 1.2; //1.2
	DataCtxParas.AugParas.DirIntyMin = 0.8; //0.8
	DataCtxParas.AugParas.SmoothnessMax = 50; //50 {0<x<100}
	DataCtxParas.AugParas.SmoothnessMin = 0.5; //0.5 {0<x<100}
	DataCtxParas.AugParas.GaussNoiseStdev = 25; //25
	DataCtxParas.AugParas.GaussNoiseDelta = 25; //25

	DataCtxParas.ResizeModel = 1;
	DataCtxParas.PreparedDataFolder = m_ClassifierWorkSpace + m_strProject + L"/DataSet/PPdBaseSetR/";
	m_MLClassCNN->ConstructDataContext(DataCtxParas, BaseSetRContext);
	DataCtxParas.PreparedDataFolder = m_ClassifierWorkSpace + m_strProject + L"/DataSet/PPdBaseSetC/";
	DataCtxParas.ResizeModel = 0;
	m_MLClassCNN->ConstructDataContext(DataCtxParas, BaseSetCContext);

	DataCtxParas.PreparedDataFolder = m_ClassifierWorkSpace + m_strProject + L"/DataSet/PPdSpBaseSetC/";
	m_MLClassCNN->ConstructDataContext(DataCtxParas, UpdateSetCContext);
	DataCtxParas.PreparedDataFolder = m_ClassifierWorkSpace + m_strProject + L"/DataSet/PPdSpBaseSetR/";
	DataCtxParas.ResizeModel = 1;
	m_MLClassCNN->ConstructDataContext(DataCtxParas, UpdateSetRContext);

	if (DataSetExist) {
		vector<MIL_DOUBLE>vecSampleRatio = { 0.25,0.25 };
		MIL_STRING CSetType = L"C";
		MIL_UNIQUE_CLASS_ID COMPSet_C, SimplSet_C, PPdCOMPSet_C, PPdSimplSet_C;
		m_MLClassCNN->MergeTagData2BaseSet(BaseDataDir, CSetType, BaseClsNames, TagDataDir, vecSampleRatio, SimplSet_C, COMPSet_C);
		m_MLClassCNN->PrepareDataset(BaseSetCContext, COMPSet_C, PPdCOMPSet_C, BaseDataDir, PPdCOMPSet + CSetType);
		m_MLClassCNN->PrepareDataset(UpdateSetCContext, SimplSet_C, PPdSimplSet_C, BaseDataDir, PPdSimplSet + CSetType);

		MIL_STRING RSetType = L"R";
		MIL_UNIQUE_CLASS_ID COMPSet_R, SimplSet_R,PPdCOMPSet_R, PPdSimplSet_R;
		m_MLClassCNN->MergeTagData2BaseSet(BaseDataDir, RSetType, BaseClsNames, TagDataDir, vecSampleRatio, SimplSet_R, COMPSet_R);
		m_MLClassCNN->PrepareDataset(BaseSetRContext, COMPSet_R, PPdCOMPSet_R, BaseDataDir, PPdCOMPSet + RSetType);
		m_MLClassCNN->PrepareDataset(UpdateSetRContext, SimplSet_R, PPdSimplSet_R, BaseDataDir, PPdSimplSet + RSetType);
		//生成R/C并合
		m_MLClassCNN->Merge2Set(PPdCOMPSet_R, PPdCOMPSet_C, BaseDataDir, PPdCOMPSet);
		m_MLClassCNN->Merge2Set(PPdSimplSet_R, PPdSimplSet_C, BaseDataDir, PPdSimplSet);
	}
	else {
		MIL_STRING AuthorName = L"aa";
		MIL_UNIQUE_CLASS_ID TagDataSet, BaseDataSet;
		m_MLClassCNN->InitializeMergeRCDataset(AuthorName,BaseDataDir,TagDataDir, TagClassNames,BaseDataSet,TagDataSet);
	}
}

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
	DataCtxParas.AugParas.AugmentationNumPerImage = 1;
	DataCtxParas.ResizeModel = 1;
	DataCtxParas.AugParas.ScaleFactorMax = 1.03; //1.03
	DataCtxParas.AugParas.ScaleFactorMin = 0.97; //0.97
	DataCtxParas.AugParas.RotateAngleDelta = 10; //10
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
		MIL_UNIQUE_CLASS_ID PreparedDataset;
		m_MLClassCNN->PrepareDataset(BaseDataContext, BaseDataSet , PreparedDataset, BaseDataDir, L"PreParedBaseDataSet");
		m_MLClassCNN->PrepareDataset(UpdateDataContext, UpdateDataSet, PreparedDataset, BaseDataDir, L"PreParedUpdateDataSet");
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
		MIL_UNIQUE_CLASS_ID PreparedDataset;
		m_MLClassCNN->PrepareDataset(BaseDataContext, BaseDataSet, PreparedDataset, BaseDataDir, L"PreParedBaseDataSet");
	}
}

void MILTest::MILTestWKSPTrain()
{
	int MaxNumberOfEpoch = 20;			//模型训练次数
	int MiniBatchSize = 64;				//模型训练单次迭代的张数

	//////*******************************必须参数*******************************//
	MIL_STRING PreparedPath = m_ClassifierWorkSpace + m_strProject + MIL_TEXT("/DataSet/PPdMergeSet.mclassd");
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


