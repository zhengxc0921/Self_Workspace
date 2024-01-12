#include"AITorch.h"

CAITensor::CAITensor(MIL_ID MilSystem,
	MIL_STRING DstImgDir,
	string	strModelPath,
	string	ImgDir) :
	m_MilSystem(MilSystem),
	m_DstImgDir(DstImgDir),
	m_strModelPath(strModelPath),
	m_ImgDir(ImgDir)
{
	m_AIParse = CAIParsePtr(new CAIParse(MilSystem));
}

void CAITensor::LoadModel()
{
	m_ptModel = torch::jit::load("E:/SoftWareInstaller/python_demo_20201228/cpu_Cu_All.pt");
}

void CAITensor::normalize_image_buffer(MIL_ID RawImage, float* pNorlzBuffer)
{

	//分配内存，做图像缩放
	MIL_ID m_MilResizeImage;
	MbufAllocColor(m_MilSystem, 3, (MIL_INT)m_InModelSizeX, (MIL_INT)m_InModelSizeY, 32 + M_FLOAT, M_IMAGE + M_PROC, &m_MilResizeImage);
	MIL_ID m_MilNorImage;
	MbufAllocColor(m_MilSystem, 3, (MIL_INT)m_InModelSizeX, (MIL_INT)m_InModelSizeY, 32 + M_FLOAT, M_IMAGE + M_PROC, &m_MilNorImage);


	float* m_pResizeBufferOrgRGB = new float[(int)(m_InModelSizeX * m_InModelSizeY * 3)];
	MIL_INT img_width = MbufInquire(RawImage, M_SIZE_X, M_NULL);
	MIL_INT img_high = MbufInquire(RawImage, M_SIZE_Y, M_NULL);

	//2、初始化裁剪图像尺寸，AI模型输入图像尺寸
	MIL_DOUBLE ScaleFactorX = (MIL_DOUBLE)m_InModelSizeX / img_width;
	MIL_DOUBLE ScaleFactorY = (MIL_DOUBLE)m_InModelSizeY / img_high;
	//做图像缩放
	MimResize(RawImage, m_MilResizeImage, ScaleFactorX, ScaleFactorY, M_BILINEAR);  //M_BICUBIC  M_BILINEAR

	MimArith(m_MilResizeImage, 255, m_MilNorImage, M_DIV_CONST);
	MbufGetColor(m_MilNorImage, M_PLANAR, M_ALL_BANDS, m_pResizeBufferOrgRGB);
	int nPixelIndex = 0;
	long nResizeCount = m_InModelSizeX * m_InModelSizeY;
	for (int i = 0; i < nResizeCount; i++) {
		//M_PLANAR的图片存放格式为：RRRR...GGGG...BBBB
		//OpenCV的图片存放格式为：BGRBGR...BGR
		//AI模型使用OPENCV图像形式训练所得，故需要将MIL格式转换
		for (int j = 2; j >= 0; j--) {
			pNorlzBuffer[nPixelIndex] = m_pResizeBufferOrgRGB[i + nResizeCount * j];    //R_OpenCv<--R_MIL
			//cout << pNorlzBuffer[nPixelIndex]<<" ";
			nPixelIndex++;
		}
		//cout << endl;
	}

	delete m_pResizeBufferOrgRGB;
	MbufFree(m_MilNorImage);
	MbufFree(m_MilResizeImage);
}

void CAITensor::imgs2buffer()
{
	m_AIParse->getFilesInFolder(m_ImgDir, m_ImgType, m_Files);
	vector<MIL_STRING>::iterator it = m_Files.begin();

	m_vNorlzBuffer.clear();
	while (it != m_Files.end()) {
		MIL_ID RawImage = MbufRestore(*it, m_MilSystem, M_NULL);
		float* pNorlzBuffer = new float[m_InModelSizeX * m_InModelSizeY * 3];
		normalize_image_buffer(RawImage, pNorlzBuffer);
		m_vNorlzBuffer.push_back(pNorlzBuffer);
		it++;
	}
}

void CAITensor::PushEnd()
{
	int i = 0;
	bool FullBatch;
	vector<torch::Tensor>vecTmp;
	vector<float*>::iterator it = m_vNorlzBuffer.begin();
	while (it != m_vNorlzBuffer.end()) {
		i++;
		int64_t pInfo[4] = { 1, m_InModelSizeX, m_InModelSizeY, 3 };
		at::IntArrayRef info(pInfo, pInfo + ARRAYSIZE(pInfo));
		//转成Tensor的连续buffer区域
		at::Tensor tmp = torch::from_blob(*it, info, torch::kFloat32);
		vecTmp.push_back(tmp);
		FullBatch = i % m_maxBatchSize == 0;
		if (FullBatch) {
			at::Tensor img_tensor_ptr = torch::cat(vecTmp, 0);
			img_tensor_ptr = img_tensor_ptr.permute({ 0,3,1,2 });
			m_BatchValue.push_back(img_tensor_ptr.contiguous());
			vecTmp.clear();
		}

		it++;

	}
	//接入最后一部分数据: m_maxBatchSize/m_maxBatchSize/PartSize
	if (!FullBatch) {
		at::Tensor img_tensor_ptr = torch::cat(vecTmp, 0);
		img_tensor_ptr = img_tensor_ptr.permute({ 0,3,1,2 });
		m_BatchValue.push_back(img_tensor_ptr.contiguous());
		vecTmp.clear();
	}

}

void CAITensor::Forward()
{
	vector<torch::jit::IValue>::iterator it = m_BatchValue.begin();

	while (it != m_BatchValue.end()) {
		at::Tensor  ResultTensor = m_ptModel.forward({ *it }).toTensor();
		m_vecResultTensor.push_back(ResultTensor);

		it++;
	}
}

void CAITensor::ExtractScore(vector<vector<float>>& vScoreList)
{
	vector<at::Tensor>::iterator it = m_vecResultTensor.begin();
	int k = 0;
	vector<vector<float>> vTmpScoreList;
	while (it != m_vecResultTensor.end()) {

		size_t row = (*it).sizes()[0];
		size_t col = (*it).sizes()[1];
		vTmpScoreList.resize(row);
		at::Tensor tmp;
		for (size_t i = 0; i < row; i++)
		{
			tmp = (*it).slice(/*dim=*/0, /*start=*/i, /*end=*/i + 1);
			vector<float>& vScore = vTmpScoreList[i];
			for (size_t j = 0; j < col; j++)
			{
				vScore.push_back(tmp.select(1, j).item().toFloat());
			}
		}
		vScoreList.insert(vScoreList.end(), vTmpScoreList.begin(), vTmpScoreList.end());
		vTmpScoreList.clear();

		it++;
	}
}


void CAITensor::SaveRst(vector<vector<float>>& vScoreList) {
	for (int i = 0; i < m_Files.size(); i++) {

		MIL_ID RawImage = MbufRestore(m_Files[i], m_MilSystem, M_NULL);
		auto max_value = max_element(vScoreList[i].begin(), vScoreList[i].end());
		auto Img_index = max_value - vScoreList[i].begin();
		string::size_type iPos = m_Files[i].find_last_of('/') + 1;
		MIL_STRING ImageRawName = m_Files[i].substr(iPos, m_Files[i].length() - iPos);
		MIL_STRING DstRootPath = m_DstImgDir + m_ClassNames[Img_index] + MIL_TEXT("//") + ImageRawName;
		MbufExport(DstRootPath, M_BMP, RawImage);
	}

}

void CAITensor::Predict(vector<vector<float>>& vScoreList)
{
	LoadModel();
	imgs2buffer();
	PushEnd();
	//clock_t start, end;
	//start = clock();
	Forward();
	//end = clock();   //结束时间
	//cout << " img num = " << m_Files.size() << endl;  //图片数量
	//cout << "per img pred time = " << double(end - start) / CLOCKS_PER_SEC / m_Files.size() << "s" << endl;  //输出时间（单位：ｓ）
	ExtractScore(vScoreList);

	if (SAVERST) {
		for (std::vector<MIL_STRING>::iterator it = m_ClassNames.begin(); it != m_ClassNames.end(); ++it) {
			m_AIParse->CreateFolder(m_DstImgDir + (*it));
		}
		SaveRst(vScoreList);
	}

}

