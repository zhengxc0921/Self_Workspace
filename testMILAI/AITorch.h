#pragma once
#pragma once
#define NOMINMAX
#undef min
#undef max

#include <boost/shared_ptr.hpp>
#include<iostream>
#include<vector>
using namespace std;
//AI头文件
#include <torch/script.h>
#include <torch/torch.h>

#include"AIParse.h"


////////////////////////////////////////
//CAITensor
class CAITensor;
typedef torch::jit::script::Module torch_model;
class CAITensor {

public:
	CAITensor(MIL_ID MilSystem,
		MIL_STRING m_DstImgDir,
		string	m_strModelPath,
		string	m_ImgDir);
	~CAITensor() {};

	void LoadModel();
	void normalize_image_buffer(MIL_ID RawImage, float* pNorlzBuffer);
	void imgs2buffer();
	void PushEnd();
	void Forward();
	void ExtractScore(vector<vector<float>>& vScoreList);
	void Predict(vector<vector<float>>& vScoreList);
	void SaveRst(vector<vector<float>>& vScoreList);

public:

	MIL_ID					m_MilSystem;
	CAIParsePtr				m_AIParse;
	////输入模型
	int						m_InModelSizeX = 224;
	int						m_InModelSizeY = 224;
	vector<MIL_STRING>		m_ClassNames = { MIL_TEXT("10"),MIL_TEXT("91") };
	torch_model				m_ptModel;
	string					m_strModelPath;
	////输入图像检测
	int						m_maxBatchSize = 64;
	string					m_ImgDir;
	string					m_ImgType = "bmp";
	vector<MIL_STRING>		m_Files;		//提取到图片
	vector<float*>			m_vNorlzBuffer;//多个Image的normalize buffer

	////模型输入、输出
	bool	   SAVERST = TRUE;
	//vector<vector<torch::jit::IValue>>m_vecBatchValue;
	vector<torch::jit::IValue>	m_BatchValue;
	vector<at::Tensor>			m_vecResultTensor;
	vector<vector<float>>		m_vScoreList;
	vector<int>					m_vClassIndex;
	MIL_STRING					m_DstImgDir;
};


