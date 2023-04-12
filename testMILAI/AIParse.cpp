
#include "AIParse.h"

CAIParse::CAIParse(MIL_ID MilSystem):m_MilSystem(MilSystem)
{
}

void CAIParse::MIL_STRING2string(MIL_STRING milstrX, string& strX)
{
	//将MIL_STRING 先转换为string , string strX;
	//获取缓冲区大小，并申请空间，缓冲区大小事按字节计算的  
	int len = WideCharToMultiByte(CP_ACP, 0, milstrX.c_str(), milstrX.size(), NULL, 0, NULL, NULL);
	char* buffer = new char[len + 1];
	//宽字节编码转换成多字节编码  
	WideCharToMultiByte(CP_ACP, 0, milstrX.c_str(), milstrX.size(), buffer, len, NULL, NULL);
	buffer[len] = '\0';
	//删除缓冲区并返回值  
	strX.append(buffer);
	delete[] buffer;
}

wchar_t* CAIParse::string2MIL_STRING(string strX)
{
	char* CStr = const_cast<char*>(strX.c_str());
	int CStrLength = MultiByteToWideChar(CP_ACP, 0, CStr, -1, NULL, 0);
	wchar_t* milX = new wchar_t[CStrLength];
	MultiByteToWideChar(CP_ACP, 0, CStr, -1, milX, CStrLength);

	return milX;
}

void CAIParse::Split(const string& str, vector<string>& tokens, const string& delimiters)
{
	// Skip delimiters at beginning.
	string::size_type lastPos = str.find_first_not_of(delimiters, 0);
	// Find first "non-delimiter".
	string::size_type pos = str.find_first_of(delimiters, lastPos);
	while (string::npos != pos || string::npos != lastPos)
	{
		// Found a token, add it to the vector.
		tokens.push_back(str.substr(lastPos, pos - lastPos));
		// Skip delimiters.  Note the "not_of"
		lastPos = str.find_first_not_of(delimiters, pos);
		// Find next "non-delimiter"
		pos = str.find_first_of(delimiters, lastPos);
	}
}



void CAIParse::getFoldersInFolder(string Path, vector<string>& Folders)
{
	long long  hFile = 0;
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(Path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			//如果是目录,保存在folder中	
			if ((fileinfo.attrib & _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					Folders.emplace_back(fileinfo.name);
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}



void CAIParse::getFilesInFolder(string Path, string FileType, vector<MIL_STRING>& Files)
{
	long long  hFile = 0;
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(Path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
		//如果是目录,递归查找
		//如果不是,把文件绝对路径存入vector中
			if ((fileinfo.attrib & _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					getFilesInFolder(p.assign(Path).append("\\").append(fileinfo.name), FileType, Files);
			}
			else
			{
				////获取文件的FileType
				string ss = fileinfo.name;
				std::size_t  ipos = ss.find_last_of(".") + 1;
				string nFileType = ss.substr(ipos, ss.size());
				if (nFileType == FileType) {
					string  FullPatchName;
					if (Path.find_last_of("/") == Path.length() - 1 || Path.find_last_of("\\") == Path.length() - 1) {
						FullPatchName = Path + fileinfo.name;
					}
					else {
						FullPatchName = Path + "/" + fileinfo.name;
					}
					char* CStr = const_cast<char*>(FullPatchName.c_str());
					int CStrLength = MultiByteToWideChar(CP_ACP, 0, CStr, -1, NULL, 0);
					wchar_t* Filename = new wchar_t[CStrLength];
					MultiByteToWideChar(CP_ACP, 0, CStr, -1, Filename, CStrLength);
					Files.push_back(Filename);
				}
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}

void CAIParse::getFilesInFolder(string Path, string FileType, vector<string>& Files)
{
	long long  hFile = 0;
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(Path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
		//如果是目录,递归查找
		//如果不是,把文件绝对路径存入vector中
			if ((fileinfo.attrib & _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					getFilesInFolder(p.assign(Path).append("\\").append(fileinfo.name), FileType, Files);
			}
			else
			{
				////获取文件的FileType
				string ss = fileinfo.name;
				std::size_t  ipos = ss.find_last_of(".") + 1;
				string nFileType = ss.substr(ipos, ss.size());
				if (nFileType == FileType) {
					string  FullPatchName;
					if (Path.find_last_of("/") == Path.length() - 1 || Path.find_last_of("\\") == Path.length() - 1) {
						FullPatchName = Path + fileinfo.name;
					}
					else {
						FullPatchName = Path + "/" + fileinfo.name;
					}
					Files.push_back(FullPatchName);
				}
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}

void CAIParse::readBlob2Vector(string file, vector<int>& blob_px, vector<int>& blob_py)
{
	//读取point_x/point_y 到vector
	ifstream in(file, ios::in);
	string line;
	while (getline(in, line)) {//每一次获取一行的数据到line
		size_t pos_point1 = line.find_first_of(".");
		size_t pos_space = line.find_first_of(" ");
		size_t pos_point2 = line.find_last_of(".");
		string px = line.substr(0, pos_point1);
		string py = line.substr(pos_space, pos_point2);
		blob_px.emplace_back(atoi(px.c_str()));
		blob_py.emplace_back(atoi(py.c_str()));
	}
}

void CAIParse::readClasses2Vector(string file, vector<MIL_STRING>& vecImgPaths)
{
	ifstream in(file, ios::in);
	string line;
	while (getline(in, line)) {
		MIL_STRING Mline = string2MIL_STRING(line);
		vecImgPaths.emplace_back(Mline);
	}
}

void CAIParse::readDataSet2Vector(string file,
	vector<MIL_STRING>& vecImgPaths,
	vector<vector<Box>>& vec2Boxes,
	vector<vector<int>>& veclabels)
{
	ifstream in(file, ios::in);
	string line;
	while (getline(in, line)) {//每一次获取一行的数据到line
		//单张图片对应的Boxes，以及对应的label
		vector<string>vecline;
		Split(line, vecline, " ");
		vector<string>::iterator lineIter;
		MIL_STRING ImgPaths = string2MIL_STRING(*(vecline.begin()));

		vecImgPaths.emplace_back(ImgPaths);
		vector<Box> tmpBoxes;
		vector<int> tmpLabels;
		for (lineIter = vecline.begin() + 1; lineIter != vecline.end(); lineIter++) {
			Box tmpBox;
			vector<string>tmpBoxLabel;
			Split(*lineIter, tmpBoxLabel, ",");
			tmpBox.x1 = atoi(tmpBoxLabel[0].c_str());
			tmpBox.y1 = atoi(tmpBoxLabel[1].c_str());
			tmpBox.x2 = atoi(tmpBoxLabel[2].c_str());
			tmpBox.y2 = atoi(tmpBoxLabel[3].c_str());

			tmpBoxes.emplace_back(tmpBox);
			tmpLabels.emplace_back(atoi(tmpBoxLabel[4].c_str()));
		}

		//读取所有图片对应的Boxes，以及对应的label
		vec2Boxes.emplace_back(tmpBoxes);
		veclabels.emplace_back(tmpLabels);
	}
}

void CAIParse::blobClip(string ImgDir,
	string ImgName,
	vector<int> blob_px,
	vector<int> blob_py,
	vector<MIL_ID>& ClipImgs,
	string CropedImgDir)
{
	string img_path = ImgDir + "/" + ImgName + ".bmp";
	MIL_STRING ImgPath = string2MIL_STRING(img_path);
	//MIL_ID RawImage = MbufRestore(L"D:/LeetCode/Img/AugImg/L1C14 (2).bmp", m_MilSystem, M_NULL);
	//MIL_ID RawImage = MbufRestore(L"G:/DefectDataCenter/lianglichuang/SXX_big_defect/SXX_big_defect_0/-1_-2_0_5_100_2234_18442_1(patch).28bbed0c64.bmp", m_MilSystem, M_NULL);

	MIL_ID RawImage = MbufRestore(ImgPath, m_MilSystem, M_NULL);
	MIL_INT ImgSizeX = MbufInquire(RawImage, M_SIZE_X, M_NULL);
	MIL_INT ImgSizeY = MbufInquire(RawImage, M_SIZE_Y, M_NULL);

	//2、按点族切割box
	int CropSizeX = 128, CropSizeY = 128;
	vector<Box> vecCropedBoxes;
	cropBox(blob_px,
		blob_py,
		CropSizeX,
		CropSizeY,
		ImgSizeX,
		ImgSizeY,
		vecCropedBoxes);

	//3、保存ClipImgs
	vector<Box>::iterator iter;
	int CropID = 1;
	for (iter = vecCropedBoxes.begin(); iter != vecCropedBoxes.end(); iter++)
	{
		MIL_ID InputImage = MbufAlloc2d(m_MilSystem,
			CropSizeX,
			CropSizeY,
			8 + M_UNSIGNED,
			M_IMAGE + M_PROC,
			M_NULL);
		MbufClear(InputImage, 0);
		auto CropBoxes = *iter;
		MbufCopyColor2d(RawImage,
			InputImage,
			M_ALL_BANDS,
			CropBoxes.x1,
			CropBoxes.y1,
			M_ALL_BANDS, 0, 0,
			CropSizeX, CropSizeY);
		string CropedImgPath = CropedImgDir + "\\" + to_string(CropID) + ImgName + ".bmp";
		MIL_STRING MCropedImgPath = string2MIL_STRING(CropedImgPath);
		MbufExport(MCropedImgPath, M_BMP, InputImage);
		MbufFree(InputImage);
		CropID++;
	}
	MbufFree(RawImage);
}

void CAIParse::cropBox(vector<int>blob_px,
	vector<int>blob_py,
	int CropSizeX,
	int CropSizeY,
	int ImgSizeX,
	int ImgSizeY,
	vector<Box>& vecCropedBoxes)
{
	int nPointNum = blob_px.size();
	map<string, vector<Point>>mapPointLists;
	//将点按box划分
	for (int k = 0; k < nPointNum; k++) {
		Point pt;
		int x_index = (blob_px[k] - 0) / CropSizeX;
		int y_index = (blob_py[k] - 0) / CropSizeY;
		string keys = to_string(x_index) + "_" + to_string(y_index);
		pt.px = blob_px[k];
		pt.py = blob_py[k];
		mapPointLists[keys].emplace_back(pt);
	}
	//对 Cluster  Points求质心，将结果保存到vecCropedBoxes
	for (auto it : mapPointLists) {
		Box CropedBoxes;
		vector<Point>vecClusterPoint = it.second;
		int tmp_x = 0;
		int tmp_y = 0;
		int nPointNum = vecClusterPoint.size();
		vector<Point>::iterator CPiter;
		for (CPiter = vecClusterPoint.begin(); CPiter != vecClusterPoint.end(); CPiter++) {
			tmp_x += (*CPiter).px;
			tmp_y += (*CPiter).py;
		}
		int leftup_x = tmp_x / nPointNum - CropSizeX / 2;
		int leftup_y = tmp_y / nPointNum - CropSizeY / 2;
		CropedBoxes.x1 = min(max(leftup_x, 0), ImgSizeX - CropSizeX);
		CropedBoxes.y1 = min(max(leftup_y, 0), ImgSizeY - CropSizeY);
		CropedBoxes.x2 = CropedBoxes.x1 + CropSizeX;
		CropedBoxes.y2 = CropedBoxes.y1 + CropSizeY;
		vecCropedBoxes.emplace_back(CropedBoxes);
	}
}

void CAIParse::ImgCenterCrop(MIL_ID ImgIn, int CropWH, MIL_ID& ImgOut)
{
	MIL_INT ImgSizeX = MbufInquire(ImgIn, M_SIZE_X, M_NULL);
	MIL_INT ImgSizeY = MbufInquire(ImgIn, M_SIZE_Y, M_NULL);
	int CropSizeX = min(ImgSizeX - 1, CropWH);
	int CropSizeY = min(ImgSizeY - 1, CropWH);

	int x1 = max((ImgSizeX - CropWH) / 2,0);
	int y1 = max((ImgSizeY - CropWH) / 2, 0);

	ImgOut = MbufAlloc2d(m_MilSystem,
		CropSizeX,CropSizeY,
		8 + M_UNSIGNED,
		M_IMAGE + M_PROC,
		M_NULL);
	MbufClear(ImgOut, 0);

	MbufCopyColor2d(ImgIn,ImgOut,
		M_ALL_BANDS,x1,y1,
		M_ALL_BANDS, 0, 0,
		CropSizeX, CropSizeY);
}
