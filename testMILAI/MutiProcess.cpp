//#include <cstdlib>
//#include <ctime>
//#include <iostream>
//#include <string>
//#include <mil.h>
//#include <Windows.h>
//using namespace std;
//
//std::string get_random_ascii_data(bool file_name = false)
//{
//    const int low = file_name ? 0x61 : 0x21;
//    const int high = file_name ? 0x7a : 0x7e;
//    const int length = file_name ? 15 : 4096;
//    std::string buffer;
//    for (size_t i = 0; i < length; ++i)
//        buffer.push_back((const char)(low + rand() % (high - low + 1)));
//
//    return buffer;
//}
//
//void create_file()
//{
//    auto name = get_random_ascii_data(true) + ".txt";
//
//    std::cout << "CreateFile " << name << std::endl;
//
//    auto file_handle = CreateFileA(name.data(), FILE_APPEND_DATA, 0, NULL, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, NULL);
//
//    if (file_handle == INVALID_HANDLE_VALUE)
//    {
//        std::cout << "Terminal failure: Unable to open file \"" << name << "\" for write" << std::endl;
//        return;
//    }
//
//    for (int i = 0; i < 4000; ++i)
//    {
//        DWORD bytes_written = 0;
//        auto buffer = get_random_ascii_data();
//        auto error_flag = WriteFile(file_handle, buffer.data(), buffer.size(), &bytes_written, nullptr);
//
//        if (FALSE == error_flag)
//        {
//            std::cout << "Terminal failure: Unable to write to file" << std::endl;
//            break;
//        }
//    }
//
//    CloseHandle(file_handle);
//    DeleteFileA(name.data());
//}
//
//void create_child_processes(std::string name, MIL_UNIQUE_CLASS_ID& pretxt)
//{
//    for (int i = 0; i < 5; ++i)
//    {
//        std::cout << "[" << i << "] CreatePocess " << name << std::endl;
//        STARTUPINFOA info = { sizeof(info) };
//        PROCESS_INFORMATION process_info;
//
//        LPSTR  str = "abcasj555";
//        LPSTR  strI = &str[i];
//
//        if (CreateProcessA(name.c_str(), (char*)pretxt, NULL, NULL, TRUE, 0, NULL, NULL, &info, &process_info))
//
//
//        //LPSTR  str = "abcasj555";
//        //LPSTR  strI = &str[i];
//        //LPCSTR Iname = name.c_str() ;
//        //if (CreateProcessA(Iname,(char*)strI, NULL, NULL, TRUE, 0, NULL, NULL, &info, &process_info))
//
//       /* auto full_cmd = name + " child";
//        if (CreateProcessA(nullptr, (char*)full_cmd.data(), NULL, NULL, TRUE, 0, NULL, NULL, &info, &process_info))*/
//        {
//            std::cout << "Create process " << process_info.dwProcessId << " (" << process_info.hProcess << ")" << std::endl;
//            WaitForSingleObject(process_info.hProcess, INFINITE);
//            std::cout << "Process " << process_info.dwProcessId << " (" << process_info.hProcess << ") finished" << std::endl;
//
//            //std::cout << "(char*)full_cmd.data(): " << (char*)full_cmd.data() << std::endl;
//
//            CloseHandle(process_info.hProcess);
//            CloseHandle(process_info.hThread);
//        }
//        else
//        {
//            std::cout << "Failed to create process" << std::endl;
//        }
//    }
//}
//
//void print_usage(std::string name)
//{
//    std::cout << "Usage: " << name << " [parent | child]" << std::endl;
//}
//
//void create_share_memory() {
//
//
//    HANDLE hFile = CreateFile("Recv1.zip",
//        GENERIC_WRITE | GENERIC_READ,
//        FILE_SHARE_READ,
//        NULL,
//        CREATE_ALWAYS,
//        FILE_FLAG_SEQUENTIAL_SCAN,
//        NULL);
//
//}
//
//
//int main(int argc, char** argv)
//{
//
//    MIL_UNIQUE_APP_ID MilApplication = MappAlloc(M_NULL, M_DEFAULT, M_UNIQUE_ID);
//    MIL_UNIQUE_SYS_ID MilSystem = MsysAlloc(M_DEFAULT, M_SYSTEM_HOST, M_DEFAULT, M_DEFAULT, M_UNIQUE_ID);
//    MIL_STRING PreClassifierName = MIL_TEXT("G:/DefectDataCenter/zhjuzhiqiang_2023/2023/spa/PreparedData/spa.mclass");
//    MIL_UNIQUE_CLASS_ID testClassifierCtx = MclassRestore(PreClassifierName, MilSystem, M_DEFAULT, M_UNIQUE_ID);
//
//    string arg_exe = "E:\\MIL_CNN\\MIL_classifier\\ClassCNNCompleteTrain - 副本\\C++\\vs2017\\x64\\Debug\\ClassCNNCompleteTrain.exe";
//    std::srand(std::time(nullptr));
//    //CreateProcessA("app.exe", "")
//    std::string role;
//    if (argc == 1)
//        role = "parent";
//    else if (argc == 2)
//        role = std::string(argv[1]);
//    else
//    {
//        print_usage(argv[0]);
//        return -1;
//    }
//    std::cout << "My role is " << role << std::endl;
//
//    if ("parent" == role)
//        create_child_processes(arg_exe, testClassifierCtx);
//    else if ("child" == role)
//        create_file();
//    else
//    {
//        print_usage(argv[0]);
//        return -1;
//    }
//
//    return 0;
//}

#include <stdio.h>
#include <tchar.h>
#include <windows.h>
#include<iostream>
#include <mil.h>
#include <fstream>

using namespace std;

// 通过stat结构体 获得文件大小，单位字节
size_t getFileSize1(const char* fileName) {

    if (fileName == NULL) {
        return 0;
    }

    // 这是一个存储文件(夹)信息的结构体，其中有文件大小和创建时间、访问时间、修改时间等
    struct stat statbuf;

    // 提供文件名字符串，获得文件属性结构体
    stat(fileName, &statbuf);

    // 获取文件大小
    size_t filesize = statbuf.st_size;

    return filesize;
}

int mainA()
{
    //MIL文件
    MIL_UNIQUE_APP_ID MilApplication = MappAlloc(M_NULL, M_DEFAULT, M_UNIQUE_ID);
    MIL_UNIQUE_SYS_ID MilSystem = MsysAlloc(M_DEFAULT, M_SYSTEM_HOST, M_DEFAULT, M_DEFAULT, M_UNIQUE_ID);
    MIL_UNIQUE_DISP_ID MilDisplay = MdispAlloc(MilSystem, M_DEFAULT, MIL_TEXT("M_DEFAULT"), M_DEFAULT, M_UNIQUE_ID);
    MdispControl(MilDisplay, M_TITLE, MIL_TEXT("AIT DisPlay"));

    /*const char* f_name = "G:/DefectDataCenter/ParseData/Detection/lslm_bmp/MIL_Data/PreparedData/lslm_bmp.mclass";*/

    const char* f_name = "G:/DefectDataCenter/ParseData/Detection/DSW_random/MIL_Data/PreparedData/DSW_random.mclass";
    size_t filesize = getFileSize1(f_name);
    string strShareMame = "testCtx";

    //创建共享内存
    HANDLE hMap = CreateFileMappingA(
        INVALID_HANDLE_VALUE,	// use paging file
        NULL,                   // default security 
        PAGE_READWRITE,         // read/write access
        0,                      // max. object size 
        filesize + sizeof(ULONGLONG),	// buffer size  max 4G
        strShareMame.c_str());	// name of mapping object

    BYTE* pBuf = (BYTE*)MapViewOfFile(
        hMap,   // handle to map object
        FILE_MAP_WRITE, // FILE_MAP_ALL_ACCESS, // read/write permission
        0,
        0,
        filesize + sizeof(ULONGLONG));
    FILE* pFile = fopen(f_name, "rb");
    size_t result = fread((pBuf + sizeof(ULONGLONG)), 1, filesize, pFile);
    printf("%s", pBuf + filesize);
    //读取共享内存
    auto m_hMap = ::OpenFileMappingA(FILE_MAP_READ, FALSE, strShareMame.c_str());
    if (m_hMap == NULL) {
        std::cout << "m_hMap is Null " << std::endl;
        return 0;
    }
    else {
        std::cout << "m_hMap is Not Null " << std::endl;
    }
    LPTSTR lpApplicationName = (LPTSTR)"I:/MIL_AI/testMILAI/x64/Debug/testMILAI.exe";
    STARTUPINFO si;
    PROCESS_INFORMATION pi;
    string ImgType = "bmp";
    string strProject = "DSW_random";

    ZeroMemory(&si, sizeof(si));
    ZeroMemory(&pi, sizeof(pi));
    ofstream ODNetResult;
    string StartFile = "G:/DefectDataCenter/ParseData/Detection/DSW_random/raw_data/StartFile.txt";
    ODNetResult.open(StartFile, ios::out);
    ODNetResult << "Start" << endl;
    ODNetResult.close();
    for (int i = 0; i < 3; i++) {
        std::stringstream ss;
        ss << strShareMame << " " << filesize << " " << i << " " << ImgType << " " << strProject;
        string ssi = ss.str();
        LPTSTR szCommandLine = (TCHAR*)ssi.c_str();
        std::cout << "szCommandLine: " << szCommandLine << std::endl;
        //创建一个新进程  
        if (CreateProcess(
            lpApplicationName,   //  指向一个NULL结尾的、用来指定可执行模块的宽字节字符串  
            szCommandLine, // 命令行字符串  
            NULL, //    指向一个SECURITY_ATTRIBUTES结构体，这个结构体决定是否返回的句柄可以被子进程继承。  
            NULL, //    如果lpProcessAttributes参数为空（NULL），那么句柄不能被继承。<同上>  
            false,//    指示新进程是否从调用进程处继承了句柄。   
            0,  //  指定附加的、用来控制优先类和进程的创建的标  
                //  CREATE_NEW_CONSOLE  新控制台打开子进程  
                //  CREATE_SUSPENDED    子进程创建后挂起，直到调用ResumeThread函数  
            NULL, //    指向一个新进程的环境块。如果此参数为空，新进程使用调用进程的环境  
            NULL, //    指定子进程的工作路径  
            &si, // 决定新进程的主窗体如何显示的STARTUPINFO结构体  
            &pi  // 接收新进程的识别信息的PROCESS_INFORMATION结构体  
        ))
        {
            cout << "create process success" << endl;
            //下面两行关闭句柄，解除本进程和新进程的关系，不然有可能不小心调用TerminateProcess函数关掉子进程  
            CloseHandle(pi.hProcess);
            CloseHandle(pi.hThread);
        }
        else {
            cerr << "failed to create process" << endl;
        }

        Sleep(100);

    }
    //终止子进程  
    TerminateProcess(pi.hProcess, 300);

    //终止本进程，状态码  
    ExitProcess(1001);
    return 0;
}



