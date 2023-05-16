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
//    string arg_exe = "E:\\MIL_CNN\\MIL_classifier\\ClassCNNCompleteTrain - ����\\C++\\vs2017\\x64\\Debug\\ClassCNNCompleteTrain.exe";
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

// ͨ��stat�ṹ�� ����ļ���С����λ�ֽ�
size_t getFileSize1(const char* fileName) {

    if (fileName == NULL) {
        return 0;
    }

    // ����һ���洢�ļ�(��)��Ϣ�Ľṹ�壬�������ļ���С�ʹ���ʱ�䡢����ʱ�䡢�޸�ʱ���
    struct stat statbuf;

    // �ṩ�ļ����ַ���������ļ����Խṹ��
    stat(fileName, &statbuf);

    // ��ȡ�ļ���С
    size_t filesize = statbuf.st_size;

    return filesize;
}

int mainA()
{
    //MIL�ļ�
    MIL_UNIQUE_APP_ID MilApplication = MappAlloc(M_NULL, M_DEFAULT, M_UNIQUE_ID);
    MIL_UNIQUE_SYS_ID MilSystem = MsysAlloc(M_DEFAULT, M_SYSTEM_HOST, M_DEFAULT, M_DEFAULT, M_UNIQUE_ID);
    MIL_UNIQUE_DISP_ID MilDisplay = MdispAlloc(MilSystem, M_DEFAULT, MIL_TEXT("M_DEFAULT"), M_DEFAULT, M_UNIQUE_ID);
    MdispControl(MilDisplay, M_TITLE, MIL_TEXT("AIT DisPlay"));

    /*const char* f_name = "G:/DefectDataCenter/ParseData/Detection/lslm_bmp/MIL_Data/PreparedData/lslm_bmp.mclass";*/

    const char* f_name = "G:/DefectDataCenter/ParseData/Detection/DSW_random/MIL_Data/PreparedData/DSW_random.mclass";
    size_t filesize = getFileSize1(f_name);
    string strShareMame = "testCtx";

    //���������ڴ�
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
    //��ȡ�����ڴ�
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
        //����һ���½���  
        if (CreateProcess(
            lpApplicationName,   //  ָ��һ��NULL��β�ġ�����ָ����ִ��ģ��Ŀ��ֽ��ַ���  
            szCommandLine, // �������ַ���  
            NULL, //    ָ��һ��SECURITY_ATTRIBUTES�ṹ�壬����ṹ������Ƿ񷵻صľ�����Ա��ӽ��̼̳С�  
            NULL, //    ���lpProcessAttributes����Ϊ�գ�NULL������ô������ܱ��̳С�<ͬ��>  
            false,//    ָʾ�½����Ƿ�ӵ��ý��̴��̳��˾����   
            0,  //  ָ�����ӵġ���������������ͽ��̵Ĵ����ı�  
                //  CREATE_NEW_CONSOLE  �¿���̨���ӽ���  
                //  CREATE_SUSPENDED    �ӽ��̴��������ֱ������ResumeThread����  
            NULL, //    ָ��һ���½��̵Ļ����顣����˲���Ϊ�գ��½���ʹ�õ��ý��̵Ļ���  
            NULL, //    ָ���ӽ��̵Ĺ���·��  
            &si, // �����½��̵������������ʾ��STARTUPINFO�ṹ��  
            &pi  // �����½��̵�ʶ����Ϣ��PROCESS_INFORMATION�ṹ��  
        ))
        {
            cout << "create process success" << endl;
            //�������йرվ������������̺��½��̵Ĺ�ϵ����Ȼ�п��ܲ�С�ĵ���TerminateProcess�����ص��ӽ���  
            CloseHandle(pi.hProcess);
            CloseHandle(pi.hThread);
        }
        else {
            cerr << "failed to create process" << endl;
        }

        Sleep(100);

    }
    //��ֹ�ӽ���  
    TerminateProcess(pi.hProcess, 300);

    //��ֹ�����̣�״̬��  
    ExitProcess(1001);
    return 0;
}



