#include "MLBase.h"

CMLBase::CMLBase(MIL_ID MilSystem):m_MilSystem(MilSystem)
{
}

CMLBase::~CMLBase()
{
}



MIL_INT CMLBase::IsTrainingSupportedOnPlatform(MIL_ID MilSystem)
{
    // Validate that the MilSystem is allocated on a 64-bit platform.
    MIL_ID MilSystemOwnerApp = M_NULL;
    MsysInquire(MilSystem, M_OWNER_APPLICATION, &MilSystemOwnerApp);

    MIL_INT SystemPlatformBitness = 0;
    MappInquire(MilSystemOwnerApp, M_PLATFORM_BITNESS, &SystemPlatformBitness);

    MIL_INT SystemOsType = M_NULL;
    MappInquire(MilSystemOwnerApp, M_PLATFORM_OS_TYPE, &SystemOsType);

    // Verify if the platform is supported for training.
    bool SupportedTrainingPlaform = ((SystemPlatformBitness == 64) && (SystemOsType == M_OS_WINDOWS));
    if (!SupportedTrainingPlaform)
    {
        MosPrintf(MIL_TEXT("\n***** MclassTrain() is available only for Windows 64-bit platforms. *****\n"));
        return M_FALSE;
    }

    // If no train engine is installed on the MIL system then the train example cannot run.
    if (CnnTrainEngineDLLInstalled(MilSystem) != M_TRUE)
    {
        MosPrintf(MIL_TEXT("\n***** No train engine installed, MclassTrain() cannot run! *****\n"));
        return M_FALSE;
    }

    return M_TRUE;
}

void CMLBase::CreateFolder(const MIL_STRING& FolderPath)
{
    MIL_INT FolderExists = M_NO;
    MappFileOperation(M_DEFAULT, FolderPath, M_NULL, M_NULL, M_FILE_EXISTS, M_DEFAULT, &FolderExists);
    if (FolderExists == M_NO)
    {
        MappFileOperation(M_DEFAULT, FolderPath, M_NULL, M_NULL, M_FILE_MAKE_DIR, M_DEFAULT, M_NULL);
    }
}

void CMLBase::ClassifierSave(MIL_UNIQUE_CLASS_ID& ClassifierCtx, const MIL_STRING& ClassifierFileName)
{
    MclassSave(ClassifierFileName, ClassifierCtx, M_DEFAULT);
}

void CMLBase::ClassifierLoad(const MIL_STRING& ClassifierFileName, MIL_UNIQUE_CLASS_ID& ClassifierCtx)
{
    MIL_INT FileExists = M_NO;
    MappFileOperation(M_DEFAULT, ClassifierFileName, M_NULL, M_NULL, M_FILE_EXISTS, M_DEFAULT, &FileExists);
    if (FileExists)
    {
        ClassifierCtx = MclassRestore(ClassifierFileName, m_MilSystem, M_DEFAULT, M_UNIQUE_ID);
    }
}

//void CMLBase::DatasetSave(MIL_UNIQUE_CLASS_ID& Dataset, const MIL_STRING& DatasetFilePath)
//{
//    MclassExport(DatasetFilePath, M_IMAGE_DATASET_FOLDER, Dataset, M_DEFAULT, M_COMPLETE, M_DEFAULT);
//}

//MIL_INT CMLBase::CnnTrainEngineDLLInstalled(MIL_ID MilSystem)
//{
//    MIL_INT IsInstalled = M_FALSE;
//
//    MIL_UNIQUE_CLASS_ID TrainCtx = MclassAlloc(MilSystem, M_TRAIN_CNN, M_DEFAULT, M_UNIQUE_ID);
//    MclassInquire(TrainCtx, M_DEFAULT, M_TRAIN_ENGINE_IS_INSTALLED + M_TYPE_MIL_INT, &IsInstalled);
//
//    return IsInstalled;
//}
