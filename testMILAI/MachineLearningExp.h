#pragma once
#define _MACHINE_LEARNING_EXPORT
#ifdef _MACHINE_LEARNING_EXPORT
#define MACHINELEARNING_DECLSPEC __declspec(dllexport)
#else
#define MACHINELEARNING_DECLSPEC __declspec(dllimport)
#endif
