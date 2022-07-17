#ifndef DEVICE_CUH
#define DEVICE_CUH

#include <stdio.h>
#include <iostream>
#include <cuda.h>


void  printAllDeviceInfo(void);
void  printDeviceInfo(int deviceId);
float getBusWidth(int deviceId);
float getPeakBandWidth(int deviceId);
float getTotalMemory(int deviceId);

#endif
