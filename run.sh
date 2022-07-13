#!/bin/bash

cmake CMakeLists.txt
make
./kernelTestBench
nvprof ./kernelTestBench
compute-sanitizer ./kernelTestBench
