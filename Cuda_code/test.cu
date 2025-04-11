int dev = 0;
    cudaDeviceProp devProp;
    CHECK(cudaGetDeviceProperties(&devProp, dev));
    std::cout << " GPU device " << dev << ": " << devProp.name << std::endl;
    std::cout << "SM " << devProp.multiProcessorCount << std::endl;
    std::cout << " " << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
    std::cout << " " << devProp.maxThreadsPerBlock << std::endl;
    std::cout << " EM " << devProp.maxThreadsPerMultiProcessor << std::endl;
    std::cout << " SM " << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;
