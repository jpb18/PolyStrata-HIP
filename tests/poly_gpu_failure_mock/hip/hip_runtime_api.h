// Test-only HIP API. No runtime, driver or device is loaded by this fixture.
#pragma once
enum hipError_t { hipSuccess = 0, hipErrorInvalidValue = 1, hipErrorOutOfMemory = 2,
                  hipErrorNotReady = 600, hipErrorIllegalAddress = 700,
                  hipErrorLaunchTimeOut = 702, hipErrorContextIsDestroyed = 709,
                  hipErrorAssert = 710, hipErrorLaunchFailure = 719, hipErrorUnknown = 999 };
using hipStream_t = void*;
inline hipError_t last_error = hipSuccess, sync_error = hipSuccess;
inline int synchronized = 0;
inline hipError_t hipGetLastError() {
    hipError_t e = last_error;
    last_error = hipSuccess;
    return e;
}
inline hipError_t hipStreamSynchronize(hipStream_t) {
    ++synchronized;
    return sync_error;
}
inline const char* hipGetErrorName(hipError_t e) {
    switch (e) {
        case hipErrorIllegalAddress: return "hipErrorIllegalAddress";
        case hipErrorLaunchTimeOut: return "hipErrorLaunchTimeOut";
        case hipErrorContextIsDestroyed: return "hipErrorContextIsDestroyed";
        case hipErrorAssert: return "hipErrorAssert";
        case hipErrorLaunchFailure: return "hipErrorLaunchFailure";
        default: return "mockHipError";
    }
}
inline const char* hipGetErrorString(hipError_t) { return "injected error (no GPU)"; }
