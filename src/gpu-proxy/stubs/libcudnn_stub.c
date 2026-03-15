/* DeCloud GPU Proxy — cuDNN stub
 *
 * Satisfies DT_NEEDED: libcudnn.so.8 for libtorch_cuda.so and
 * libtorch_python.so. All functions return CUDNN_STATUS_NOT_INITIALIZED(1)
 * so PyTorch falls back to native attention kernels (Bug 19 — cuDNN
 * export table context struct unknown; native attention is correct and fast).
 *
 * Symbols exported with @@libcudnn.so.8 version tags to satisfy
 * versioned symbol imports in libtorch_cuda.so:
 *   cmp $0, cudnnGetRNNLinLayerMatrixParams@@libcudnn.so.8  (required)
 *   cmp $0, cudnnGetProperty@@libcudnn.so.8                (required)
 *
 * Built without CUDA headers — no CUDA toolkit dependency.
 * Compiled in Ubuntu 20.04 container (glibc 2.31+) for universal compat.
 *
 * Symbol count: 85 (84 from libtorch_cuda.so DT_NEEDED + cudnnGetProperty
 * from libtorch_python.so DT_NEEDED).
 */
#include <stddef.h>

/* ----------------------------------------------------------------
 * Special functions with typed signatures
 * ---------------------------------------------------------------- */

/* Returns cuDNN version 8.9.2 — matches what PyTorch 2.3+cu121 expects */
int cudnnGetVersion(void) { return 8902; }

const char *cudnnGetErrorString(int status) { (void)status; return "NOT_SUPPORTED"; }

/* cudnnGetProperty — required by libtorch_python.so.
 * Returns 0 for all properties (MAJOR=0, MINOR=0, PATCH=0). */
int cudnnGetProperty(int type, int *value) { (void)type; if (value) *value = 0; return 0; }

/* ----------------------------------------------------------------
 * All other cuDNN functions — return CUDNN_STATUS_NOT_INITIALIZED (1)
 * PyTorch checks for non-zero return and falls back to native kernels.
 * ---------------------------------------------------------------- */
int cudnnBackendCreateDescriptor(void *a, ...) { (void)a; return 1; }
int cudnnBackendDestroyDescriptor(void *a, ...) { (void)a; return 1; }
int cudnnBackendExecute(void *a, ...) { (void)a; return 1; }
int cudnnBackendFinalize(void *a, ...) { (void)a; return 1; }
int cudnnBackendGetAttribute(void *a, ...) { (void)a; return 1; }
int cudnnBackendSetAttribute(void *a, ...) { (void)a; return 1; }
int cudnnBatchNormalizationBackwardEx(void *a, ...) { (void)a; return 1; }
int cudnnBatchNormalizationForwardInference(void *a, ...) { (void)a; return 1; }
int cudnnBatchNormalizationForwardTrainingEx(void *a, ...) { (void)a; return 1; }
int cudnnCTCLoss(void *a, ...) { (void)a; return 1; }
int cudnnConvolutionBackwardData(void *a, ...) { (void)a; return 1; }
int cudnnConvolutionBackwardFilter(void *a, ...) { (void)a; return 1; }
int cudnnConvolutionBiasActivationForward(void *a, ...) { (void)a; return 1; }
int cudnnConvolutionForward(void *a, ...) { (void)a; return 1; }
int cudnnCreate(void *a, ...) { (void)a; return 1; }
int cudnnCreateActivationDescriptor(void *a, ...) { (void)a; return 1; }
int cudnnCreateCTCLossDescriptor(void *a, ...) { (void)a; return 1; }
int cudnnCreateConvolutionDescriptor(void *a, ...) { (void)a; return 1; }
int cudnnCreateDropoutDescriptor(void *a, ...) { (void)a; return 1; }
int cudnnCreateFilterDescriptor(void *a, ...) { (void)a; return 1; }
int cudnnCreatePoolingDescriptor(void *a, ...) { (void)a; return 1; }
int cudnnCreateRNNDataDescriptor(void *a, ...) { (void)a; return 1; }
int cudnnCreateRNNDescriptor(void *a, ...) { (void)a; return 1; }
int cudnnCreateSpatialTransformerDescriptor(void *a, ...) { (void)a; return 1; }
int cudnnCreateTensorDescriptor(void *a, ...) { (void)a; return 1; }
int cudnnDestroy(void *a, ...) { (void)a; return 1; }
int cudnnDestroyActivationDescriptor(void *a, ...) { (void)a; return 1; }
int cudnnDestroyCTCLossDescriptor(void *a, ...) { (void)a; return 1; }
int cudnnDestroyConvolutionDescriptor(void *a, ...) { (void)a; return 1; }
int cudnnDestroyDropoutDescriptor(void *a, ...) { (void)a; return 1; }
int cudnnDestroyFilterDescriptor(void *a, ...) { (void)a; return 1; }
int cudnnDestroyRNNDataDescriptor(void *a, ...) { (void)a; return 1; }
int cudnnDestroyRNNDescriptor(void *a, ...) { (void)a; return 1; }
int cudnnDestroySpatialTransformerDescriptor(void *a, ...) { (void)a; return 1; }
int cudnnDestroyTensorDescriptor(void *a, ...) { (void)a; return 1; }
int cudnnDropoutGetStatesSize(void *a, ...) { (void)a; return 1; }
int cudnnFindConvolutionBackwardDataAlgorithmEx(void *a, ...) { (void)a; return 1; }
int cudnnFindConvolutionBackwardFilterAlgorithmEx(void *a, ...) { (void)a; return 1; }
int cudnnFindConvolutionForwardAlgorithmEx(void *a, ...) { (void)a; return 1; }
int cudnnGetBatchNormalizationBackwardExWorkspaceSize(void *a, ...) { (void)a; return 1; }
int cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(void *a, ...) { (void)a; return 1; }
int cudnnGetBatchNormalizationTrainingExReserveSpaceSize(void *a, ...) { (void)a; return 1; }
int cudnnGetCTCLossWorkspaceSize(void *a, ...) { (void)a; return 1; }
int cudnnGetConvolutionBackwardDataAlgorithm_v7(void *a, ...) { (void)a; return 1; }
int cudnnGetConvolutionBackwardDataWorkspaceSize(void *a, ...) { (void)a; return 1; }
int cudnnGetConvolutionBackwardFilterAlgorithm_v7(void *a, ...) { (void)a; return 1; }
int cudnnGetConvolutionBackwardFilterWorkspaceSize(void *a, ...) { (void)a; return 1; }
int cudnnGetConvolutionForwardAlgorithm_v7(void *a, ...) { (void)a; return 1; }
int cudnnGetConvolutionForwardWorkspaceSize(void *a, ...) { (void)a; return 1; }
int cudnnGetCudartVersion(void *a, ...) { (void)a; return 1; }
int cudnnGetFilterNdDescriptor(void *a, ...) { (void)a; return 1; }
int cudnnGetRNNLinLayerBiasParams(void *a, ...) { (void)a; return 1; }
int cudnnGetRNNLinLayerMatrixParams(void *a, ...) { (void)a; return 1; }
int cudnnGetRNNParamsSize(void *a, ...) { (void)a; return 1; }
int cudnnGetRNNTrainingReserveSize(void *a, ...) { (void)a; return 1; }
int cudnnGetRNNWorkspaceSize(void *a, ...) { (void)a; return 1; }
int cudnnGetStream(void *a, ...) { (void)a; return 1; }
int cudnnGetTensorNdDescriptor(void *a, ...) { (void)a; return 1; }
int cudnnPoolingForward(void *a, ...) { (void)a; return 1; }
int cudnnRNNBackwardData(void *a, ...) { (void)a; return 1; }
int cudnnRNNBackwardWeights(void *a, ...) { (void)a; return 1; }
int cudnnRNNForwardInference(void *a, ...) { (void)a; return 1; }
int cudnnRNNForwardTraining(void *a, ...) { (void)a; return 1; }
int cudnnRestoreDropoutDescriptor(void *a, ...) { (void)a; return 1; }
int cudnnSetActivationDescriptor(void *a, ...) { (void)a; return 1; }
int cudnnSetCTCLossDescriptorEx(void *a, ...) { (void)a; return 1; }
int cudnnSetConvolutionGroupCount(void *a, ...) { (void)a; return 1; }
int cudnnSetConvolutionMathType(void *a, ...) { (void)a; return 1; }
int cudnnSetConvolutionNdDescriptor(void *a, ...) { (void)a; return 1; }
int cudnnSetDropoutDescriptor(void *a, ...) { (void)a; return 1; }
int cudnnSetFilterNdDescriptor(void *a, ...) { (void)a; return 1; }
int cudnnSetPooling2dDescriptor(void *a, ...) { (void)a; return 1; }
int cudnnSetRNNDataDescriptor(void *a, ...) { (void)a; return 1; }
int cudnnSetRNNDescriptor_v6(void *a, ...) { (void)a; return 1; }
int cudnnSetRNNMatrixMathType(void *a, ...) { (void)a; return 1; }
int cudnnSetRNNProjectionLayers(void *a, ...) { (void)a; return 1; }
int cudnnSetSpatialTransformerNdDescriptor(void *a, ...) { (void)a; return 1; }
int cudnnSetStream(void *a, ...) { (void)a; return 1; }
int cudnnSetTensorNdDescriptor(void *a, ...) { (void)a; return 1; }
int cudnnSpatialTfGridGeneratorBackward(void *a, ...) { (void)a; return 1; }
int cudnnSpatialTfGridGeneratorForward(void *a, ...) { (void)a; return 1; }
int cudnnSpatialTfSamplerBackward(void *a, ...) { (void)a; return 1; }
int cudnnSpatialTfSamplerForward(void *a, ...) { (void)a; return 1; }
