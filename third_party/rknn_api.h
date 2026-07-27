/****************************************************************************
*
*    Vendored from Rockchip's rknpu2 / rknn-toolkit2 SDK (rknn_api.h).
*    This is the declarations-only header needed to compile against
*    librknnrt on the board. It is NOT redistributed source of the
*    library itself -- librknnrt.so must come from the board's SDK
*    install (matching the version documented in transition.md:
*    librknnrt 2.4.2a2+).
*
*    Trimmed to the subset of declarations rknn_executor.cpp actually
*    uses (rknn_init, rknn_query, rknn_inputs_set, rknn_run,
*    rknn_outputs_get, rknn_outputs_release, rknn_destroy, rknn_dup_context,
*    rknn_set_core_mask) plus the supporting types. See the board SDK for
*    the full API.
*
*****************************************************************************/

#ifndef _RKNN_API_H
#define _RKNN_API_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/* Error codes (subset actually checked by rknn_executor.cpp). */
#define RKNN_SUCC                               0
#define RKNN_ERR_FAIL                           -1
#define RKNN_ERR_TIMEOUT                        -2
#define RKNN_ERR_DEVICE_UNAVAILABLE             -3
#define RKNN_ERR_MALLOC_FAIL                    -4
#define RKNN_ERR_PARAM_INVALID                  -5
#define RKNN_ERR_MODEL_INVALID                  -6
#define RKNN_ERR_CTX_INVALID                    -7
#define RKNN_ERR_INPUT_INVALID                  -8
#define RKNN_ERR_OUTPUT_INVALID                 -9

#define RKNN_MAX_DIMS                           16
#define RKNN_MAX_NAME_LEN                       256

#ifdef __arm__
typedef uint32_t rknn_context;
#else
typedef uint64_t rknn_context;
#endif

typedef enum _rknn_query_cmd {
    RKNN_QUERY_IN_OUT_NUM = 0,
    RKNN_QUERY_INPUT_ATTR = 1,
    RKNN_QUERY_OUTPUT_ATTR = 2,
    RKNN_QUERY_SDK_VERSION = 5,
    RKNN_QUERY_CMD_MAX
} rknn_query_cmd;

typedef enum _rknn_tensor_type {
    RKNN_TENSOR_FLOAT32 = 0,
    RKNN_TENSOR_FLOAT16,
    RKNN_TENSOR_INT8,
    RKNN_TENSOR_UINT8,
    RKNN_TENSOR_INT16,
    RKNN_TENSOR_UINT16,
    RKNN_TENSOR_INT32,
    RKNN_TENSOR_UINT32,
    RKNN_TENSOR_INT64,
    RKNN_TENSOR_TYPE_MAX
} rknn_tensor_type;

typedef enum _rknn_tensor_qnt_type {
    RKNN_TENSOR_QNT_NONE = 0,
    RKNN_TENSOR_QNT_DFP,
    RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC,
    RKNN_TENSOR_QNT_MAX
} rknn_tensor_qnt_type;

typedef enum _rknn_tensor_format {
    RKNN_TENSOR_NCHW = 0,
    RKNN_TENSOR_NHWC,
    RKNN_TENSOR_NC1HWC2,
    RKNN_TENSOR_UNDEFINED,
    RKNN_TENSOR_FORMAT_MAX
} rknn_tensor_format;

/* Which NPU core(s) a context's inference calls run on. Combine the
 * single-core values with bitwise OR, or use one of the preset combos. */
typedef enum _rknn_core_mask {
    RKNN_NPU_CORE_AUTO  = 0,
    RKNN_NPU_CORE_0     = 1,
    RKNN_NPU_CORE_1     = 2,
    RKNN_NPU_CORE_2     = 4,
    RKNN_NPU_CORE_0_1   = 3,
    RKNN_NPU_CORE_0_1_2 = 7,
    RKNN_NPU_CORE_ALL   = 0xffff,
} rknn_core_mask;

typedef struct _rknn_input_output_num {
    uint32_t n_input;
    uint32_t n_output;
} rknn_input_output_num;

typedef struct _rknn_tensor_attr {
    uint32_t index;
    uint32_t n_dims;
    uint32_t dims[RKNN_MAX_DIMS];
    char name[RKNN_MAX_NAME_LEN];
    uint32_t n_elems;
    uint32_t size;
    rknn_tensor_format fmt;
    rknn_tensor_type type;
    rknn_tensor_qnt_type qnt_type;
    int8_t fl;
    int32_t zp;
    float scale;
    uint32_t w_stride;
    uint32_t size_with_stride;
    uint8_t pass_through;
    uint32_t h_stride;
} rknn_tensor_attr;

typedef struct _rknn_sdk_version {
    char api_version[256];
    char drv_version[256];
} rknn_sdk_version;

typedef struct _rknn_input {
    uint32_t index;
    void* buf;
    uint32_t size;
    uint8_t pass_through;
    rknn_tensor_type type;
    rknn_tensor_format fmt;
} rknn_input;

typedef struct _rknn_output {
    uint8_t want_float;
    uint8_t is_prealloc;
    uint32_t index;
    void* buf;
    uint32_t size;
} rknn_output;

int rknn_init(rknn_context* context, void* model, uint32_t size, uint32_t flag, void* extend);
int rknn_destroy(rknn_context context);
int rknn_query(rknn_context context, rknn_query_cmd cmd, void* info, uint32_t size);
int rknn_inputs_set(rknn_context context, uint32_t n_inputs, rknn_input inputs[]);
int rknn_run(rknn_context context, void* extend);
int rknn_outputs_get(rknn_context context, uint32_t n_outputs, rknn_output outputs[], void* extend);
int rknn_outputs_release(rknn_context context, uint32_t n_ouputs, rknn_output outputs[]);

/* Duplicates an already-initialized context so it shares the source's loaded
 * model weights but can run independently (e.g. pinned to a different core). */
int rknn_dup_context(rknn_context* context_in, rknn_context* context_out);

/* Pins a context's future rknn_run() calls to the given NPU core(s). */
int rknn_set_core_mask(rknn_context context, rknn_core_mask core_mask);

#ifdef __cplusplus
} //extern "C"
#endif

#endif  //_RKNN_API_H
