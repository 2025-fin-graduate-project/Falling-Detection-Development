/**
  ******************************************************************************
  * @file    full_stateful.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-05-22T12:27:34+0900
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */

#include "ai_lite_inspect.h"
#include "ai_platform_interface.h"
#include "layers.h"
#include "core_convert.h"
#include "full_stateful.h"
#include "full_stateful_details.h"
#include "full_stateful_data.h"
#include "stai_events.h"

#include "lite_operators.h"

#include "ai_lite_inspect.h"
/*****************************************************************************/
#define STAI_INTERNAL_API_MAJOR               (1)
#define STAI_INTERNAL_API_MINOR               (0)
#define STAI_INTERNAL_API_MICRO               (0)

#define STAI_MAGIC                            (0xB1C00100)

/*****************************************************************************/
#define _STAI_CONCAT_ARG(a, b)     a ## b
#define STAI_CONCAT(a, b)         _STAI_CONCAT_ARG(a, b)

/*!  STAI_CAST SECTION                       *********************************/
#define STAI_CAST(type, expr) \
  ((type)(expr))


/*****************************************************************************/
#define STAI_SIZE(_size) \
  ((stai_size)(_size))

/*****************************************************************************/
#define STAI_INIT_BUFFER(_flags, _size, _address) \
  { \
    .size = (_size), \
    .address = (uintptr_t)(_address), \
    .flags = (_flags), \
  }

#define STAI_INIT_TENSOR(_name, _flags, _fmt, _size_bytes, _shape, _scale, _zeropoint) \
  { \
    .size_bytes = (_size_bytes), \
    .flags = (_flags), \
    .format = (stai_format)(_fmt), \
    .shape = STAI_PACK(_shape), \
    .scale = STAI_PACK(_scale), \
    .zeropoint = STAI_PACK(_zeropoint), \
    .name = (_name) \
  }

#define STAI_INIT_ARRAY(_size, _ptr) \
  { .size = STAI_SIZE(_size), .data = STAI_PACK(_ptr) }


#define STAI_CAST_ARRAY(_type, _size, _ptr) \
  { .size = STAI_SIZE(_size), .data = (_type)STAI_PACK(_ptr) }


#define STAI_DECLARE_ARRAY(_type, _size, ...) \
  { .size = STAI_SIZE(_size), .data = (_type[_size]) { STAI_PACK(__VA_ARGS__) } }


#define STAI_EMPTY_ARRAY() \
  { .size = 0, .data = NULL }


#define STAI_INIT_VERSION(_major, _minor, _micro) \
  { .major = (_major), .minor = (_minor), .micro = (_micro), .reserved = 0x0 }

/*****************************************************************************/
/**  Getters and setters  **/

#define STAI_GET_ARRAY_SIZE(nd_array) \
  (nd_array.size)


#define STAI_GET_ARRAY_ELEM(nd_array, pos) \
  (nd_array.data[(pos)])

#define _STAI_SET_ERROR(net_ctx, cond, value, exit) { \
  if (!(net_ctx)) { return STAI_ERROR_NETWORK_INVALID_CONTEXT_HANDLE; } \
  if (((uintptr_t)net_ctx) & (_STAI_CONTEXT_ALIGNMENT-1)) { return STAI_ERROR_NETWORK_INVALID_CONTEXT_ALIGNMENT; } \
  if (((value) >= STAI_ERROR_GENERIC) && (cond)) { \
    if ((net_ctx)->_return_code == STAI_SUCCESS) { \
      (net_ctx)->_return_code = (value); \
    } \
    return (exit); \
  } \
}

/*****************************************************************************/
/* TODO REMOVE THESE TWO MACROS */
#define STAI_EVENT_NODE_START_CB
#define STAI_EVENT_NODE_STOP_CB

#ifdef STAI_EVENT_NODE_START_CB
#ifndef _STAI_FULL_STATEFUL_EVENT_NODE_START_CB
  #define _STAI_FULL_STATEFUL_EVENT_NODE_START_CB(_node_id, _buffers_size, ...) \
  if (net_ctx->_callback) { \
    const stai_event_node_start_stop _start_event = { \
      .node_id=(_node_id), \
      .buffers={ \
        .size=(_buffers_size), \
        .data=(stai_ptr const*)(const stai_ptr[_buffers_size])STAI_PACK(__VA_ARGS__) \
      } \
    }; \
    net_ctx->_callback(net_ctx->_callback_cookie, STAI_EVENT_NODE_START, (const void*)&_start_event); \
  }
#endif
#else
  #define _STAI_FULL_STATEFUL_EVENT_NODE_START_CB(_node_id, _buffers_size, ...) \
    do { /* _STAI_FULL_STATEFUL_EVENT_NODE_START_CB() */ } while(0);
#endif      /* STAI_EVENT_NODE_START_CB */

#ifdef STAI_EVENT_NODE_STOP_CB
#ifndef _STAI_FULL_STATEFUL_EVENT_NODE_STOP_CB
  #define _STAI_FULL_STATEFUL_EVENT_NODE_STOP_CB(_node_id, _buffers_size, ...) \
  if (net_ctx->_callback) { \
    const stai_event_node_start_stop _stop_event = { \
      .node_id=(_node_id), \
      .buffers={ \
        .size=(_buffers_size), \
        .data=(stai_ptr const*)(stai_ptr[_buffers_size])STAI_PACK(__VA_ARGS__) \
      } \
    }; \
    net_ctx->_callback(net_ctx->_callback_cookie, STAI_EVENT_NODE_STOP, (const void*)&_stop_event); \
  }
#endif
#else
  #define _STAI_FULL_STATEFUL_EVENT_NODE_STOP_CB(_node_id, _buffers_size, ...) \
    do { /* _STAI_FULL_STATEFUL_EVENT_NODE_STOP_CB() */ } while(0);
#endif      /* STAI_EVENT_NODE_STOP_CB */


/*****************************************************************************/
#define _STAI_FULL_STATEFUL_MODEL_SIGNATURE     "0x759e2d5902530abd074eafe61269e63e"
#define _STAI_FULL_STATEFUL_DATETIME            "2026-05-22T12:27:34+0900"
#define _STAI_FULL_STATEFUL_COMPILE_DATETIME    __DATE__ " " __TIME__

#define _STAI_CONTEXT_ALIGNMENT        STAI_FULL_STATEFUL_CONTEXT_ALIGNMENT

/*****************************************************************************/
#define g_full_stateful_activations_1     (NULL)



#if defined(HAVE_FULL_STATEFUL_INFO)
/*****************************************************************************/
static const stai_network_info g_full_stateful_info = {
  .model_signature = _STAI_FULL_STATEFUL_MODEL_SIGNATURE,
  .c_compile_datetime = _STAI_FULL_STATEFUL_COMPILE_DATETIME,
  .c_model_name = STAI_FULL_STATEFUL_MODEL_NAME,
  .c_model_datetime = _STAI_FULL_STATEFUL_DATETIME,
  .c_model_signature = 0x0,
  .runtime_version = STAI_INIT_VERSION(12, 0, 0),
  .tool_version = STAI_INIT_VERSION(4, 0, 0),
  .api_version = STAI_INIT_VERSION(1, 0, 0),
  .n_macc = STAI_FULL_STATEFUL_MACC_NUM,
  .n_nodes = STAI_FULL_STATEFUL_NODES_NUM,
  .flags = STAI_FULL_STATEFUL_FLAGS,
  .n_inputs = STAI_FULL_STATEFUL_IN_NUM,
  .n_outputs = STAI_FULL_STATEFUL_OUT_NUM,
  .n_activations = STAI_FULL_STATEFUL_ACTIVATIONS_NUM,
  .n_weights = STAI_FULL_STATEFUL_WEIGHTS_NUM,
  .n_states = STAI_FULL_STATEFUL_STATES_NUM,
  .inputs = (stai_tensor[STAI_FULL_STATEFUL_IN_NUM]) {
    STAI_INIT_TENSOR(
      STAI_FULL_STATEFUL_IN_1_NAME,
      STAI_FULL_STATEFUL_IN_1_FLAGS,
      STAI_FULL_STATEFUL_IN_1_FORMAT,
      STAI_FULL_STATEFUL_IN_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 3, 1, 40, 45),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    },
    .outputs = (stai_tensor[STAI_FULL_STATEFUL_OUT_NUM]) {
    STAI_INIT_TENSOR(
      STAI_FULL_STATEFUL_OUT_1_NAME,
      STAI_FULL_STATEFUL_OUT_1_FLAGS,
      STAI_FULL_STATEFUL_OUT_1_FORMAT,
      STAI_FULL_STATEFUL_OUT_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 2, 1, 2),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    },
  .activations = (stai_tensor[STAI_FULL_STATEFUL_ACTIVATIONS_NUM]) {
    STAI_INIT_TENSOR(
      (NULL),
      STAI_FULL_STATEFUL_ACTIVATION_1_FLAGS,
      STAI_FORMAT_U8,
      STAI_FULL_STATEFUL_ACTIVATION_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 22016),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    },
  .weights = (stai_tensor[STAI_FULL_STATEFUL_WEIGHTS_NUM]) {
    STAI_INIT_TENSOR(
      (NULL),
      STAI_FULL_STATEFUL_WEIGHT_1_FLAGS,
      STAI_FORMAT_U8,
      STAI_FULL_STATEFUL_WEIGHT_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 281992),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    },

  .states = NULL
};
#endif

#define _STAI_CONTEXT_ACQUIRE(_net_ctx, _net_handle) \
  _stai_full_stateful_context* _net_ctx = (_stai_full_stateful_context*)(_net_handle); \
  STAI_ASSERT(_net_ctx != NULL) \
  _STAI_SET_ERROR(_net_ctx, _net_ctx->_magic != STAI_MAGIC, \
                  STAI_ERROR_NETWORK_INVALID_CONTEXT_HANDLE, _net_ctx->_return_code)


/*****************************************************************************/
static
void _stai_full_stateful_check(_stai_full_stateful_context* net_ctx)
{
  stai_size idx;

// Check activations status
  for (idx=0; idx<STAI_FULL_STATEFUL_ACTIVATIONS_NUM; idx++) {
    if (net_ctx->_activations[idx] == NULL) break;
  }
  net_ctx->_flags |= (idx == STAI_FULL_STATEFUL_ACTIVATIONS_NUM) ? STAI_FLAG_ACTIVATIONS : STAI_FLAG_NONE;
// Check inputs status
  for (idx=0; idx<STAI_FULL_STATEFUL_IN_NUM; idx++) {
    if (net_ctx->_inputs[idx] == NULL) break;
  }
  net_ctx->_flags |= (idx == STAI_FULL_STATEFUL_IN_NUM) ? STAI_FLAG_INPUTS : STAI_FLAG_NONE;

  // Check outputs status
  for (idx=0; idx<STAI_FULL_STATEFUL_OUT_NUM; idx++) {
    if (net_ctx->_outputs[idx] == NULL) break;
  }
  net_ctx->_flags |= (idx == STAI_FULL_STATEFUL_OUT_NUM) ? STAI_FLAG_OUTPUTS : STAI_FLAG_NONE;

// Check weights status
  for (idx=0; idx<STAI_FULL_STATEFUL_WEIGHTS_NUM; idx++) {
    if (net_ctx->_weights[idx] == NULL) break;
  }
  net_ctx->_flags |= (idx == STAI_FULL_STATEFUL_WEIGHTS_NUM) ? STAI_FLAG_WEIGHTS : STAI_FLAG_NONE;
STAI_PRINT("  [_stai_network_check] flags: 0x%08x\n", net_ctx->_flags)
}


/*****************************************************************************/
STAI_API_ENTRY
stai_return_code stai_full_stateful_init(
  stai_network* network)
{
  /* Memory where to store internal context is provided by applications as a raw byte buffer */
  _stai_full_stateful_context* net_ctx = (_stai_full_stateful_context*)(network);
  net_ctx->_return_code = STAI_SUCCESS;
  STAI_PRINT("[Entering Network Init] network(%p) context_size(%d)\n", net_ctx, (int32_t)sizeof(_stai_full_stateful_context))

  _STAI_SET_ERROR(net_ctx, STAI_FULL_STATEFUL_CONTEXT_SIZE != sizeof(_stai_full_stateful_context),
                 STAI_ERROR_NETWORK_INVALID_CONTEXT_SIZE, net_ctx->_return_code)

  {
    const _stai_full_stateful_context _full_stateful_context = {
      ._magic = STAI_MAGIC,
      ._signature = STAI_FULL_STATEFUL_MODEL_SIGNATURE,
      ._flags = STAI_FULL_STATEFUL_FLAGS,
      ._return_code = STAI_SUCCESS,
      ._callback = NULL,
      ._callback_cookie = NULL,
      ._activations = {
      (stai_ptr)g_full_stateful_activations_1
      },
      ._weights = {
      (stai_ptr)g_full_stateful_weights_array
      },
      ._inputs = {
    NULL},
      ._outputs = {
    NULL},
    };

    // Deep copy of internal context to opaque buffer provided by app
    *net_ctx = _full_stateful_context;

    _stai_full_stateful_check(net_ctx);
  }

  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_full_stateful_deinit(
  stai_network* network)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  /*  Reset flags to initial state  */
  net_ctx->_flags = STAI_FULL_STATEFUL_FLAGS;
  return net_ctx->_return_code;
}

/*****************************************************************************/





/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  conv1d_1_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2560, AI_STATIC)

/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  gru_output0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2560, AI_STATIC)

/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  gru_kernel_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12288, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  gru_recurrent_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12288, AI_STATIC)

/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  gru_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 384, AI_STATIC)

/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  gru_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 384, AI_STATIC)



/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  conv1d_1_output, AI_STATIC,
  4, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 40), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &conv1d_1_output_array, NULL)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  gru_bias, AI_STATIC,
  23, 0x0,
  AI_SHAPE_INIT(4, 192, 2, 1, 1), AI_STRIDE_INIT(4, 4, 768, 1536, 1536),
  1, &gru_bias_array, NULL)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  gru_kernel, AI_STATIC,
  24, 0x0,
  AI_SHAPE_INIT(4, 64, 192, 1, 1), AI_STRIDE_INIT(4, 4, 256, 49152, 49152),
  1, &gru_kernel_array, NULL)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  gru_output0, AI_STATIC,
  25, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 40), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &gru_output0_array, NULL)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  gru_recurrent, AI_STATIC,
  26, 0x0,
  AI_SHAPE_INIT(4, 64, 192, 1, 1), AI_STRIDE_INIT(4, 4, 256, 49152, 49152),
  1, &gru_recurrent_array, NULL)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  gru_scratch0, AI_STATIC,
  27, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 4, 4, 1536, 1536),
  1, &gru_scratch0_array, NULL)


AI_TENSOR_CHAIN_OBJ_DECLARE(
  gru_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv1d_1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gru_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 6, &gru_kernel, &gru_recurrent, NULL, NULL, &gru_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gru_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gru_layer, 3,
  GRU_TYPE, 0x0, NULL,
  gru, forward_gru,
  &gru_chain,
  NULL, &gru_layer, AI_STATIC, 
  .n_units = 64, 
  .activation_nl = nl_func_tanh_array_f32, 
  .go_backwards = false, 
  .reverse_seq = false, 
  .return_state = false, 
  .reset_after = true, 
  .recurrent_nl = nl_func_sigmoid_array_f32, 
  .state = AI_HANDLE_PTR(NULL), 
  .init = AI_LAYER_FUNC(NULL), 
  .destroy = AI_LAYER_FUNC(NULL), 
)
/**  Hybrid layers declarations section  *************************************/
void forward_lite_gru_gru(_stai_full_stateful_context* net_ctx)
{
  conv1d_1_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  conv1d_1_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  gru_kernel_array.data = AI_PTR(net_ctx->_weights[0] + 140032);
  gru_kernel_array.data_start = AI_PTR(net_ctx->_weights[0] + 140032);
  gru_recurrent_array.data = AI_PTR(net_ctx->_weights[0] + 189184);
  gru_recurrent_array.data_start = AI_PTR(net_ctx->_weights[0] + 189184);
  gru_bias_array.data = AI_PTR(net_ctx->_weights[0] + 238336);
  gru_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 238336);
  gru_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 10240);
  gru_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 10240);
  gru_output0_array.data = AI_PTR(net_ctx->_activations[0] + 11776);
  gru_output0_array.data_start = AI_PTR(net_ctx->_activations[0] + 11776);
  _STAI_FULL_STATEFUL_EVENT_NODE_START_CB(3, 1, { conv1d_1_output.data->data});
  forward_gru(&gru_layer);
  _STAI_FULL_STATEFUL_EVENT_NODE_STOP_CB(3, 1, { gru_output0.data->data});
}

/*****************************************************************************/


static const ai_u32 conv1d_conv2d_t_in_0_shape_ch_const_u32 = 45;
static const ai_u32 conv1d_conv2d_t_out_0_shape_ch_const_u32 = 64;
static const ai_u32 conv1d_conv2d_t_in_0_shape_w_const_u32 = 1;
static const ai_u32 conv1d_conv2d_t_in_0_shape_h_const_u32 = 40;
static const ai_u32 conv1d_conv2d_t_out_0_shape_w_const_u32 = 1;
static const ai_u32 conv1d_conv2d_t_out_0_shape_h_const_u32 = 40;
static const ai_u32 conv1d_conv2d_t_weight_0_shape_w_const_u32 = 1;
static const ai_u32 conv1d_conv2d_t_weight_0_shape_h_const_u32 = 5;
static const ai_i32 conv1d_conv2d_l_pad_W_0_const_s32 = 0;
static const ai_i32 conv1d_conv2d_l_pad_H_0_const_s32 = 4;
static const ai_u16 conv1d_conv2d_l_stride_1_const_u16 = 1;
static const ai_u16 conv1d_conv2d_l_stride_0_const_u16 = 1;
static const ai_u16 conv1d_conv2d_l_dilation_H_const_u16 = 1;
static const ai_u16 conv1d_conv2d_l_dilation_W_const_u16 = 1;

static const ai_i32 conv1d_t_in_0_shape_ch_h_prod_const_s32 = 2560;

static const ai_u32 conv1d_1_conv2d_t_in_0_shape_ch_const_u32 = 64;
static const ai_u32 conv1d_1_conv2d_t_out_0_shape_ch_const_u32 = 64;
static const ai_u32 conv1d_1_conv2d_t_in_0_shape_w_const_u32 = 1;
static const ai_u32 conv1d_1_conv2d_t_in_0_shape_h_const_u32 = 40;
static const ai_u32 conv1d_1_conv2d_t_out_0_shape_w_const_u32 = 1;
static const ai_u32 conv1d_1_conv2d_t_out_0_shape_h_const_u32 = 40;
static const ai_u32 conv1d_1_conv2d_t_weight_0_shape_w_const_u32 = 1;
static const ai_u32 conv1d_1_conv2d_t_weight_0_shape_h_const_u32 = 5;
static const ai_i32 conv1d_1_conv2d_l_pad_W_0_const_s32 = 0;
static const ai_i32 conv1d_1_conv2d_l_pad_H_0_const_s32 = 4;
static const ai_u16 conv1d_1_conv2d_l_stride_1_const_u16 = 1;
static const ai_u16 conv1d_1_conv2d_l_stride_0_const_u16 = 1;
static const ai_u16 conv1d_1_conv2d_l_dilation_H_const_u16 = 1;
static const ai_u16 conv1d_1_conv2d_l_dilation_W_const_u16 = 1;

static const ai_i32 conv1d_1_t_in_0_shape_ch_h_prod_const_s32 = 2560;


static const ai_size gru_1_t_in_0_shape_h_const_size = 40;
static const ai_size gru_1_t_in_0_shape_ch_const_size = 64;


static const ai_i32 dense_t_in_0_shape_ch_prod_const_s32 = 32;


static const ai_i32 dense_1_t_in_0_shape_ch_prod_const_s32 = 2;
STAI_API_ENTRY
stai_return_code stai_full_stateful_run(
  stai_network* network,
  const stai_run_mode mode)
{
   STAI_UNUSED(mode)
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  _STAI_SET_ERROR(net_ctx, (net_ctx->_flags & STAI_FLAG_ACTIVATIONS) != STAI_FLAG_ACTIVATIONS,
        STAI_ERROR_NETWORK_INVALID_ACTIVATIONS_PTR, net_ctx->_return_code)

  _STAI_SET_ERROR(net_ctx, (net_ctx->_flags & STAI_FLAG_INPUTS) != STAI_FLAG_INPUTS,
                  STAI_ERROR_NETWORK_INVALID_IN_PTR, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, (net_ctx->_flags & STAI_FLAG_OUTPUTS) != STAI_FLAG_OUTPUTS,
                  STAI_ERROR_NETWORK_INVALID_OUT_PTR, net_ctx->_return_code)

  _STAI_SET_ERROR(net_ctx, (net_ctx->_flags & STAI_FLAG_WEIGHTS) != STAI_FLAG_WEIGHTS,
                  STAI_ERROR_NETWORK_INVALID_WEIGHTS_PTR, net_ctx->_return_code)


  /* LITE_KERNEL_SECTION BEGIN conv1d_conv2d */
  {
      const ai_float* conv1d_conv2d_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_inputs[0] + 0);
    ai_float* conv1d_conv2d_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 1536);
    const ai_u8* conv1d_conv2d_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 0);
    const ai_u8* conv1d_conv2d_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 57600);
    ai_float* conv1d_conv2d_t_scratch_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 13388);
  
  _STAI_FULL_STATEFUL_EVENT_NODE_START_CB(1, 1, {(stai_ptr) conv1d_conv2d_t_in_0_ptr_const_f32});
    
  forward_lite_conv2d_if32of32wf32(conv1d_conv2d_t_in_0_ptr_const_f32, conv1d_conv2d_t_out_0_ptr_f32, conv1d_conv2d_t_weight_0_ptr_const_u8, conv1d_conv2d_t_weight_1_ptr_const_u8, conv1d_conv2d_t_scratch_0_ptr_f32, conv1d_conv2d_t_in_0_shape_ch_const_u32, conv1d_conv2d_t_out_0_shape_ch_const_u32, conv1d_conv2d_t_in_0_shape_w_const_u32, conv1d_conv2d_t_in_0_shape_h_const_u32, conv1d_conv2d_t_out_0_shape_w_const_u32, conv1d_conv2d_t_out_0_shape_h_const_u32, conv1d_conv2d_t_weight_0_shape_w_const_u32, conv1d_conv2d_t_weight_0_shape_h_const_u32, conv1d_conv2d_l_pad_W_0_const_s32, conv1d_conv2d_l_pad_H_0_const_s32, conv1d_conv2d_l_stride_1_const_u16, conv1d_conv2d_l_stride_0_const_u16, 5, 1, conv1d_conv2d_l_dilation_H_const_u16, conv1d_conv2d_l_dilation_W_const_u16, (ai_size)(1));
    
  _STAI_FULL_STATEFUL_EVENT_NODE_STOP_CB(1, 1, {(stai_ptr) conv1d_conv2d_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END conv1d_conv2d */
  /* LITE_KERNEL_SECTION BEGIN conv1d */
  {
      ai_handle conv1d_t_out_0_ptr_handle = (ai_handle)(net_ctx->_activations[0] + 1536);
    const ai_handle conv1d_t_in_0_ptr_const_handle = (ai_handle)(net_ctx->_activations[0] + 1536);
  
  _STAI_FULL_STATEFUL_EVENT_NODE_START_CB(1, 1, {(stai_ptr) conv1d_t_in_0_ptr_const_handle});
    
  forward_lite_nl_relu_if32of32(conv1d_t_out_0_ptr_handle, conv1d_t_in_0_ptr_const_handle, conv1d_t_in_0_shape_ch_h_prod_const_s32, NULL);
    
  _STAI_FULL_STATEFUL_EVENT_NODE_STOP_CB(1, 1, {(stai_ptr) conv1d_t_out_0_ptr_handle});
  }
  /* LITE_KERNEL_SECTION END conv1d */
  /* LITE_KERNEL_SECTION BEGIN conv1d_1_conv2d */
  {
      const ai_float* conv1d_1_conv2d_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 1536);
    ai_float* conv1d_1_conv2d_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 0);
    const ai_u8* conv1d_1_conv2d_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 57856);
    const ai_u8* conv1d_1_conv2d_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 139776);
    ai_float* conv1d_1_conv2d_t_scratch_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 11776);
  
  _STAI_FULL_STATEFUL_EVENT_NODE_START_CB(2, 1, {(stai_ptr) conv1d_1_conv2d_t_in_0_ptr_const_f32});
    
  forward_lite_conv2d_if32of32wf32(conv1d_1_conv2d_t_in_0_ptr_const_f32, conv1d_1_conv2d_t_out_0_ptr_f32, conv1d_1_conv2d_t_weight_0_ptr_const_u8, conv1d_1_conv2d_t_weight_1_ptr_const_u8, conv1d_1_conv2d_t_scratch_0_ptr_f32, conv1d_1_conv2d_t_in_0_shape_ch_const_u32, conv1d_1_conv2d_t_out_0_shape_ch_const_u32, conv1d_1_conv2d_t_in_0_shape_w_const_u32, conv1d_1_conv2d_t_in_0_shape_h_const_u32, conv1d_1_conv2d_t_out_0_shape_w_const_u32, conv1d_1_conv2d_t_out_0_shape_h_const_u32, conv1d_1_conv2d_t_weight_0_shape_w_const_u32, conv1d_1_conv2d_t_weight_0_shape_h_const_u32, conv1d_1_conv2d_l_pad_W_0_const_s32, conv1d_1_conv2d_l_pad_H_0_const_s32, conv1d_1_conv2d_l_stride_1_const_u16, conv1d_1_conv2d_l_stride_0_const_u16, 5, 1, conv1d_1_conv2d_l_dilation_H_const_u16, conv1d_1_conv2d_l_dilation_W_const_u16, (ai_size)(1));
    
  _STAI_FULL_STATEFUL_EVENT_NODE_STOP_CB(2, 1, {(stai_ptr) conv1d_1_conv2d_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END conv1d_1_conv2d */
  /* LITE_KERNEL_SECTION BEGIN conv1d_1 */
  {
      ai_handle conv1d_1_t_out_0_ptr_handle = (ai_handle)(net_ctx->_activations[0] + 0);
    const ai_handle conv1d_1_t_in_0_ptr_const_handle = (ai_handle)(net_ctx->_activations[0] + 0);
  
  _STAI_FULL_STATEFUL_EVENT_NODE_START_CB(2, 1, {(stai_ptr) conv1d_1_t_in_0_ptr_const_handle});
    
  forward_lite_nl_relu_if32of32(conv1d_1_t_out_0_ptr_handle, conv1d_1_t_in_0_ptr_const_handle, conv1d_1_t_in_0_shape_ch_h_prod_const_s32, NULL);
    
  _STAI_FULL_STATEFUL_EVENT_NODE_STOP_CB(2, 1, {(stai_ptr) conv1d_1_t_out_0_ptr_handle});
  }
  /* LITE_KERNEL_SECTION END conv1d_1 */
  /* LITE_KERNEL_SECTION BEGIN gru */
  {
    
  forward_lite_gru_gru(net_ctx);
  }
  /* LITE_KERNEL_SECTION END gru */
  /* LITE_KERNEL_SECTION BEGIN gru_1 */
  {
      ai_float* gru_1_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 768);
    const ai_float* gru_1_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 11776);
    const ai_float* gru_1_t_weight_0_ptr_const_f32 = (ai_float*)(net_ctx->_weights[0] + 239872);
    const ai_float* gru_1_t_weight_1_ptr_const_f32 = (ai_float*)(net_ctx->_weights[0] + 264448);
    const ai_float* gru_1_t_weight_2_ptr_const_f32 = (ai_float*)(net_ctx->_weights[0] + 276736);
    ai_float* gru_1_t_scratch_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 0);
  
  _STAI_FULL_STATEFUL_EVENT_NODE_START_CB(4, 1, {(stai_ptr) gru_1_t_in_0_ptr_const_f32});
    
  forward_lite_gru_if32of32wf32(gru_1_t_out_0_ptr_f32, gru_1_t_in_0_ptr_const_f32, gru_1_t_weight_0_ptr_const_f32, gru_1_t_weight_1_ptr_const_f32, gru_1_t_weight_2_ptr_const_f32, gru_1_t_scratch_0_ptr_f32, (ai_u32)(32), gru_1_t_in_0_shape_h_const_size, gru_1_t_in_0_shape_ch_const_size, AI_HANDLE_PTR(forward_lite_nl_tanh_if32of32), AI_HANDLE_PTR(forward_lite_nl_sigmoid_if32of32), false, false, false, true, NULL, NULL, NULL);
    
  _STAI_FULL_STATEFUL_EVENT_NODE_STOP_CB(4, 1, {(stai_ptr) gru_1_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END gru_1 */
  /* LITE_KERNEL_SECTION BEGIN dense_dense */
  {
      forward_lite_dense_if32of32wf32_args arg_30f51e = {
      .output = (float*)(net_ctx->_activations[0] + 0),
      .input = (float*)(net_ctx->_activations[0] + 768),
      .weights = (float*)(net_ctx->_weights[0] + 277504),
      .bias = (float*)(net_ctx->_weights[0] + 281600),
      .n_channel_in = 32,
      .n_channel_out = 32,
      .n_elements = 1,
    };
  
  _STAI_FULL_STATEFUL_EVENT_NODE_START_CB(5, 1, {(stai_ptr) (float*)(net_ctx->_activations[0] + 768)});
    
  forward_lite_dense_if32of32wf32((forward_lite_dense_if32of32wf32_args*)&arg_30f51e);
    
  _STAI_FULL_STATEFUL_EVENT_NODE_STOP_CB(5, 1, {(stai_ptr) (float*)(net_ctx->_activations[0] + 0)});
  }
  /* LITE_KERNEL_SECTION END dense_dense */
  /* LITE_KERNEL_SECTION BEGIN dense */
  {
      ai_handle dense_t_out_0_ptr_handle = (ai_handle)(net_ctx->_activations[0] + 128);
    const ai_handle dense_t_in_0_ptr_const_handle = (ai_handle)(net_ctx->_activations[0] + 0);
  
  _STAI_FULL_STATEFUL_EVENT_NODE_START_CB(5, 1, {(stai_ptr) dense_t_in_0_ptr_const_handle});
    
  forward_lite_nl_relu_if32of32(dense_t_out_0_ptr_handle, dense_t_in_0_ptr_const_handle, dense_t_in_0_shape_ch_prod_const_s32, NULL);
    
  _STAI_FULL_STATEFUL_EVENT_NODE_STOP_CB(5, 1, {(stai_ptr) dense_t_out_0_ptr_handle});
  }
  /* LITE_KERNEL_SECTION END dense */
  /* LITE_KERNEL_SECTION BEGIN dense_1_dense */
  {
      forward_lite_dense_if32of32wf32_args arg_30f51e = {
      .output = (float*)(net_ctx->_activations[0] + 0),
      .input = (float*)(net_ctx->_activations[0] + 128),
      .weights = (float*)(net_ctx->_weights[0] + 281728),
      .bias = (float*)(net_ctx->_weights[0] + 281984),
      .n_channel_in = 32,
      .n_channel_out = 2,
      .n_elements = 1,
    };
  
  _STAI_FULL_STATEFUL_EVENT_NODE_START_CB(7, 1, {(stai_ptr) (float*)(net_ctx->_activations[0] + 128)});
    
  forward_lite_dense_if32of32wf32((forward_lite_dense_if32of32wf32_args*)&arg_30f51e);
    
  _STAI_FULL_STATEFUL_EVENT_NODE_STOP_CB(7, 1, {(stai_ptr) (float*)(net_ctx->_activations[0] + 0)});
  }
  /* LITE_KERNEL_SECTION END dense_1_dense */
  /* LITE_KERNEL_SECTION BEGIN dense_1 */
  {
      ai_handle dense_1_t_out_0_ptr_handle = (ai_handle)(net_ctx->_outputs[0] + 0);
    const ai_handle dense_1_t_in_0_ptr_const_handle = (ai_handle)(net_ctx->_activations[0] + 0);
  
  _STAI_FULL_STATEFUL_EVENT_NODE_START_CB(7, 1, {(stai_ptr) dense_1_t_in_0_ptr_const_handle});
    
  forward_lite_nl_softmax_if32of32(dense_1_t_out_0_ptr_handle, dense_1_t_in_0_ptr_const_handle, dense_1_t_in_0_shape_ch_prod_const_s32, 1, 2);
    
  _STAI_FULL_STATEFUL_EVENT_NODE_STOP_CB(7, 1, {(stai_ptr) dense_1_t_out_0_ptr_handle});
  }
  /* LITE_KERNEL_SECTION END dense_1 */
  return net_ctx->_return_code;
}

/*****************************************************************************/
/*  Getters APIs Section  */
STAI_API_ENTRY
stai_size stai_full_stateful_get_context_size()
{
  return (stai_size)STAI_FULL_STATEFUL_CONTEXT_SIZE;
}

#if defined(HAVE_FULL_STATEFUL_INFO)
STAI_API_ENTRY
stai_return_code stai_full_stateful_get_info(
  stai_network* network,
  stai_network_info* info)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, info==NULL, STAI_ERROR_NETWORK_INVALID_INFO, net_ctx->_return_code)

  // Copy of network info struct
  *info = g_full_stateful_info;

  return STAI_SUCCESS;
}
#endif


STAI_API_ENTRY
stai_return_code stai_full_stateful_get_activations(
  stai_network* network, stai_ptr* activations, stai_size* n_activations)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  _STAI_SET_ERROR(net_ctx, !n_activations, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  *n_activations = STAI_FULL_STATEFUL_ACTIVATIONS_NUM;
for (stai_size idx=0; activations && (idx<STAI_FULL_STATEFUL_ACTIVATIONS_NUM); idx++) {
    // get address of the activations buffers
    activations[idx] = net_ctx->_activations[idx];
  }return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_full_stateful_get_weights(
  stai_network* network, stai_ptr* weights, stai_size* n_weights)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !n_weights, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  *n_weights = STAI_FULL_STATEFUL_WEIGHTS_NUM;
for (stai_size idx=0; weights && (idx<STAI_FULL_STATEFUL_WEIGHTS_NUM); idx++) {
    // get address of the weights buffers
    weights[idx] = net_ctx->_weights[idx];
  }return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_full_stateful_get_inputs(
  stai_network* network, stai_ptr* inputs, stai_size* n_inputs)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !n_inputs, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  *n_inputs = STAI_FULL_STATEFUL_IN_NUM;
  for (stai_size idx=0; inputs && (idx<STAI_FULL_STATEFUL_IN_NUM); idx++) {
    inputs[idx] = net_ctx->_inputs[idx];
  }
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_full_stateful_get_outputs(
  stai_network* network, stai_ptr* outputs, stai_size* n_outputs)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !n_outputs, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  *n_outputs = STAI_FULL_STATEFUL_OUT_NUM;
  for (stai_size idx=0; outputs && (idx<STAI_FULL_STATEFUL_OUT_NUM); idx++) {
    outputs[idx] = net_ctx->_outputs[idx];
  }
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_full_stateful_get_error(
  stai_network* network)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  /* return 1st generated error or STAI_SUCCESS if no errors so far */
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_full_stateful_get_states(
  stai_network* network, stai_ptr* states, stai_size* n_states)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !n_states, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  /* get the number of internals states (supporting multi-heap also for internal states) */
  *n_states = STAI_FULL_STATEFUL_STATES_NUM;

  STAI_UNUSED(states)
return net_ctx->_return_code;
}


/*****************************************************************************/
/*  Setters APIs Section  */

STAI_API_ENTRY
stai_return_code stai_full_stateful_set_activations(
  stai_network* network,
  const stai_ptr* activations,
  const stai_size n_activations)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
const uintptr_t _activations_alignment[] = STAI_FULL_STATEFUL_ACTIVATIONS_ALIGNMENTS;
  STAI_PRINT("  [stai_full_stateful_set_activations] network(%p) activations[%d]: %p\n\n", net_ctx, n_activations, activations)
  _STAI_SET_ERROR(net_ctx, !activations,
                  STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, n_activations!=STAI_FULL_STATEFUL_ACTIVATIONS_NUM,
                  STAI_ERROR_NETWORK_INVALID_ACTIVATIONS_NUM, net_ctx->_return_code)

  for (stai_size idx=0; activations && idx<STAI_FULL_STATEFUL_ACTIVATIONS_NUM; idx++) {
    STAI_PRINT("  activation[%d]: %p\n", idx, activations[idx])
    _STAI_SET_ERROR(net_ctx, activations[idx]==NULL,
                    STAI_ERROR_NETWORK_INVALID_ACTIVATIONS_PTR, net_ctx->_return_code)
    _STAI_SET_ERROR(net_ctx, ((uintptr_t)activations[idx]) & (_activations_alignment[idx]-1),
                    STAI_ERROR_INVALID_BUFFER_ALIGNMENT, net_ctx->_return_code)
    net_ctx->_activations[idx] = activations[idx];
  }
  net_ctx->_inputs[0] = activations[0] + 6188;

  net_ctx->_outputs[0] = activations[0] + 8;
_stai_full_stateful_check(net_ctx);
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_full_stateful_set_weights(
  stai_network* network,
  const stai_ptr* weights,
  const stai_size n_weights)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
const uintptr_t _weights_alignment[] = STAI_FULL_STATEFUL_WEIGHTS_ALIGNMENTS;
  _STAI_SET_ERROR(net_ctx, !weights,
                  STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, n_weights!=STAI_FULL_STATEFUL_WEIGHTS_NUM,
                  STAI_ERROR_NETWORK_INVALID_WEIGHTS_NUM, net_ctx->_return_code)
  for (stai_size idx=0; weights && idx<STAI_FULL_STATEFUL_WEIGHTS_NUM; idx++) {
    STAI_PRINT("  weight[%d]: %p\n", idx, weights[idx])
    _STAI_SET_ERROR(net_ctx, weights[idx]==NULL,
                    STAI_ERROR_NETWORK_INVALID_WEIGHTS_PTR, net_ctx->_return_code)
    _STAI_SET_ERROR(net_ctx, ((uintptr_t)weights[idx]) & (_weights_alignment[idx]-1),
                    STAI_ERROR_INVALID_BUFFER_ALIGNMENT, net_ctx->_return_code)
    net_ctx->_weights[idx] = weights[idx];
  }_stai_full_stateful_check(net_ctx);
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_full_stateful_set_inputs(
  stai_network* network,
  const stai_ptr* inputs,
  const stai_size n_inputs)
{
  const uintptr_t _inputs_alignment[] = STAI_FULL_STATEFUL_IN_ALIGNMENTS;
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !inputs,
                  STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, n_inputs!=STAI_FULL_STATEFUL_IN_NUM,
                  STAI_ERROR_NETWORK_INVALID_IN_NUM, net_ctx->_return_code)

  for (stai_size idx=0; inputs && idx<STAI_FULL_STATEFUL_IN_NUM; idx++) {
    STAI_PRINT("  input[%d]: %p\n", idx, inputs[idx])
    _STAI_SET_ERROR(net_ctx, inputs[idx]==NULL,
                    STAI_ERROR_NETWORK_INVALID_IN_PTR, net_ctx->_return_code)
    _STAI_SET_ERROR(net_ctx, ((uintptr_t)inputs[idx]) & (_inputs_alignment[idx]-1),
                    STAI_ERROR_INVALID_BUFFER_ALIGNMENT, net_ctx->_return_code)
    net_ctx->_inputs[idx] = inputs[idx];
  }

  _stai_full_stateful_check(net_ctx);
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_full_stateful_set_outputs(
  stai_network* network,
  const stai_ptr* outputs,
  const stai_size n_outputs)
{
  const uintptr_t _outputs_alignment[] = STAI_FULL_STATEFUL_OUT_ALIGNMENTS;
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !outputs,
                  STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, n_outputs!=STAI_FULL_STATEFUL_OUT_NUM,
                  STAI_ERROR_NETWORK_INVALID_OUT_NUM, net_ctx->_return_code)

  for (stai_size idx=0; outputs && idx<n_outputs; idx++) {
    STAI_PRINT("  output[%d]: %p\n", idx, outputs[idx])
    _STAI_SET_ERROR(net_ctx, outputs[idx]==NULL,
                    STAI_ERROR_NETWORK_INVALID_OUT_PTR, net_ctx->_return_code)
    _STAI_SET_ERROR(net_ctx, ((uintptr_t)outputs[idx]) & (_outputs_alignment[idx]-1),
                    STAI_ERROR_INVALID_BUFFER_ALIGNMENT, net_ctx->_return_code)
    net_ctx->_outputs[idx] = outputs[idx];
  }

  _stai_full_stateful_check(net_ctx);
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_full_stateful_set_states(
  stai_network* network,
  const stai_ptr* states,
  const stai_size n_states)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  STAI_UNUSED(states)
  STAI_UNUSED(n_states)
_stai_full_stateful_check(net_ctx);
  return net_ctx->_return_code;
}

STAI_API_ENTRY
stai_return_code stai_full_stateful_set_callback(
  stai_network* network, const stai_event_cb cb, void* cb_cookie)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  STAI_PRINT("  set_callback %p cb %p cookie %p\n", net_ctx, cb, cb_cookie)
  // _STAI_SET_ERROR(net_ctx, cb==NULL, STAI_ERROR_NETWORK_INVALID_CALLBACK, net_ctx->_return_code)
  net_ctx->_callback = cb;
  net_ctx->_callback_cookie = cb_cookie;
  return net_ctx->_return_code;
}

#undef _STAI_SET_ERROR
#undef _STAI_CONTEXT_ALIGNMENT
#undef _STAI_CONTEXT_ACQUIRE
#undef _STAI_FULL_STATEFUL_EVENT_NODE_START_CB
#undef _STAI_FULL_STATEFUL_EVENT_NODE_STOP_CB
#undef _STAI_FULL_STATEFUL_MODEL_SIGNATURE
#undef _STAI_FULL_STATEFUL_DATETIME
#undef _STAI_FULL_STATEFUL_COMPILE_DATETIME

