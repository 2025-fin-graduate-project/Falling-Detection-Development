/**
  ******************************************************************************
  * @file    full_model.h
  * @date    2026-05-22T12:27:20+0900
  * @brief   ST.AI Tool Automatic Code Generator for Embedded NN computing
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
#ifndef STAI_FULL_MODEL_DETAILS_H
#define STAI_FULL_MODEL_DETAILS_H

#include "stai.h"
#include "layers.h"

const stai_network_details g_full_model_details = {
  .tensors = (const stai_tensor[11]) {
   { .size_bytes = 7200, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 40, 45}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "input_layer_output" },
   { .size_bytes = 10240, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 40, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "conv1d_conv2d_output" },
   { .size_bytes = 10240, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 40, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "conv1d_output" },
   { .size_bytes = 10240, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 40, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "conv1d_1_conv2d_output" },
   { .size_bytes = 10240, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 40, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "conv1d_1_output" },
   { .size_bytes = 10240, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 40, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "gru_output0" },
   { .size_bytes = 128, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 32}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "gru_1_output0" },
   { .size_bytes = 128, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 32}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "dense_dense_output" },
   { .size_bytes = 128, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 32}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "dense_output" },
   { .size_bytes = 8, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 2}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "dense_1_dense_output" },
   { .size_bytes = 8, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 2}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "dense_1_output" }
  },
  .nodes = (const stai_node_details[10]){
    {.id = 1, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){0}}, .output_tensors = {1, (const int32_t[1]){1}} }, /* conv1d_conv2d */
    {.id = 1, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){1}}, .output_tensors = {1, (const int32_t[1]){2}} }, /* conv1d */
    {.id = 2, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){2}}, .output_tensors = {1, (const int32_t[1]){3}} }, /* conv1d_1_conv2d */
    {.id = 2, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){3}}, .output_tensors = {1, (const int32_t[1]){4}} }, /* conv1d_1 */
    {.id = 3, .type = AI_LAYER_GRU_TYPE, .input_tensors = {1, (const int32_t[1]){4}}, .output_tensors = {1, (const int32_t[1]){5}} }, /* gru */
    {.id = 4, .type = AI_LAYER_GRU_TYPE, .input_tensors = {1, (const int32_t[1]){5}}, .output_tensors = {1, (const int32_t[1]){6}} }, /* gru_1 */
    {.id = 5, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){6}}, .output_tensors = {1, (const int32_t[1]){7}} }, /* dense_dense */
    {.id = 5, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){7}}, .output_tensors = {1, (const int32_t[1]){8}} }, /* dense */
    {.id = 7, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){8}}, .output_tensors = {1, (const int32_t[1]){9}} }, /* dense_1_dense */
    {.id = 7, .type = AI_LAYER_SM_TYPE, .input_tensors = {1, (const int32_t[1]){9}}, .output_tensors = {1, (const int32_t[1]){10}} } /* dense_1 */
  },
  .n_nodes = 10
};
#endif

