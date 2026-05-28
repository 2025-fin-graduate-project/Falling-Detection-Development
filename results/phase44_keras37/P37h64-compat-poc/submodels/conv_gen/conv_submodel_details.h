/**
  ******************************************************************************
  * @file    conv_submodel.h
  * @date    2026-05-22T12:25:11+0900
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
#ifndef STAI_CONV_SUBMODEL_DETAILS_H
#define STAI_CONV_SUBMODEL_DETAILS_H

#include "stai.h"
#include "layers.h"

const stai_network_details g_conv_submodel_details = {
  .tensors = (const stai_tensor[5]) {
   { .size_bytes = 7200, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 40, 45}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "input_layer_output" },
   { .size_bytes = 10240, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 40, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "conv1d_conv2d_output" },
   { .size_bytes = 10240, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 40, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "conv1d_output" },
   { .size_bytes = 10240, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 40, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "conv1d_1_conv2d_output" },
   { .size_bytes = 10240, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 40, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "conv1d_1_output" }
  },
  .nodes = (const stai_node_details[4]){
    {.id = 1, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){0}}, .output_tensors = {1, (const int32_t[1]){1}} }, /* conv1d_conv2d */
    {.id = 1, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){1}}, .output_tensors = {1, (const int32_t[1]){2}} }, /* conv1d */
    {.id = 2, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){2}}, .output_tensors = {1, (const int32_t[1]){3}} }, /* conv1d_1_conv2d */
    {.id = 2, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){3}}, .output_tensors = {1, (const int32_t[1]){4}} } /* conv1d_1 */
  },
  .n_nodes = 4
};
#endif

