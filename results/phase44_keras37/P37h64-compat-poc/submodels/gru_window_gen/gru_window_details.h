/**
  ******************************************************************************
  * @file    gru_window.h
  * @date    2026-05-22T12:26:49+0900
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
#ifndef STAI_GRU_WINDOW_DETAILS_H
#define STAI_GRU_WINDOW_DETAILS_H

#include "stai.h"
#include "layers.h"

const stai_network_details g_gru_window_details = {
  .tensors = (const stai_tensor[3]) {
   { .size_bytes = 10240, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 40, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "conv_features_window_output" },
   { .size_bytes = 10240, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 40, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "gru_window_1_output0" },
   { .size_bytes = 128, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 32}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "gru_window_2_output0" }
  },
  .nodes = (const stai_node_details[2]){
    {.id = 1, .type = AI_LAYER_GRU_TYPE, .input_tensors = {1, (const int32_t[1]){0}}, .output_tensors = {1, (const int32_t[1]){1}} }, /* gru_window_1 */
    {.id = 2, .type = AI_LAYER_GRU_TYPE, .input_tensors = {1, (const int32_t[1]){1}}, .output_tensors = {1, (const int32_t[1]){2}} } /* gru_window_2 */
  },
  .n_nodes = 2
};
#endif

