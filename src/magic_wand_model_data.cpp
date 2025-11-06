/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "magic_wand_model_data.h"

// 参照元の .cpp を直接取り込み、長大なバイト配列定義を再利用します。
// 参照元は `const int g_magic_wand_model_data_len` なので、
// 一時的に別名へ置換し、最後に `unsigned int` で本シンボルを定義し直します。
#define g_magic_wand_model_data_len g_magic_wand_model_data_len_int
#include "magic_wand_model_data copy.cpp"
#undef g_magic_wand_model_data_len

// 参照元で定義された整数長（int）を取り込み
extern const int g_magic_wand_model_data_len_int;
// 本プロジェクトの想定シンボル（unsigned int）に合わせて公開
const unsigned int g_magic_wand_model_data_len =
    static_cast<unsigned int>(g_magic_wand_model_data_len_int);

