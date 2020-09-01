/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_EXAMPLES_SSD_CLASSIFIER_SSD_CLASSIFIER_H_
#define TENSORFLOW_LITE_EXAMPLES_SSD_CLASSIFIER_SSD_CLASSIFIER_H_

#include "tensorflow/lite/model.h"
#include "tensorflow/lite/string_type.h"

namespace tflite {
namespace ssd_classifier {

using TfLiteDelegatePtr = tflite::Interpreter::TfLiteDelegatePtr;
using TfLiteDelegatePtrMap = std::map<std::string, TfLiteDelegatePtr>;
using TfLiteFlatBufferModelPtr = std::unique_ptr<tflite::FlatBufferModel>;
using TfLiteInterpreterPtr = std::unique_ptr<tflite::Interpreter>;


struct Settings {
  int verbose = 0;
  TfLiteType input_type = kTfLiteFloat32;
  bool allow_fp16 = false;
  int loop_count = 1;
  float input_mean = 127.5f;
  float input_std = 127.5f;
  string model_name = "";
  tflite::FlatBufferModel* model;
  string input_bmp_name = "";
  string labels_file_name = "";
  int number_of_threads = 1;
  int number_of_results = 5;
  int number_of_warmup_runs = 2;
};

class SsdClassifier{
  public:
  SsdClassifier(Settings *s);

  // Takes a file name, and loads a list of labels from it, one per line, and
  // returns a vector of the strings. It pads with empty strings so the length
  // of the result is a multiple of 16, because our model expects that.
  TfLiteStatus ReadLabelsFile(const string& file_name,
                            std::vector<string>* result,
                            size_t* found_label_count);
  void RunInference();

  private:

  TfLiteFlatBufferModelPtr model_;
  TfLiteInterpreterPtr interpreter_;
  Settings *s_;
  int input_tf_idx_;
  int input_tf_height_;
  int input_tf_width_;
  int input_tf_channel_;

  std::vector<string> labels_;
  size_t label_count_;
};



}  // namespace ssd_classifier
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXAMPLES_SSD_CLASSIFIER_SSD_CLASSIFIER_H_
