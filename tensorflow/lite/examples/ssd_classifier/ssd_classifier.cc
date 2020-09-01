#include "tensorflow/lite/examples/ssd_classifier/ssd_classifier.h"

#include <fcntl.h>      // NOLINT(build/include_order)
#include <getopt.h>     // NOLINT(build/include_order)
#include <sys/time.h>   // NOLINT(build/include_order)
#include <sys/types.h>  // NOLINT(build/include_order)
#include <sys/uio.h>    // NOLINT(build/include_order)
#include <unistd.h>     // NOLINT(build/include_order)

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "tensorflow/lite/examples/ssd_classifier/bitmap_helpers.h"
#include "tensorflow/lite/examples/ssd_classifier/get_top_n.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/tools/evaluation/utils.h"

#define LOG(x) std::cerr

namespace tflite {
namespace ssd_classifier {

double get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

SsdClassifier::SsdClassifier(Settings *s):s_(s){

  if (s_->model_name.empty()) {
    LOG(ERROR) << "no model file name\n";
    exit(-1);
  }

  model_ = tflite::FlatBufferModel::BuildFromFile(s_->model_name.c_str());
  if (!model_) {
    LOG(FATAL) << "\nFailed to mmap model " << s_->model_name << std::endl;
    exit(-1);
  }
  s_->model = model_.get();
  LOG(INFO) << "Loaded model " << s_->model_name << std::endl;
  model_->error_reporter();
  LOG(INFO) << "resolved reporter" << std::endl;

  tflite::ops::builtin::BuiltinOpResolver resolver;

  tflite::InterpreterBuilder(*model_, resolver)(&interpreter_);
  if (!interpreter_) {
    LOG(FATAL) << "Failed to construct interpreter\n";
    exit(-1);
  }

  interpreter_->SetAllowFp16PrecisionForFp32(s_->allow_fp16);

  // Input dimentions
  input_tf_idx_ = interpreter_->inputs()[0];
  const auto & data = interpreter_->tensor(input_tf_idx_)->dims->data;
  input_tf_height_ = data[1];
  input_tf_width_ = data[2];
  input_tf_channel_ = data[3];

  if (s_->verbose) {
    LOG(INFO) << "tensors size: " << interpreter_->tensors_size() << "\n";
    LOG(INFO) << "nodes size: " << interpreter_->nodes_size() << "\n";
    LOG(INFO) << "inputs: " << interpreter_->inputs().size() << "\n";
    LOG(INFO) << "input(0) name: " << interpreter_->GetInputName(0) << "\n";
    LOG(INFO) << "input(0) idx: " << interpreter_->inputs()[0]<< "\n";
    LOG(INFO) << "input dims: " << input_tf_height_<<"x"<<input_tf_width_<<"x"<<input_tf_channel_<<std::endl;

    if (s_->verbose >= 2){
      int t_size = interpreter_->tensors_size();
      for (int i = 0; i < t_size; i++) {
        if (interpreter_->tensor(i)->name)
          LOG(INFO) << i << ": " << interpreter_->tensor(i)->name << ", "
                    << interpreter_->tensor(i)->bytes << ", "
                    << interpreter_->tensor(i)->type << ", "
                    << interpreter_->tensor(i)->params.scale << ", "
                    << interpreter_->tensor(i)->params.zero_point << "\n";
      }
    }
  }

  if (s_->number_of_threads != -1) {
    interpreter_->SetNumThreads(s_->number_of_threads);
  }

  if (ReadLabelsFile(s_->labels_file_name, &labels_, &label_count_) != kTfLiteOk){
    LOG(FATAL) << "Failed to construct read label file: "<< s_->labels_file_name<< std::endl;
    exit(-1);
  }
}

TfLiteStatus SsdClassifier::ReadLabelsFile(const string& file_name,
                            std::vector<string>* result,
                            size_t* found_label_count) {
  std::ifstream file(file_name);
  if (!file) {
    LOG(FATAL) << "Labels file " << file_name << " not found\n";
    return kTfLiteError;
  }
  result->clear();
  string line;
  while (std::getline(file, line)) {
    result->push_back(line);
  }
  *found_label_count = result->size();
  const int padding = 16;
  while (result->size() % padding) {
    result->emplace_back();
  }
  return kTfLiteOk;
}

void SsdClassifier::RunInference() {

  int image_width = 224;
  int image_height = 224;
  int image_channels = 3;
  std::vector<uint8_t> in = read_bmp(s_->input_bmp_name, &image_width,
                                     &image_height, &image_channels, s_);

  const std::vector<int> inputs = interpreter_->inputs();
  const std::vector<int> outputs = interpreter_->outputs();

  int input = interpreter_->inputs()[0];
  if (s_->verbose) LOG(INFO) << "input: " << input << "\n";

  if (s_->verbose) {
    LOG(INFO) << "number of inputs: " << inputs.size() << "\n";
    LOG(INFO) << "number of outputs: " << outputs.size() << "\n";
  }


  if (interpreter_->AllocateTensors() != kTfLiteOk) {
    LOG(FATAL) << "Failed to allocate tensors!";
  }



  s_->input_type = interpreter_->tensor(input_tf_idx_)->type;
  LOG(FATAL) << "Input type " << s_->input_type<<std::endl ;

  switch (s_->input_type) {
    case kTfLiteFloat32:
      resize<float>(interpreter_->typed_tensor<float>(input_tf_idx_), in.data(),
                    image_height, image_width, image_channels, input_tf_height_,
                    input_tf_width_, input_tf_channel_, s_);
      break;
    case kTfLiteInt8:
      resize<int8_t>(interpreter_->typed_tensor<int8_t>(input_tf_idx_), in.data(),
                     image_height, image_width, image_channels, input_tf_height_,
                     input_tf_width_, input_tf_channel_, s_);
      break;
    case kTfLiteUInt8:
      resize<uint8_t>(interpreter_->typed_tensor<uint8_t>(input_tf_idx_), in.data(),
                      image_height, image_width, image_channels, input_tf_height_,
                      input_tf_width_, input_tf_channel_, s_);
      break;
    default:
      LOG(FATAL) << "cannot handle input type "
                 << interpreter_->tensor(input_tf_idx_)->type << " yet";
      exit(-1);
  }

  if (s_->loop_count > 1)
    for (int i = 0; i < s_->number_of_warmup_runs; i++) {
      if (interpreter_->Invoke() != kTfLiteOk) {
        LOG(FATAL) << "Failed to invoke tflite!\n";
      }
    }

  struct timeval start_time, stop_time;
  gettimeofday(&start_time, nullptr);
  for (int i = 0; i < s_->loop_count; i++) {
    if (interpreter_->Invoke() != kTfLiteOk) {
      LOG(FATAL) << "Failed to invoke tflite!\n";
    }
  }
  gettimeofday(&stop_time, nullptr);
  LOG(INFO) << "invoked \n";
  LOG(INFO) << "average time: "
            << (get_us(stop_time) - get_us(start_time)) / (s_->loop_count * 1000)
            << " ms \n";

  const float threshold = 0.001f;

  std::vector<std::pair<float, int>> top_results;

  int output = interpreter_->outputs()[0];
  TfLiteIntArray* output_dims = interpreter_->tensor(output)->dims;
  // assume output dims to be something like (1, 1, ... ,size)
  auto output_size = output_dims->data[output_dims->size - 1];
  switch (interpreter_->tensor(output)->type) {
    case kTfLiteFloat32:
      get_top_n<float>(interpreter_->typed_output_tensor<float>(0), output_size,
                       s_->number_of_results, threshold, &top_results,
                       s_->input_type);
      break;
    case kTfLiteInt8:
      get_top_n<int8_t>(interpreter_->typed_output_tensor<int8_t>(0),
                        output_size, s_->number_of_results, threshold,
                        &top_results, s_->input_type);
      break;
    case kTfLiteUInt8:
      get_top_n<uint8_t>(interpreter_->typed_output_tensor<uint8_t>(0),
                         output_size, s_->number_of_results, threshold,
                         &top_results, s_->input_type);
      break;
    default:
      LOG(FATAL) << "cannot handle output type "
                 << interpreter_->tensor(output)->type << " yet";
      exit(-1);
  }

  for (const auto& result : top_results) {
    const float confidence = result.first;
    const int index = result.second;
    LOG(INFO) << confidence << ": " << index << " " << labels_[index] << "\n";
  }
}

void display_usage() {
  LOG(INFO)
      << "ssd_classifier\n"
      << "--allow_fp16, -f: [0|1], allow running fp32 models with fp16 or not\n"
      << "--count, -c: loop interpreter->Invoke() for certain times\n"
      << "--input_mean, -b: input mean\n"
      << "--input_std, -s: input standard deviation\n"
      << "--image, -i: image_name.bmp\n"
      << "--labels, -l: labels for the model\n"
      << "--tflite_model, -m: model_name.tflite\n"
      << "--num_results, -r: number of results to show\n"
      << "--threads, -t: number of threads\n"
      << "--verbose, -v: [0|1] print more information\n"
      << "--warmup_runs, -w: number of warmup runs\n"
      << "\n";
}

int Main(int argc, char** argv) {
  Settings s;

  int c;
  while (true) {
    static struct option long_options[] = {
        {"allow_fp16", required_argument, nullptr, 'f'},
        {"count", required_argument, nullptr, 'c'},
        {"verbose", required_argument, nullptr, 'v'},
        {"image", required_argument, nullptr, 'i'},
        {"labels", required_argument, nullptr, 'l'},
        {"tflite_model", required_argument, nullptr, 'm'},
        {"threads", required_argument, nullptr, 't'},
        {"input_mean", required_argument, nullptr, 'b'},
        {"input_std", required_argument, nullptr, 's'},
        {"num_results", required_argument, nullptr, 'r'},
        {"warmup_runs", required_argument, nullptr, 'w'},
        {nullptr, 0, nullptr, 0}};

    /* getopt_long stores the option index here. */
    int option_index = 0;

    c = getopt_long(argc, argv,
                    "a:b:c:f:g:i:j:l:m:r:s:t:v:w:x:", long_options,
                    &option_index);

    /* Detect the end of the options. */
    if (c == -1) break;

    switch (c) {
      case 'a':
        s.accel = strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'b':
        s.input_mean = strtod(optarg, nullptr);
        break;
      case 'c':
        s.loop_count =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'f':
        s.allow_fp16 =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'i':
        s.input_bmp_name = optarg;
        break;
      case 'l':
        s.labels_file_name = optarg;
        break;
      case 'm':
        s.model_name = optarg;
        break;
      case 'r':
        s.number_of_results =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 's':
        s.input_std = strtod(optarg, nullptr);
        break;
      case 't':
        s.number_of_threads = strtol(  // NOLINT(runtime/deprecated_fn)
            optarg, nullptr, 10);
        break;
      case 'v':
        s.verbose =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'w':
        s.number_of_warmup_runs =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'h':
      case '?':
        /* getopt_long already printed an error message. */
        display_usage();
        exit(-1);
      default:
        exit(-1);
    }
  }
  SsdClassifier classifier(&s);
  classifier.RunInference();
  return 0;
}

}  // namespace ssd_classifier
}  // namespace tflite

int main(int argc, char** argv) {
  return tflite::ssd_classifier::Main(argc, argv);
}
