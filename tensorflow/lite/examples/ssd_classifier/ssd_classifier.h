#ifndef TENSORFLOW_LITE_EXAMPLES_SSD_CLASSIFIER_SSD_CLASSIFIER_H_
#define TENSORFLOW_LITE_EXAMPLES_SSD_CLASSIFIER_SSD_CLASSIFIER_H_

#include <vector>
#include <unordered_set>
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

// Detected objects of the same label
struct LabeledObjects{
  void insert(float box[4], float score){
    // x1, y1, x2, y2 -> y1, x1, y2, x2
    bboxes.push_back({box[1], box[0], box[3], box[2]});
    // bboxes.push_back({box[0], box[1], box[2], box[3]});

    scores.emplace_back(score);
  }
  std::vector<std::array<float, 4>> bboxes;
  std::vector<float> scores;
};

std::ostream & operator<< (std::ostream &out, const LabeledObjects & in){
  for (auto i=0; i<in.bboxes.size(); ++i){
    out<<"Box[ "<<in.bboxes[i][0]<<" "<<in.bboxes[i][1]<<" "<<
                  in.bboxes[i][2]<<" "<<in.bboxes[i][3]<<" ] score "<<
                  in.scores[i]<<std::endl;
  }
  return out;
}






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
  void Resize();
  int GetNumObjects();
  std::unordered_set<int> GetDetectedClasses();
  void CategoryObjectsByLabel();
  void NmsBoxes(LabeledObjects *objs,
    int max_output_size, float iou_threshld, float score_threshold); // Non Maximal Supression

  TfLiteFlatBufferModelPtr model_;
  TfLiteInterpreterPtr interpreter_;
  Settings *s_;
  int input_tf_idx_;
  int input_tf_height_;
  int input_tf_width_;
  int input_tf_channel_;

  std::vector<string> labels_;
  size_t label_count_;
  std::unordered_map<int, LabeledObjects> labeled_objects_;
};




}  // namespace ssd_classifier
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXAMPLES_SSD_CLASSIFIER_SSD_CLASSIFIER_H_
