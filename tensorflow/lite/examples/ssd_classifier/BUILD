# Description:
# TensorFlow Lite Example Label Image.

load("//tensorflow/lite:build_def.bzl", "tflite_linkopts")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(glob([
    "testdata/*.bmp",
]))

cc_binary(
    name = "ssd_classifier",
    srcs = [
        "get_top_n.h",
        "get_top_n_impl.h",
        "ssd_classifier.cc",
    ],
    linkopts = tflite_linkopts(),
    deps = [
        ":bitmap_helpers",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/tools/evaluation:utils",
    ],
)

cc_library(
    name = "bitmap_helpers",
    srcs = ["bitmap_helpers.cc"],
    hdrs = [
        "bitmap_helpers.h",
        "bitmap_helpers_impl.h",
        "ssd_classifier.h",
    ],
    deps = [
        "//tensorflow/lite:string",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite/kernels:builtin_ops",
    ],
)

