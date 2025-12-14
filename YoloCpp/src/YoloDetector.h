#pragma once
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>

// Structure for detection result
struct Detection {
    int class_id;
    float confidence;
    cv::Rect box; // (x, y, w, h)
};

class YoloDetector {
public:
    YoloDetector(const std::string& model_path, bool use_gpu = true);
    std::vector<Detection> detect(const cv::Mat& image);

private:
    // ONNX Runtime resources
    Ort::Env env;
    Ort::Session session;

    std::vector<char*> input_names_ptr;
    std::vector<char*> output_names_ptr;

    int input_width = 640;
    int input_height = 640;

    std::vector<float> input_tensor_values;
    std::vector<int64_t> input_shape;

    // Helper methods
    void preprocess(const cv::Mat& image);
    std::vector<Detection> postprocess(float* data,
        float conf_threshold,
        const cv::Size& original_size);
};