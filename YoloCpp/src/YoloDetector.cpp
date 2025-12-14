#include "YoloDetector.h"
#include <iostream>

// Constructor
YoloDetector::YoloDetector(const std::string& model_path, bool use_gpu)
    : env(ORT_LOGGING_LEVEL_WARNING, "YoloDetector")
    , session(nullptr)
{
    // Session settings
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    input_tensor_values.resize(input_width * input_height * 3);
    input_shape = { 1, 3, input_height, input_width };

    if (use_gpu) {
        try {
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = 0;
            session_options.AppendExecutionProvider_CUDA(cuda_options);
            std::cout << "CUDA provider added successfully." << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "Failed to add CUDA provider: " << e.what() << ". Using CPU." << std::endl;
        }
    }

#ifdef _WIN32
    std::wstring w_model_path(model_path.begin(), model_path.end());
    session = Ort::Session(env, w_model_path.c_str(), session_options);
#else
    session = Ort::Session(env, model_path.c_str(), session_options);
#endif
    Ort::AllocatorWithDefaultOptions allocator;

    // Input
    auto input_name = session.GetInputNameAllocated(0, allocator);
    input_names_ptr.push_back(strdup(input_name.get()));

    // Output
    auto output_name = session.GetOutputNameAllocated(0, allocator);
    output_names_ptr.push_back(strdup(output_name.get()));
}

std::vector<Detection> YoloDetector::detect(const cv::Mat& image) {
    preprocess(image);

    // Create ONNX tensor
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_tensor_values.data(),
        input_tensor_values.size(),
        input_shape.data(),
        input_shape.size()
    );

    std::vector<const char*> input_names_char = { input_names_ptr[0] };
    std::vector<const char*> output_names_char = { output_names_ptr[0] };

    auto output_tensors = session.Run(
        Ort::RunOptions{ nullptr },
        input_names_char.data(),
        &input_tensor,
        1,
        output_names_char.data(),
        1
    );

    float* floatarr = output_tensors[0].GetTensorMutableData<float>();

    // Will use later
    auto output_info = output_tensors[0].GetTensorTypeAndShapeInfo();
    auto shape = output_info.GetShape();

    return postprocess(floatarr, 0.5f, image.size());
}

void YoloDetector::preprocess(const cv::Mat& image) {
    cv::Mat blob;
    cv::dnn::blobFromImage(image, blob, 1.0 / 255.0, cv::Size(input_width, input_height), cv::Scalar(), true, false);
    
    std::memcpy(input_tensor_values.data(), blob.ptr<float>(), input_tensor_values.size() * sizeof(float));
}

std::vector<Detection> YoloDetector::postprocess(float* data, float conf_threshold, const cv::Size& original_size) {
    std::vector<Detection> detections;

    float x_factor = (float)original_size.width / input_width;
    float y_factor = (float)original_size.height / input_height;

    int max_det = 300;
    int dimensions = 6; // x, y, w, h, conf, class

    for (int i = 0; i < max_det; ++i) {
        float* row = data + (i * dimensions);

        float confidence = row[4];

        if (confidence < conf_threshold) continue;

        float x = row[0];
        float y = row[1];
        float w = row[2];
        float h = row[3];
        int class_id = (int)row[5];

        // x1, y1, x2, y2:
        int left = int(x * x_factor);
        int top = int(y * y_factor);
        int width = int((w - x) * x_factor);
        int height = int((h - y) * y_factor);

        detections.push_back({ class_id, confidence, cv::Rect(left, top, width, height) });
    }

    return detections;
}