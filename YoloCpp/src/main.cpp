#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "YoloDetector.h"
#include <chrono>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <vector>
#include <numeric>
#include <cmath>
#include <iomanip>
#include <ctime>

// Threaded camera class
class CameraStream {
public:
    CameraStream(int src = 0) : stopped(false) {
        cap.open(src);
        if (!cap.isOpened()) {
            std::cerr << "Error: Could not open webcam." << std::endl;
            stopped = true;
            return;
        }
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

        // Read the first frame synchronously to make sure it works
        cap >> frame;
    }

    void start() {
        if (stopped) return;
        worker = std::thread(&CameraStream::update, this);
    }

    void update() {
        cv::Mat temp;
        while (!stopped) {
            if (!cap.read(temp)) {
                stopped = true;
                break;
            }

            // Frame writing
            {
                std::lock_guard<std::mutex> lock(mtx);
                temp.copyTo(frame);
            }
        }
    }

    cv::Mat read() {
        cv::Mat result;
        {
            std::lock_guard<std::mutex> lock(mtx);
            if (!frame.empty()) {
                frame.copyTo(result);
            }
        }
        return result;
    }

    void stop() {
        stopped = true;
        if (worker.joinable()) {
            worker.join();
        }
        cap.release();
    }

    ~CameraStream() {
        stop();
    }

    bool isOpened() const {
        return cap.isOpened();
    }

private:
    cv::VideoCapture cap;
    cv::Mat frame;
    std::thread worker;
    std::mutex mtx;
    std::atomic<bool> stopped;
};

// Statistics helpers
double getMean(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    return sum / v.size();
}

double getStdDev(const std::vector<double>& v, double mean) {
    if (v.empty()) return 0.0;
    double sq_sum = 0.0;
    for (const auto& val : v) {
        sq_sum += (val - mean) * (val - mean);
    }
    return std::sqrt(sq_sum / v.size());
}

// Helper for timestamp formatting
std::string getCurrentTimeStr() {
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%H:%M:%S");
    return oss.str();
}


int main() {
    // Setting up logger
    std::ofstream logFile("web_fps_report_cpp.txt");
    auto log = [&](const std::string& msg) {
        std::cout << msg << std::endl;
        if (logFile.is_open()) {
            logFile << getCurrentTimeStr() << " - INFO - " << msg << std::endl;
        }
    };

    std::cout << "Initializing YoloDetector..." << std::endl;

    YoloDetector detector("yolov8n.onnx", true);

    CameraStream cam(0);
    if (!cam.isOpened()) return -1;

    cam.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(1000)); // Warmup

    std::cout << "System active. Press ESC to stop." << std::endl;

    std::vector<double> inference_times;
    std::vector<double> frame_times;

    double fps_display = 0.0;
    int frame_count = 0;

    // Warm up inference engine keep the statistics clear
    std::cout << "Warming up inference engine..." << std::endl;
    cv::Mat empty_frame = cv::Mat::zeros(480, 640, CV_8UC3);
    detector.detect(empty_frame);
    std::cout << "Warmup complete. Starting measurements..." << std::endl;

    auto prev_time = std::chrono::high_resolution_clock::now();

    while (true) {
        auto t_frame_start = std::chrono::high_resolution_clock::now();

        // Async read
        cv::Mat frame = cam.read();

        if (frame.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        // Inference
        auto t_infer_start = std::chrono::high_resolution_clock::now();
        std::vector<Detection> results = detector.detect(frame);
        auto t_infer_end = std::chrono::high_resolution_clock::now();

        double inf_time_ms = std::chrono::duration<double, std::milli>(t_infer_end - t_infer_start).count();
        inference_times.push_back(inf_time_ms);

        // Visualization
        for (const auto& det : results) {
            cv::rectangle(frame, det.box, cv::Scalar(0, 255, 0), 2);
        }

        frame_count++;
        auto curr_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = curr_time - prev_time;

        if (elapsed.count() >= 1.0) {
            fps_display = frame_count / elapsed.count();
            frame_count = 0;
            prev_time = curr_time;
        }

        std::string info = "FPS: " + std::to_string((int)fps_display) +
            " | Inf: " + std::to_string((int)inf_time_ms) + " ms";
        cv::putText(frame, info, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);

        cv::imshow("Yolo C++ Demo", frame);

        if (cv::waitKey(1) == 27) break;

        auto t_frame_end = std::chrono::high_resolution_clock::now();
        double frame_time_ms = std::chrono::duration<double, std::milli>(t_frame_end - t_frame_start).count();
        frame_times.push_back(frame_time_ms);
    }

    cam.stop();
    cv::destroyAllWindows();

    // Report generation
    if (!inference_times.empty()) {
        double total_time_sec = std::accumulate(frame_times.begin(), frame_times.end(), 0.0) / 1000.0;
        size_t total_frames = inference_times.size();

        double avg_inf = getMean(inference_times);
        double std_inf = getStdDev(inference_times, avg_inf);

        double avg_frame_time = getMean(frame_times);
        double std_frame_time = getStdDev(frame_times, avg_frame_time);

        double real_fps = (avg_frame_time > 0) ? (1000.0 / avg_frame_time) : 0.0;
        double potential_fps = (avg_inf > 0) ? (1000.0 / avg_inf) : 0.0;

        std::ostringstream report;
        report << "\nPERFORMANCE REPORT:\n"
            << "Total Frames Processed: " << total_frames << "\n"
            << "Total Duration:         " << std::fixed << std::setprecision(2) << total_time_sec << " sec\n"
            << "\n"
            << "INFERENCE STATS (CV model only):\n"
            << "  Avg Time: " << avg_inf << " ms\n"
            << "  Std Dev:  " << std_inf << " ms (Jitter)\n"
            << "  Max Potential FPS: " << potential_fps << "\n"
            << "\n"
            << "SYSTEM STATS (end-to-end):\n"
            << "  Avg Frame Time: " << avg_frame_time << " ms\n"
            << "  Std Dev:        " << std_frame_time << " ms\n"
            << "  Real Average FPS: " << real_fps << "\n";

        log(report.str());
    }
    else {
        log("No frames processed.");
    }

    logFile.close();
    return 0;
}
