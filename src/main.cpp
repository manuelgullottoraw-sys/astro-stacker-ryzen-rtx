#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <numeric>
#include <chrono>
#include <omp.h>
#include <opencv2/opencv.hpp>

// Tailored Hardware Constants for Ryzen 5 5600 & 32GB RAM
constexpr int TARGET_THREADS = 12;      // 6 Cores / 12 Threads (Ryzen 5 5600)
constexpr size_t RAM_BUDGET_GB = 24;    // Safely use up to 24GB out of 32GB RAM

// Master Frame Generator (Parallelized for Zen 3)
cv::Mat createMasterFrame(const std::vector<std::string>& filePaths) {
    if (filePaths.empty()) return cv::Mat();

    int numImages = static_cast<int>(filePaths.size());
    cv::Mat firstImg = cv::imread(filePaths[0], cv::IMREAD_COLOR);
    int rows = firstImg.rows;
    int cols = firstImg.cols;

    cv::Mat master = cv::Mat::zeros(rows, cols, CV_32FC3);

    #pragma omp parallel for num_threads(TARGET_THREADS) reduction(+:master)
    for (int i = 0; i < numImages; ++i) {
        cv::Mat img = cv::imread(filePaths[i], cv::IMREAD_COLOR);
        cv::Mat imgFloat;
        img.convertTo(imgFloat, CV_32FC3);
        
        #pragma omp critical
        {
            master += imgFloat;
        }
    }

    master /= static_cast<float>(numImages);
    return master;
}

// Hardware-Matched Stacking Pipeline
cv::Mat stackImagesSigmaClip(
    const std::vector<std::string>& lightPaths,
    const std::vector<std::string>& darkPaths,
    const std::vector<std::string>& flatPaths,
    const std::vector<std::string>& biasPaths,
    float kappa = 2.0f
) {
    if (lightPaths.empty()) return cv::Mat();

    // Force OpenMP thread count to match 12 Logical Processors
    omp_set_num_threads(TARGET_THREADS);

    int numLights = static_cast<int>(lightPaths.size());
    cv::Mat firstImg = cv::imread(lightPaths[0], cv::IMREAD_COLOR);
    int rows = firstImg.rows;
    int cols = firstImg.cols;
    int channels = firstImg.channels();

    std::cout << "[+] Hardware Profile Applied:" << std::endl;
    std::cout << "    - CPU: AMD Ryzen 5 5600 (" << TARGET_THREADS << " Threads active)" << std::endl;
    std::cout << "    - Target RAM Allocation: Up to " << RAM_BUDGET_GB << " GB" << std::endl;

    // 1. Process Calibration Frames
    cv::Mat masterBias = cv::Mat::zeros(rows, cols, CV_32FC3);
    cv::Mat masterDark = cv::Mat::zeros(rows, cols, CV_32FC3);
    cv::Mat masterFlat = cv::Mat::ones(rows, cols, CV_32FC3) * 255.0f;

    if (!biasPaths.empty()) {
        std::cout << "[+] Building Master Bias..." << std::endl;
        masterBias = createMasterFrame(biasPaths);
    }

    if (!darkPaths.empty()) {
        std::cout << "[+] Building Master Dark..." << std::endl;
        masterDark = createMasterFrame(darkPaths);
        if (!biasPaths.empty()) {
            masterDark -= masterBias;
            cv::max(masterDark, 0.0f, masterDark);
        }
    }

    if (!flatPaths.empty()) {
        std::cout << "[+] Building Master Flat..." << std::endl;
        masterFlat = createMasterFrame(flatPaths);
        if (!biasPaths.empty()) {
            masterFlat -= masterBias;
            cv::max(masterFlat, 0.001f, masterFlat);
        }
        cv::Scalar meanFlat = cv::mean(masterFlat);
        float avgVal = static_cast<float>((meanFlat[0] + meanFlat[1] + meanFlat[2]) / 3.0);
        masterFlat /= avgVal;
    }

    // 2. Load & Calibrate Light Frames in RAM
    std::cout << "[+] Pre-loading " << numLights << " Light Frames into 32GB RAM..." << std::endl;
    std::vector<cv::Mat> images(numLights);

    #pragma omp parallel for num_threads(TARGET_THREADS)
    for (int i = 0; i < numLights; ++i) {
        cv::Mat img = cv::imread(lightPaths[i], cv::IMREAD_COLOR);
        cv::Mat imgFloat;
        img.convertTo(imgFloat, CV_32FC3);

        // Standard Astrophotography Calibration
        cv::Mat calibrated = (imgFloat - masterDark - masterBias) / masterFlat;
        cv::max(calibrated, 0.0f, calibrated);

        images[i] = calibrated;
    }

    // 3. Sigma Clipping Stacking (Dynamic Load-Balancing across 12 Threads)
    std::cout << "[+] Multi-core Stacking in progress..." << std::endl;
    cv::Mat result = cv::Mat::zeros(rows, cols, CV_32FC3);

    #pragma omp parallel for schedule(dynamic, 16) num_threads(TARGET_THREADS)
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            for (int ch = 0; ch < channels; ++ch) {
                
                std::vector<float> pixelValues(numLights);
                for (int i = 0; i < numLights; ++i) {
                    pixelValues[i] = images[i].at<cv::Vec3f>(r, c)[ch];
                }

                float sum = std::accumulate(pixelValues.begin(), pixelValues.end(), 0.0f);
                float mean = sum / numLights;

                float sq_sum = 0.0f;
                for (float val : pixelValues) {
                    sq_sum += (val - mean) * (val - mean);
                }
                float stdDev = std::sqrt(sq_sum / numLights);

                float filteredSum = 0.0f;
                int count = 0;
                for (float val : pixelValues) {
                    if (std::abs(val - mean) <= kappa * stdDev) {
                        filteredSum += val;
                        count++;
                    }
                }

                result.at<cv::Vec3f>(r, c)[ch] = (count > 0) ? (filteredSum / count) : mean;
            }
        }
    }

    cv::Mat finalOutput;
    result.convertTo(finalOutput, CV_8UC3);
    return finalOutput;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: astro_stacker.exe <output.png> -light img1.jpg img2.jpg [-dark d1.jpg] [-flat f1.jpg] [-bias b1.jpg]" << std::endl;
        return -1;
    }

    std::string outputPath = argv[1];
    std::vector<std::string> lightPaths, darkPaths, flatPaths, biasPaths;

    std::string currentType = "";
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-light") { currentType = "light"; continue; }
        if (arg == "-dark")  { currentType = "dark"; continue; }
        if (arg == "-flat")  { currentType = "flat"; continue; }
        if (arg == "-bias")  { currentType = "bias"; continue; }

        if (currentType == "light") lightPaths.push_back(arg);
        else if (currentType == "dark") darkPaths.push_back(arg);
        else if (currentType == "flat") flatPaths.push_back(arg);
        else if (currentType == "bias") biasPaths.push_back(arg);
        else lightPaths.push_back(arg);
    }

    if (lightPaths.empty()) {
        std::cerr << "[ERROR] No light frames provided!" << std::endl;
        return -1;
    }

    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat stacked = stackImagesSigmaClip(lightPaths, darkPaths, flatPaths, biasPaths);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    if (!stacked.empty()) {
        cv::imwrite(outputPath, stacked);
        std::cout << "[SUCCESS] Stacking completed in " << elapsed.count() << " seconds!" << std::endl;
        std::cout << "[+] Output written to: " << outputPath << std::endl;
    } else {
        std::cerr << "[ERROR] Stacking failed." << std::endl;
    }

    return 0;
}
