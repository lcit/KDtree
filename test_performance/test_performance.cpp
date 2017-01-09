/*  =========================================================================
    Author: Leonardo Citraro
    Company:
    Filename: test_performance.cpp
    Last modifed:   09.01.2017 by Leonardo Citraro
    Description:    Test of performance

    =========================================================================

    =========================================================================
*/
#include "KDtree.hpp"
#include "opencv2/ml/ml.hpp"
#include <iostream>
#include <algorithm>
#include <memory>
#include <functional>
#include <chrono>
#include <random>
#include <cstdlib>
#include <iomanip>

template<typename TimeT = std::chrono::milliseconds>
struct measure {
    template<typename F, typename ...Args>
    static typename TimeT::rep run(F&& func, Args&&... args) {
        auto start = std::chrono::steady_clock::now();
        std::forward<decltype(func)>(func)(std::forward<Args>(args)...);
        auto duration = std::chrono::duration_cast<TimeT>(std::chrono::steady_clock::now() - start);
        return duration.count();
    }
};

template<size_t N>
struct mean_stddev {
    template<typename F, typename ...Args>
    static auto run(F&& func, Args&&... args){
        std::array<double, N> buffer;
        for(auto& buf : buffer)
            buf = std::forward<decltype(func)>(func)(std::forward<Args>(args)...);
        auto sum = std::accumulate(std::begin(buffer), std::end(buffer), 0.0);
        auto mean = sum/buffer.size();
        std::array<double, N> diff;
        std::transform(std::begin(buffer), std::end(buffer), std::begin(diff), [mean](auto x) { return x - mean; });
        auto sq_sum = std::inner_product(std::begin(diff), std::end(diff), std::begin(diff), 0.0);
        auto stddev = std::sqrt(sq_sum/buffer.size());
        return std::make_pair(mean,stddev);
    }
};

using TYPE = float;

int main(int argc, char* argv[]) {
    
    // ----------------------------------------------------------------------------
    // Data creation
    // ----------------------------------------------------------------------------
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> gauss1(-10,10);
    
    const int ROWS = 10000;
    const int COLS = 200;
    // data can be huge thus we use the static keyword to prevent stack overflows
    static std::array<std::array<TYPE,COLS>,ROWS> data;
    for(int i=0; i<ROWS; ++i){
        std::array<TYPE,COLS> row;
        for(int j=0; j<COLS; ++j)
            row[j] = static_cast<TYPE>(gauss1(gen));
        data[i] = std::move(row);
    }
    
    std::cout << COLS << "\n";
    
    // test samples
    const int TEST_SAMPLES = 100;
    static std::array<std::array<TYPE,COLS>,TEST_SAMPLES> test_samples;
    for(int i=0; i<TEST_SAMPLES; ++i) {
        std::array<TYPE,COLS> row;
        for(int j=0; j<COLS; ++j)
            row[j] = static_cast<TYPE>(gauss1(gen));
        test_samples[i] = std::move(row);
    }
    
    // ----------------------------------------------------------------------------
    // Conversion to cv::Mat
    // ----------------------------------------------------------------------------
    cv::Mat mat_data(data.size(), data[0].size(), CV_32F);
    for(size_t i = 0; i < mat_data.rows; ++i) {
        for (size_t j = 0; j < mat_data.cols; ++j) {
            mat_data.at<TYPE>(i,j) = data[i][j];
        }
    }

    // ----------------------------------------------------------------------------
    // Testing functions
    // ----------------------------------------------------------------------------

    auto my_brute_force = [&](const int k){
        for(int n=0; n<TEST_SAMPLES; ++n) {

            Distance::euclidean<TYPE> dist;
            std::vector<TYPE> distances(data.size());
            std::vector<TYPE> k_nearest_distances(k);
            std::vector<int> k_nearest_idx(k);
            for(int i=0; i<data.size(); ++i) {
                distances[i] = dist(test_samples[n], data[i]);
                if(i>0) {
                    if(distances[i] < k_nearest_distances[0]) {
                        k_nearest_distances[0] = distances[i];
                        k_nearest_idx[0] = i;
                    }
                } else {
                    k_nearest_distances[0] = distances[0];
                    k_nearest_idx[0] = 0;
                }
            }
            distances[k_nearest_idx[0]] = -1;
            
            for(int kk=1; kk<k; ++kk) {
                int not_used_yet;
                for(int i=0; i<data.size(); ++i) {
                    if(distances[i] != -1) {
                        not_used_yet = i;
                        break;
                    }
                }
                k_nearest_distances[kk] = distances[not_used_yet];
                k_nearest_idx[kk] = not_used_yet;
                
                for(int i=1; i<data.size(); ++i) {
                    if(distances[i] < k_nearest_distances[kk] && distances[i] != -1) {
                        k_nearest_distances[kk] = distances[i];
                        k_nearest_idx[kk] = i;
                    }
                }
                distances[k_nearest_idx[kk]] = -1;
            }
        }
    };

    auto my_kdtree = [&](const int k,KDtree<TYPE,ROWS,COLS>& kdtree){
        for(int n=0; n<TEST_SAMPLES; ++n) {
            volatile auto k_nearest = kdtree.find_k_nearest<Distance::euclidean>(k, test_samples[n]);
        }
    };
    
    auto OpenCV_knn_kdtree = [&](const int k, cv::Ptr<cv::ml::KNearest> opencv_knn_kdtree){
        for(int n=0; n<TEST_SAMPLES; ++n) {
            //~ int prediction = opencv_knn_kdtree->predict(cv::Mat(1,COLS,CV_32F,test_samples[n].data()));
            cv::Mat res;
            float prediction = opencv_knn_kdtree->findNearest(cv::Mat(1,COLS,CV_32F,test_samples[n].data()),k,res);
        }
    };
    
    auto OpenCV_brute_force = [&](const int k, cv::Ptr<cv::ml::KNearest> opencv_knn_brute){
        for(int n=0; n<TEST_SAMPLES; ++n) {
            //~ int prediction = opencv_knn_brute->predict(cv::Mat(1,COLS,CV_32F,test_samples[n].data()));
            cv::Mat res;
            float prediction = opencv_knn_brute->findNearest(cv::Mat(1,COLS,CV_32F,test_samples[n].data()),k,res);
        }
    };
    

    
    // ----------------------------------------------------------------------------
    // Get execution times
    // ----------------------------------------------------------------------------
    
    std::cout << "------------------------------------------------------------------------------------------\n";
    std::cout << "Dimensionality = " << COLS << "\n";
    std::cout << "                      my BruteForce        my KDtree          OpenCV BruteForce    OpenCV KDtree\n";
    
    
    //~ std::vector<int> K = {1};
    std::vector<int> K = {1, 2, 5, 10, 50};
    const int times = 5;
    for(auto k:K) {
        std::cout << "Time elapsed (k=" << std::setw(2) << k << ")   ";
        
        auto res = mean_stddev<times>::run([&](){return measure<>::run(my_brute_force, k);});
        std::cout << std::setw(4) <<  res.first << "(+-" << std::setprecision(3) << std::setw(6) << res.second << ")     ";
        
        // the construction of the tree can be done beforehand in the training process.
        KDtree<TYPE,ROWS,COLS> kdtree(&data, 1);
        
        res = mean_stddev<times>::run([&](){return measure<>::run(my_kdtree, k, kdtree);});
        std::cout << std::setw(4) <<  res.first << "(+-" << std::setprecision(3) << std::setw(6) << res.second << ")      ";
        
        // the construction of the knn-bruteforce can be done beforehand in the training process.
        cv::Ptr<cv::ml::KNearest> opencv_knn_brute = cv::ml::KNearest::create();
        opencv_knn_brute->setAlgorithmType(cv::ml::KNearest::BRUTE_FORCE);
        opencv_knn_brute->setIsClassifier(true);
        cv::Mat labels(ROWS,1,CV_32F);
        for(int i=0; i<ROWS; ++i)
            labels.at<float>(i) = i;
        opencv_knn_brute->train(mat_data, cv::ml::SampleTypes::ROW_SAMPLE, labels);
        
        res = mean_stddev<times>::run([&](){return measure<>::run(OpenCV_brute_force, k, opencv_knn_brute);});
        std::cout << std::setw(4) <<  res.first << "(+-" << std::setprecision(3) << std::setw(6) << res.second << ")      ";
        
        // the construction of the knn-tree can be done beforehand in the training process.
        cv::Ptr<cv::ml::KNearest> opencv_knn_kdtree = cv::ml::KNearest::create();
        opencv_knn_kdtree->setAlgorithmType(cv::ml::KNearest::KDTREE);
        opencv_knn_kdtree->setIsClassifier(true);
        opencv_knn_kdtree->train(mat_data, cv::ml::SampleTypes::ROW_SAMPLE, labels);
        
        res = mean_stddev<times>::run([&](){return measure<>::run(OpenCV_knn_kdtree, k, opencv_knn_kdtree);});
        std::cout << std::setw(4) <<  res.first << "(+-" << std::setprecision(3) << std::setw(6) << res.second << ")\n";
        
    }
    
    return 0;
}

/*
 * 
 * Tested on an Intel quad-core hyperthreading i7-4700MQ 2.4GHz 64 bits architecture
 * The results are in milliseconds (mean & std-dev). The results correspond to the time 
 * required to classify 100 test samples with a training dataset of 10000 samples
 * 
------------------------------------------------------------------------------------------
Dimensionality = 2
                      my BruteForce        my KDtree          OpenCV BruteForce    OpenCV KDtree
Time elapsed (k= 1)      8(+-     0)        1(+-     0)      34.2(+-   0.4)       4.8(+-   9.6)
Time elapsed (k= 2)   11.4(+-  0.49)        2(+-     0)      34.8(+- 0.748)       0.4(+-  0.49)
Time elapsed (k= 5)     19(+-     0)        4(+-     0)        33(+-  3.03)       0.2(+-   0.4)
Time elapsed (k=10)   25.6(+-  0.49)      8.4(+-  0.49)      35.4(+-  0.49)         1(+-     0)
Time elapsed (k=50)    144(+-  3.16)       52(+-   5.4)      40.4(+-  4.45)         3(+-     0)
* 
------------------------------------------------------------------------------------------
Dimensionality = 3
                      my BruteForce       my KDtree         OpenCV BruteForce    OpenCV KDtree
Time elapsed (k= 1)   14.4(+-  0.49)      2.6(+-  0.49)        31(+-  1.55)         4(+-  7.51)
Time elapsed (k= 2)   13.4(+-  0.49)      3.2(+-   0.4)        30(+-     0)         0(+-     0)
Time elapsed (k= 5)   19.8(+-   0.4)      5.4(+-  0.49)      23.2(+-   0.4)         0(+-     0)
Time elapsed (k=10)   38.4(+-  0.49)       16(+-     0)      39.8(+-   0.4)         2(+-     0)
Time elapsed (k=50)    155(+-  4.26)     93.6(+-   1.2)      47.8(+- 0.748)         5(+-     0)
*
------------------------------------------------------------------------------------------
Dimensionality = 10
                      my BruteForce         my KDtree      OpenCV BruteForce    OpenCV KDtree
Time elapsed (k= 1)     34(+- 0.632)       47(+-     0)      52.6(+-  5.31)      18.2(+-  7.91)
Time elapsed (k= 2)   31.2(+-  3.12)     51.2(+- 0.748)        63(+- 0.632)      26.8(+- 0.748)
Time elapsed (k= 5)   45.2(+-   0.4)     60.2(+- 0.748)      62.4(+-  0.49)      42.6(+-  0.49)
Time elapsed (k=10)   57.8(+- 0.748)       62(+-  5.87)      64.6(+-  3.72)      66.6(+-  4.32)
Time elapsed (k=50)    170(+-  2.19)      194(+-  1.79)      71.8(+-   0.4)       141(+-  5.11)
*
------------------------------------------------------------------------------------------
Dimensionality = 100
                      my BruteForce        my KDtree          OpenCV BruteForce    OpenCV KDtree
Time elapsed (k= 1)  359.8(+-  2.79)      484(+-  26.1)       284(+-  15.9)       515(+-  10.1)
Time elapsed (k= 2)    351(+-  15.6)      479(+-  9.03)       281(+-  14.8)       504(+-  25.6)
Time elapsed (k= 5)    358(+-  15.1)      474(+-  25.3)       292(+-   1.2)       497(+-  3.72)
Time elapsed (k=10)    383(+-  2.42)      493(+-  6.77)       292(+-  1.33)       508(+-   8.2)
Time elapsed (k=50)    499(+-  3.76)      629(+-  5.16)       303(+-  2.19)       510(+-  4.87)
*
------------------------------------------------------------------------------------------
Dimensionality = 200
                      my BruteForce        my KDtree          OpenCV BruteForce    OpenCV KDtree
Time elapsed (k= 1)    716(+-  50.2)      930(+-  58.4)       557(+-  37.1)       695(+-  32.5)
Time elapsed (k= 2)    606(+-  87.8)      761(+-   142)       555(+-  39.7)       695(+-  52.3)
Time elapsed (k= 5)    722(+-  47.3)      781(+-   135)       555(+-  26.8)       733(+-  19.5)
Time elapsed (k=10)    675(+-  62.1)      960(+-  43.6)       545(+-  41.3)       717(+-    43)
Time elapsed (k=50)    861(+-  37.2)     1.04e+03(+-  55.3)       550(+-  69.3)       732(+-  35.7)
*/
