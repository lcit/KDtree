/*  =========================================================================
    Author: Leonardo Citraro
    Company:
    Filename: test_performance.cpp
    Last modifed:   03.01.2017 by Leonardo Citraro
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

template<typename TimeT = std::chrono::milliseconds>
struct measure
{
    template<typename F, typename ...Args>
    static typename TimeT::rep run(F&& func, Args&&... args)
    {
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
    std::normal_distribution<> gauss1(0,15);
    
    const int N0 = 100000;
    const int COLS = 6;
    std::array<std::array<TYPE,COLS>,N0> data;
    for(int i=0; i<N0; ++i){
        std::array<TYPE,COLS> row = {static_cast<TYPE>(gauss1(gen)), static_cast<TYPE>(gauss1(gen)), 
                                static_cast<TYPE>(gauss1(gen)), static_cast<TYPE>(gauss1(gen)), 
                                static_cast<TYPE>(gauss1(gen)), static_cast<TYPE>(gauss1(gen))};
        //~ std::array<TYPE,COLS> row = {static_cast<TYPE>(gauss1(gen)), static_cast<TYPE>(gauss1(gen))};
        data[i] = row;
    }
    
    // test samples
    const int N1 = 100;
    std::array<std::array<TYPE,COLS>,N1> test_samples;
    for(int i=0; i<N1; ++i) {
        std::array<TYPE,COLS> sample = {static_cast<TYPE>(gauss1(gen)), static_cast<TYPE>(gauss1(gen)), 
                                    static_cast<TYPE>(gauss1(gen)), static_cast<TYPE>(gauss1(gen)), 
                                    static_cast<TYPE>(gauss1(gen)), static_cast<TYPE>(gauss1(gen))};
        //~ std::array<TYPE,COLS> sample = {static_cast<TYPE>(gauss1(gen)), static_cast<TYPE>(gauss1(gen))};
        test_samples[i] = sample;
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
        for(int n=0; n<N1; ++n) {

            Distance::euclidean<TYPE> dist;
            std::vector<TYPE> distances(data.size());
            std::vector<TYPE> k_nearest_distances(k);
            std::vector<int> k_nearest_idx(k);
            for(int i=0; i<data.size(); ++i) {
                distances[i] = dist(test_samples[n].data(), data[i].data(), COLS);
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
            
            //~ for(auto& kn:k_nearest_idx)
                //~ std::cout << kn << ",";
            //~ std::cout << "\n";
        }
    };

    auto my_kdtree = [&](const int k,KDtree<TYPE,N0,COLS>& kdtree){
        for(int n=0; n<N1; ++n) {
            volatile auto k_nearest = kdtree.find_k_nearest<Distance::euclidean>(k, test_samples[n], 3);
            //~ for(auto& kn:k_nearest)
                //~ std::cout << kn << ",";
            //~ std::cout << "\n";
        }
    };
    
    auto OpenCV_knn_kdtree = [&](const int k, cv::Ptr<cv::ml::KNearest> opencv_knn_kdtree){
        for(int n=0; n<N1; ++n) {
            //~ int prediction = opencv_knn_kdtree->predict(cv::Mat(1,COLS,CV_32F,test_samples[n].data()));
            cv::Mat res;
            float prediction = opencv_knn_kdtree->findNearest(cv::Mat(1,COLS,CV_32F,test_samples[n].data()),k,res);
            //~ std::cout << res.at<float>(0,0) << "\n";
        }
    };
    
    auto OpenCV_brute_force = [&](const int k, cv::Ptr<cv::ml::KNearest> opencv_knn_brute){
        for(int n=0; n<N1; ++n) {
            //~ int prediction = opencv_knn_brute->predict(cv::Mat(1,COLS,CV_32F,test_samples[n].data()));
            cv::Mat res;
            float prediction = opencv_knn_brute->findNearest(cv::Mat(1,COLS,CV_32F,test_samples[n].data()),k,res);
            //~ std::cout << prediction << "\n";
        }
    };
    

    
    // ----------------------------------------------------------------------------
    // Get execution times
    // ----------------------------------------------------------------------------
    
    //~ std::vector<int> K = {1};
    std::vector<int> K = {1, 2, 5, 10, 50};
    const int times = 1;
    for(auto k:K) {
        std::cout << "------------------------------------------------\n";
        std::cout << "Number of nearest neighbours to find: " << k << " \n";
        
        auto res = mean_stddev<times>::run([&](){return measure<>::run(my_brute_force, k);});
        std::cout << "Time elapsed (my BruteForce) " << res.first << "(+-" << res.second << ") [ms]\n";
        
        // the construction of the tree can be done beforehand in the training process.
        KDtree<TYPE,N0,COLS> kdtree(&data, 1);
        
        res = mean_stddev<times>::run([&](){return measure<>::run(my_kdtree, k, kdtree);});
        std::cout << "Time elapsed (my KDtree) " << res.first << "(+-" << res.second << ") [ms]\n";
        
        // the construction of the knn-bruteforce can be done beforehand in the training process.
        cv::Ptr<cv::ml::KNearest> opencv_knn_brute = cv::ml::KNearest::create();
        opencv_knn_brute->setAlgorithmType(cv::ml::KNearest::BRUTE_FORCE);
        opencv_knn_brute->setIsClassifier(true);
        cv::Mat labels(N0,1,CV_32F);
        for(int i=0; i<N0; ++i)
            labels.at<float>(i) = i;
        opencv_knn_brute->train(mat_data, cv::ml::SampleTypes::ROW_SAMPLE, labels);
        
        res = mean_stddev<times>::run([&](){return measure<>::run(OpenCV_brute_force, k, opencv_knn_brute);});
        std::cout << "Time elapsed (OpenCV BruteForce) " << res.first << "(+-" << res.second << ") [ms]\n";
        
        // the construction of the knn-tree can be done beforehand in the training process.
        cv::Ptr<cv::ml::KNearest> opencv_knn_kdtree = cv::ml::KNearest::create();
        opencv_knn_kdtree->setAlgorithmType(cv::ml::KNearest::KDTREE);
        opencv_knn_kdtree->setIsClassifier(true);
        cv::Mat labels2(N0,1,CV_32F);
        for(int i=0; i<N0; ++i)
            labels2.at<float>(i) = i;
        opencv_knn_kdtree->train(mat_data, cv::ml::SampleTypes::ROW_SAMPLE, labels2);
        
        res = mean_stddev<times>::run([&](){return measure<>::run(OpenCV_knn_kdtree, k, opencv_knn_kdtree);});
        std::cout << "Time elapsed (OpenCV KDtree) " << res.first << "(+-" << res.second << ") [ms]\n";
        
    }
    
    return 0;

}

// Tested on an Intel quad-core hyperthreading i7-4700MQ 2.4GHz 64 bits architecture

/*
 * Using HOG::none
 * 
    Time elapsed (n_threads=1): 407(+-18.5472) [ms]
    Time elapsed (n_threads=2): 277.667(+-1.88562) [ms]
    Time elapsed (n_threads=4): 164(+-5.09902) [ms]
    Time elapsed (n_threads=8): 141(+-6.37704) [ms]
*/

/*
 * Using HOG::L2hys
 * 
    Time elapsed (n_threads=1): 1088(+-4.08248) [ms]
    Time elapsed (n_threads=2): 635.667(+-2.62467) [ms]
    Time elapsed (n_threads=4): 382.333(+-18.8031) [ms]
    Time elapsed (n_threads=8): 347.333(+-3.39935) [ms]
*/
