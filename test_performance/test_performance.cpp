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
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/flann/flann.hpp"
#include <iostream>
#include <algorithm>
#include <memory>
#include <functional>
#include <chrono>
#include <random>

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

int main(int argc, char* argv[]) {
    
    using TYPE = float;
    
    // ----------------------------------------------------------------------------
    // Data creation
    // ----------------------------------------------------------------------------
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> gauss1(15,0);
    
    const int N = 1000;
    std::vector<std::vector<TYPE>> data;
    for(int y=0; y<N; ++y){
        std::vector<TYPE> row = {static_cast<TYPE>(gauss1(gen)), static_cast<TYPE>(gauss1(gen))};
        data.push_back(row);
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
    auto my_kdtree = [&](){
        KDtree<TYPE> kdtree(&data);
    };
    
    //auto OpenCV_kdtree = [&](){
    //    cv::flann::KDTreeIndex<cv::DistanceTypes::DIST_L2> kdtree(mat_data);
    //};
    
    // ----------------------------------------------------------------------------
    // Get execution times
    // ----------------------------------------------------------------------------
    auto res = mean_stddev<3>::run([&](){return measure<>::run(my_kdtree);});
    std::cout << "Time elapsed (my KDtree) " << res.first << "(+-" << res.second << ") [ms]\n";
    
    //res = mean_stddev<3>::run([&](){return measure<>::run(OpenCV_kdtree);});
    //std::cout << "Time elapsed (OpenCV KDtree) " << res.first << "(+-" << res.second << ") [ms]\n";

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
