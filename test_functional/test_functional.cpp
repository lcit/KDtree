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
#include <iostream>
#include <algorithm>
#include <random>

int main(int argc, char* argv[]) {
    
    using TYPE = float;
    
    // ----------------------------------------------------------------------------
    // Data creation
    // ----------------------------------------------------------------------------
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> gauss1(0.5,1);
    
    //~ const int N = 1000;
    //~ std::vector<std::vector<TYPE>> data;
    //~ for(int y=0; y<N; ++y){
        //~ std::vector<TYPE> row = {static_cast<TYPE>(gauss1(gen)), static_cast<TYPE>(gauss1(gen))};
        //~ data.push_back(row);
    //~ }
    
    std::vector<std::vector<TYPE>> data = {{1.1, 0.6},{0.4, 0.5},{0.2, 0.6},{0.5, 0.9},
                                            {1.2, 0.3},{0.7, 0.4},{0.8, 1.0},{0.1, 0.2}};
                                            
    std::vector<TYPE> sample = {static_cast<TYPE>(gauss1(gen)), static_cast<TYPE>(gauss1(gen))};
    for(auto& v:sample)
            std::cout << v << ",";
        std::cout << "\n";
    
    const int k = 1;
    
    // ----------------------------------------------------------------------------
    // Get the nearest points using the KDtree
    // ----------------------------------------------------------------------------
    KDtree<TYPE> kdtree(&data);
    std::vector<std::vector<TYPE>> nearest_samples = kdtree.find_k_nearest<Distance::euclidean>(k, sample);
    
    // ----------------------------------------------------------------------------
    // Get the nearest points using brute force
    // ----------------------------------------------------------------------------
    std::vector<TYPE> distances(data.size());
    Distance::euclidean<TYPE> dist;
    for(int i=0; i<data.size(); ++i) {
        distances[i] = dist(sample, data[i]);
    }
    std::vector<int> indexes_brute_force = sort_indexes(distances);
    
    // ----------------------------------------------------------------------------
    // Comparison
    // ----------------------------------------------------------------------------
    for(int i=0; i<k; ++i) {
        for(auto& v:nearest_samples[i])
            std::cout << v << ",";
        std::cout << "\n";
        for(auto& v:data[indexes_brute_force[i]])
            std::cout << v << ",";
        std::cout << "\n----------\n";
    }

    return 0;

}
