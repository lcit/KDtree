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
    

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> gauss1(0.5,1);

    for(int n1=0; n1<1; ++n1) {
        
        // ----------------------------------------------------------------------------
        // Data creation
        // ----------------------------------------------------------------------------
        const int N = 1000;
        std::array<std::array<TYPE,2>,N> data;
        for(int y=0; y<N; ++y){
            std::array<TYPE,2> row = {static_cast<TYPE>(gauss1(gen)), static_cast<TYPE>(gauss1(gen))};
            data[y] = row;
        }
        //~ const int N = 8;
        //~ std::array<std::array<TYPE,2>,N> data = {{{{1.1, 0.6}},{{0.4, 0.5}},{{0.2, 0.6}},{{0.5, 0.9}},{
                                            //~ {1.2, 0.3}},{{0.7, 0.4}},{{0.8, 1.0}},{{0.1, 0.2}}}};
        
        // ----------------------------------------------------------------------------
        // Model creation
        // ----------------------------------------------------------------------------
        KDtree<TYPE,N,2> kdtree(&data);
        
        for(int n2=0; n2<1; ++n2) {
            
            // test sample
            std::array<TYPE,2> sample = {static_cast<TYPE>(gauss1(gen)), static_cast<TYPE>(gauss1(gen))};
            
            // ----------------------------------------------------------------------------
            // Get the nearest points using the KDtree
            // ----------------------------------------------------------------------------
            const int k = 1;
            std::vector<int> nearest_samples = kdtree.find_k_nearest<Distance::euclidean>(k, sample);
            
            // ----------------------------------------------------------------------------
            // Get the nearest points using brute force
            // ----------------------------------------------------------------------------
            std::vector<TYPE> distances(data.size());
            Distance::euclidean<TYPE> dist;
            for(int i=0; i<data.size(); ++i) {
                distances[i] = dist(sample.data(), data[i].data(), 2);
            }
            std::vector<int> indexes_brute_force = sort_indexes(distances);
            
            // ----------------------------------------------------------------------------
            // Checking if the results are identical
            // ----------------------------------------------------------------------------
            for(int i=0; i<k; ++i) {
                for(int j=0; j<data[0].size(); ++j){
                    //~ std::cout << "kdtree:" << nearest_samples[i] << " brute: " << indexes_brute_force[i] << "\n";
                    if( std::abs(data[nearest_samples[i]][j]-data[indexes_brute_force[i]][j]) > 0.0001) {
                        std::cerr << "Error: result of KDtree is defferent form the result of BruteForce!\n";
                        std::cout << "Sample point:";
                        for(auto& v:sample)
                            std::cout << v << ",";
                        std::cout << "\n";
                        std::cout << "KDtree point:";
                        for(auto& v:data[nearest_samples[i]])
                            std::cout << v << ",";
                        std::cout << "\n";
                        std::cout << "BruteForce point:";
                        for(auto& v:data[indexes_brute_force[i]])
                            std::cout << v << ",";
                        std::cout << "\n";
                        exit(-1);
                    }
                }
            }
        }
    }
    
    std::cout << "\nTest passed!\n\n";
    
    return 0;

}
