/*  =========================================================================
    Author: Leonardo Citraro
    Company:
    Filename: test_functional.cpp
    Last modifed:   09.01.2017 by Leonardo Citraro
    Description:    Functional test

    =========================================================================

    =========================================================================
*/
#include "KDtree.hpp"
#include <iostream>
#include <algorithm>
#include <random>

//~ #define DEBUG 1
//~ #define LOWDIM 1

int main(int argc, char* argv[]) {
    
    using TYPE = float;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> gauss1(0,2);

    // perform the same tets on different random datasets
    for(int n1=0; n1<100; ++n1) {
        
        // ----------------------------------------------------------------------------
        // Data creation
        // ----------------------------------------------------------------------------
#if LOWDIM
        const int N = 8;
        const int C = 2;
        std::array<std::array<TYPE,C>,N> data = {{{{1.1, 0.6}},{{0.4, 0.5}},{{0.2, 0.6}},{{0.5, 0.9}},{
                                                {1.2, 0.3}},{{0.7, 0.4}},{{0.8, 1.0}},{{0.1, 0.2}}}};
#else
        const int N = 1000;
        const int C = 8;
        std::array<std::array<TYPE,C>,N> data;
        for(int y=0; y<N; ++y){
            std::array<TYPE,C> row = {  static_cast<TYPE>(gauss1(gen)), static_cast<TYPE>(gauss1(gen)),
                                        static_cast<TYPE>(gauss1(gen)), static_cast<TYPE>(gauss1(gen)),
                                        static_cast<TYPE>(gauss1(gen)), static_cast<TYPE>(gauss1(gen)),
                                        static_cast<TYPE>(gauss1(gen)), static_cast<TYPE>(gauss1(gen))};
            data[y] = row;
        }
#endif
        
        // ----------------------------------------------------------------------------
        // Model creation
        // ----------------------------------------------------------------------------
        KDtree<TYPE,N,C> kdtree(&data);
        
        // perform multiple searches on the same tree
        for(int n2=0; n2<100; ++n2) {
#if LOWDIM
            // test sample
            std::array<TYPE,C> sample = {  static_cast<TYPE>(gauss1(gen)), static_cast<TYPE>(gauss1(gen))};
#else
            std::array<TYPE,C> sample = {   static_cast<TYPE>(gauss1(gen)), static_cast<TYPE>(gauss1(gen)),
                                            static_cast<TYPE>(gauss1(gen)), static_cast<TYPE>(gauss1(gen)),
                                            static_cast<TYPE>(gauss1(gen)), static_cast<TYPE>(gauss1(gen)),
                                            static_cast<TYPE>(gauss1(gen)), static_cast<TYPE>(gauss1(gen))};
#endif
            // ----------------------------------------------------------------------------
            // Get the nearest points using the KDtree
            // ----------------------------------------------------------------------------
            const int k = 5;
            std::vector<int> nearest_samples = kdtree.find_k_nearest<Distance::euclidean>(k, sample, 0);
            
            // ----------------------------------------------------------------------------
            // Get the nearest points using brute force
            // ----------------------------------------------------------------------------
            // compute the distances
            Distance::euclidean<TYPE> dist;
            std::vector<TYPE> distances(data.size());
            for(int i=0; i<data.size(); ++i)
                distances[i] = dist(sample, data[i]);
            
            // sort the distances
            std::vector<int> indexes_brute_force(data.size());
            std::iota(std::begin(indexes_brute_force), std::end(indexes_brute_force), 0);
            std::sort(std::begin(indexes_brute_force), std::end(indexes_brute_force), 
                        [&distances](int i1, int i2) {return distances[i1] < distances[i2];});
#if DEBUG
            std::cout << "k_nearest_distances_brute[0]=" << distances[indexes_brute_force[0]] << "\n";
#endif
            // ----------------------------------------------------------------------------
            // Checking if the results are identical
            // ----------------------------------------------------------------------------
            for(int i=0; i<k; ++i) {
                TYPE e = dist(data[nearest_samples[i]], data[indexes_brute_force[i]]);
                if(e > 0.0000001) {
                    std::cerr << "Error: KDtree result is defferent form the BruteForce! " << e << " \n";
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
    
    std::cout << "\nTest passed!\n\n";
    
    return 0;
}
