/*  =========================================================================
    Author: Leonardo Citraro
    Company:
    Filename: main.cpp
    Last modifed:   06.01.2017 by Leonardo Citraro
    Description:    KDtree example

    =========================================================================

    =========================================================================
*/
#include "metrics.hpp"
#include "KDtree.hpp"
#include <iostream>
#include <vector>

int main(int argc, char* argv[]) {

    using TYPE = float;
    
    std::vector<std::array<TYPE,2>> data = {{1.1, 0.6},{0.4, 0.5},{0.2, 0.6},{0.5, 0.9},
                                            {1.2, 0.3},{0.7, 0.4},{0.8, 1.0},{0.1, 0.2}};
    KDtree<TYPE,2> kdtree(&data);
    
    auto node = kdtree.get_node0();
    std::cout << "Is root node? " << std::boolalpha << node->is_root() << "\n";
    std::cout << "Point(1)=\n" << node->get_split_point() << "\n";
    node = node->go_left();
    std::cout << "Point(2a)=\n" << node->get_split_point() << "\n";
    node = node->go_left();
    std::cout << "Point(3a)=\n" << node->get_split_point() << "\n";
    node = node->go_back();
    node = node->go_right();
    std::cout << "Point(3b)=\n" << node->get_split_point() << "\n";
    
    // these data is an Eigen::Map (view) of the original data
    auto node_data = node->get_data_sliced();
    
    std::cout << "The point nearest to (0.15,0.48) is: \n";
    std::array<TYPE,2> sample = {0.15,0.48};
    auto nearest_samples_idx = kdtree.find_k_nearest<Distance::euclidean>(1, sample);
    for(auto& ns:nearest_samples_idx){
        for(auto& v:data[ns])
            std::cout << v << ",";
        std::cout << "\n";
    }
    
    return 0;
}
