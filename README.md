# KDtree

C++ - KDtree implementation.

### Prerequisites

Eigen 3.3.90

### Example usage

![alt tag](https://raw.githubusercontent.com/lcit/KDtree/master/kdtree_example.JPG)

```C++
#include "KDtree.hpp"
#include <iostream>
#include <vector>

int main(int argc, char* argv[]) {

    using TYPE = float;
    
    std::vector<std::vector<TYPE>> data = {{1.1, 0.6},{0.4, 0.5},{0.2, 0.6},{0.5, 0.9},
                                            {1.2, 0.3},{0.7, 0.4},{0.8, 1.0},{0.1, 0.2}};
    KDtree<TYPE> kdtree(&data);
    
    auto node = kdtree.get_node0();
    std::cout << "Point(1)=\n" << node->get_point() << "\n";
    node = node->go_left();
    std::cout << "Point(2a)=\n" << node->get_point() << "\n";
    node = node->go_left();
    std::cout << "Point(3a)=\n" << node->get_point() << "\n";
    node = node->go_back();
    node = node->go_right();
    std::cout << "Point(3b)=\n" << node->get_point() << "\n";
    
    std::cout << "The point nearest to (0.15,0.48) is: \n";
    std::vector<TYPE> sample = {0.15,0.48};
    std::vector<std::vector<TYPE>> nearest_samples = kdtree.find_k_nearest<Distance::euclidean>(1, sample);
    for(auto& ns:nearest_samples){
        for(auto& v:ns)
            std::cout << v << ",";
         std::cout << "\n";
    }
    
    return 0;
}
```
The output:
```
Point(1)=
0.5
0.9
Point(2a)=
0.4
0.5
Point(3a)=
0.1
0.2
Point(3b)=
0.2
0.6
The nearest point to (0.15,0.48) is: 
0.2,0.6,
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

