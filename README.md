# KDtree

C++ - Simple yet effective KDtree implementation with (exact) find k nearest neighbours capability.

### Prerequisites

Eigen 3.3.90 and OpenCV 3.1 if you want to run test_performance

### Run example
```
./clean.sh; ./build.sh
./run.sh
```

### Example usage

![alt tag](https://raw.githubusercontent.com/lcit/KDtree/master/kdtree_example.JPG)

```C++
#include "KDtree.hpp"
#include <iostream>
#include <array>

int main(int argc, char* argv[]) {

    using TYPE = float;

    std::array<std::array<TYPE,2>,8> data = {{{{1.1, 0.6}},{{0.4, 0.5}},{{0.2, 0.6}},{{0.5, 0.9}},
                                            {{1.2, 0.3}},{{0.7, 0.4}},{{0.8, 1.0}},{{0.1, 0.2}}}};
    KDtree<TYPE,8,2> kdtree(&data);

    auto node = kdtree.get_node0();
    std::cout << "Is root node? " << std::boolalpha << node->is_root() << "\n";
    std::cout << "Split point(0)=\n" << node->get_split_point() << "\n";
    node = node->go_left();
    std::cout << "Split point(1a)=\n" << node->get_split_point() << "\n";
    node = node->go_left();
    std::cout << "Split point(2a)=\n" << node->get_split_point() << "\n";
    node = node->go_back();
    node = node->go_right();
    std::cout << "Split point(2b)=\n" << node->get_split_point() << "\n";

    // node_data is an Eigen::Map (view) of the original data
    auto node_data = node->get_data_sliced();

    std::cout << "The point nearest to (0.55,0.4) is: \n";
    std::array<TYPE,2> sample = {0.55,0.4};
    auto nearest_samples_idx = kdtree.find_k_nearest<Distance::euclidean>(1, sample);
    for(auto& ns:nearest_samples_idx){
        for(auto& v:data[ns])
            std::cout << v << ",";
        std::cout << "\n";
    }

    return 0;
}
```
The output:
```
Is root node? true
Split point(0)=
0.6
Split point(1a)=
0.55
Split point(2a)=
0.25
Split point(2b)=
0.35
The point nearest to (0.55,0.4) is:
0.7,0.4,
```

## Performance

For low dimensional data the KDtree produces faster results comapred to brute force algorithms whereas for high dimensional data (>10) brute force becomes a better solution.

```
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
```
## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments
