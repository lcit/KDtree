/*  ==========================================================================================
    Author: Leonardo Citraro
    Company:
    Filename: metrics.hpp
    Last modifed:   06.01.2017 by Leonardo Citraro
    Description:    Collection of distance metrics

    ==========================================================================================
    Copyright (c) 2016 Leonardo Citraro <ldo.citraro@gmail.com>

    Permission is hereby granted, free of charge, to any person obtaining a copy of this
    software and associated documentation files (the "Software"), to deal in the Software
    without restriction, including without limitation the rights to use, copy, modify,
    merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
    permit persons to whom the Software is furnished to do so, subject to the following
    conditions:

    The above copyright notice and this permission notice shall be included in all copies
    or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
    INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
    PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
    FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
    OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.
    ==========================================================================================
*/
#ifndef __METRICS_HPP__
#define __METRICS_HPP__

#include <algorithm>
#include <memory>
#include <vector>

namespace Distance {
    
    template<typename T>
    struct euclidean {
        inline T operator()(const T* a, const T* b, const int len) {
            T distance = T();
            for(int i=0; i<len; ++i) {
                T temp = a[i]-b[i];
                distance += temp*temp;
            }
            return std::sqrt(distance);
        }
        T operator()(const T a, const T b) {
            return std::abs(a-b);
        }
    };
    
    template<typename T>
    struct manhattan {
        inline T operator()(const T* a, const T* b, const int len) {
            T distance = T();
            for(int i=0; i<len; ++i) {
                distance += a[i]-b[i];
            }
            return std::sqrt(distance);
        }
        T operator()(const T a, const T b) {
            return std::abs(a-b);
        }
    };
    
}

#endif
