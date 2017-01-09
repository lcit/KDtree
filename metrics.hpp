/*  ==========================================================================================
    Author: Leonardo Citraro
    Company:
    Filename: metrics.hpp
    Last modifed:   09.01.2017 by Leonardo Citraro
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
    
    // WARNING: the result provided by these function is the square of the true euclidean distance
    template<typename T>
    struct euclidean {
        template<size_t N>
        inline T operator()(const std::array<T,N>& a, const std::array<T,N>& b) {
            T distance = T();
            for(int i=0; i<N; ++i) {
                // calling .data() here seem improving the performance
                T temp = a.data()[i]-b.data()[i];
                distance += temp*temp;
            }
            //return std::sqrt(distance);
            return distance;
        }
        inline T operator()(const T* a, const T* b, const int len) {
            T distance = T();
            for(int i=0; i<len; ++i) {
                T temp = a[i]-b[i];
                distance += temp*temp;
            }
            //~ return std::sqrt(distance);
            return distance;
        }
        inline T operator()(const T a, const T b) {
            T temp = a-b;
            return temp*temp;
            //~ return std::abs(a-b);
        }
    };
    
    template<typename T>
    struct manhattan {
        template<size_t N>
        inline T operator()(const std::array<T,N>& a, const std::array<T,N>& b) {
            T distance = T();
            for(int i=0; i<N; ++i) {
                // calling .data() here seem improving the performance
                distance += std::abs(a.data()[i]-b.data()[i]);
            }
            return distance;
        }
        inline T operator()(const T* a, const T* b, const int len) {
            T distance = T();
            for(int i=0; i<len; ++i) {
                distance += std::abs(a[i]-b[i]);
            }
            return distance;
        }
        T operator()(const T a, const T b) {
            return std::abs(a-b);
        }
    };
    
}

#endif
