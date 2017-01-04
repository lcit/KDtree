/*  ==========================================================================================
    Author: Leonardo Citraro
    Company:
    Filename: KDtree.hpp
    Last modifed:   03.01.2017 by Leonardo Citraro
    Description:    KDtree

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
#ifndef __KDTREE_HPP__
#define __KDTREE_HPP__

#include <iostream>
#include <algorithm>
#include <memory>
#include <vector>
#include <Eigen/Dense>

template<typename T>
using Matrix = Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
template<typename T>
using MatrixView = Eigen::Map<Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>>;
template<typename T>
using VectorRow = Eigen::Matrix<T,Eigen::Dynamic,1>;
template<typename T>
using VectorRowView = Eigen::Map<Eigen::Matrix<T,Eigen::Dynamic,1>>;
template<typename T>
using VectorCol = Eigen::Matrix<T,1,Eigen::Dynamic>;
template<typename T>
using VectorColView = Eigen::Map<Eigen::Matrix<T,1,Eigen::Dynamic>>;
using Indexes = Eigen::Matrix<int,Eigen::Dynamic,1>;

//~ template<typename Vector>
//~ Eigen::VectorXi sort_indexes(const Eigen::Ref<const Vector>& v) {
    //~ Eigen::VectorXi idxs = Eigen::VectorXi::LinSpaced(v.size(), 0, v.size()-1);
    //~ std::sort(idxs.data(), idxs.data() + idxs.size(), 
                //~ [&v](int i1, int i2) {return v[i1] < v[i2];});
    //~ return idxs;
//~ }

template<typename Vector>
std::vector<int> sort_indexes(const Vector& v) {
    std::vector<int> idxs(v.size());
    std::iota(std::begin(idxs), std::end(idxs), 0);
    std::sort(std::begin(idxs), std::end(idxs), 
                [&v](int i1, int i2) {return v[i1] < v[i2];});
    return idxs;
}

template<typename Vector>
Vector extract(const Vector& src, const int start, const int end) {
    Vector dest(end-start);
    for (int i = 0; i < dest.size(); i++) {
        dest[i] = src[i+start];
    }
    return dest;
}

template<typename Vector1, typename Vector2>
Vector1 extract(const Vector1& src, const Vector2& idxs) {
    Vector1 dest(idxs.size());
    for (int i = 0; i < dest.size(); i++) {
        dest[i] = src[idxs[i]];
    }
    return dest;
}

namespace Distance {
    //~ template<typename T>
    //~ class euclidean {
        //~ T operator()(const T* a, const T* b, const int len) {
            //~ T distance = 0;
            //~ for(int i=0; i<len; ++i) {
                //~ T temp = a[i]-b[i];
                //~ distance += temp*temp;
            //~ }
            //~ return std::sqrt(distance);
        //~ }
    //~ }
    template<typename T>
    struct euclidean {
        T operator()(const std::vector<T>& a, const std::vector<T>& b) {
            std::vector<T> c(a.size());
            std::transform(std::begin(a), std::end(a), std::begin(b), std::begin(c), std::minus<T>());
            std::transform(std::begin(c), std::end(c), std::begin(c), std::begin(c), std::multiplies<T>());
            return std::sqrt(std::accumulate(std::begin(c), std::end(c), 0.0));
        }
    };
    
    //~ template<typename Derived>
    //~ auto manhattan(const Eigen::MatrixBase<Derived>& a, const Eigen::MatrixBase<Derived>& b) {
        //~ return (a-b).sum();
    //~ }
}

template<typename T>
class KDnode {
private:
    const std::vector<std::vector<T>>* _original_data;
    Indexes _indexes;
    KDnode<T>* _parent;
    const int _split_axe;
    const int _n_samples_per_split;
    std::vector<T> _data_sliced_vector;
    MatrixView<T> _data_sliced = MatrixView<T>(nullptr,0,0);
    VectorRow<T> _point;
    std::unique_ptr<KDnode<T>> _left, _right;
    static bool _root_node;
    bool _terminal_node;
    
public:
    KDnode(const std::vector<std::vector<T>>* data, Indexes& indexes, KDnode<T>* parent, const int split_axe, const int n_samples_per_split) 
    : _original_data(data), _indexes(indexes), _parent(parent), _split_axe(split_axe), _n_samples_per_split(n_samples_per_split), _terminal_node(false) {
        
        //std::cout << "Split_axe=" << _split_axe << " Indexes=" << _indexes.transpose() << "\n";
        
        build_view();
        
        // split node until we have less than the desired number of samples per split
        if(indexes.size() > _n_samples_per_split)
            split_node();
        else
            _terminal_node = true;
        if(_root_node)
            _root_node = false;
    }
    ~KDnode() {
        _parent = nullptr;
    }
    
    auto get_indexes() const { return _indexes; }
    auto get_split_axe() const { return _split_axe; }
    auto get_parent_split_axe() const { return _parent->get_split_axe(); }
    auto get_data_sliced() const { return _data_sliced; }
    auto get_data_sliced_vector() const { return _data_sliced_vector; }
    auto get_point() const { return _point; }
    auto get_parent_point() const { return _parent->get_point(); }
    auto go_left() { return _left.get(); }
    auto go_right() { return _right.get(); }
    auto go_back() { return _parent; }
    auto is_root() { return _root_node; }
    auto is_end() { return _terminal_node; }
    
private:
    void build_view() {
        // build the view of the original data
        int rows = _indexes.size();
        int cols = (*_original_data)[0].size();
        
        //std::cout << "rows=" << rows << " cols=" << cols << "\n";
        
        _data_sliced_vector.reserve(rows*cols);
        for(int i=0; i<rows; ++i)
            std::move((*_original_data)[_indexes[i]].begin(), (*_original_data)[_indexes[i]].end(), std::back_inserter(_data_sliced_vector));
            
        // The Eigen::Map doesn't have a default constructor therefore we are obliged initialiized it with NULL.
        // Succesively we make use of the C++ "placement new" syntax to chnage the Map as explained here:
        // https://eigen.tuxfamily.org/dox/group__TutorialMapClass.html#TutorialMapPlacementNew
        new (&_data_sliced) MatrixView<T>(&_data_sliced_vector[0], rows, cols);
        
        //std::cout << "_data_sliced=...\n" << _data_sliced << "\n";
    }
    
    void split_node() {
        VectorRow<T> feature_to_split = _data_sliced.col(_split_axe);
        auto sorted_indices = sort_indexes(feature_to_split);
        _indexes = extract(_indexes, sorted_indices);
        int indexes_size = _indexes.size();
        int median_pos = indexes_size/2;
        int new_split_axe = (_split_axe+1)%(*_original_data)[0].size();
        Indexes indices_left = extract(_indexes, 0, median_pos);
        Indexes indices_right = extract(_indexes, median_pos, indexes_size);
        if(indexes_size%2 == 0)
            _point = _data_sliced.row(sorted_indices[median_pos-1]);
        else
            _point = _data_sliced.row(sorted_indices[median_pos]);
        _left = std::make_unique<KDnode<T>>(_original_data, indices_left, this, new_split_axe, _n_samples_per_split);
        _right = std::make_unique<KDnode<T>>(_original_data, indices_right, this, new_split_axe, _n_samples_per_split);
    }
};

template<typename T>
bool KDnode<T>::_root_node = true;

template<typename T>
class KDtree {
private:
    const std::vector<std::vector<T>>* _original_data;
    std::unique_ptr<KDnode<T>> node0;
    const int _n_samples_per_split;
    
public:
    KDtree(const std::vector<std::vector<T>>* original_data, const int n_samples_per_split = 1) 
        : _original_data(original_data), _n_samples_per_split(n_samples_per_split) {
        Indexes indexes0 = Eigen::VectorXi::LinSpaced(_original_data->size(), 0, _original_data->size()-1);
        node0 = std::make_unique<KDnode<T>>(_original_data, indexes0, nullptr, 0, _n_samples_per_split);
    }
    ~KDtree() {
    }
    
    auto get_original_data()  const { return _original_data; }
    auto get_n_samples_per_split()  const { return _n_samples_per_split; }
    auto get_node0() { return node0.get(); }
    auto go_last_node(const std::vector<T>& sample) {
        auto node = get_node0();
        while(!node->is_end()) {
            auto split_axe = node->get_split_axe();
            if(sample[split_axe] > node->get_point()[split_axe])
                node = node->go_right();
            else
                node = node->go_left();
        }
        return node;
    }
    
    template<template<typename> class Dist>
    //template<typename Dist>
    std::vector<std::vector<T>> find_k_nearest(const int k, std::vector<T>& sample) {
        
        Dist<T> distance_metric;
        
        //VectorRow<T> vsample(&sample[0], sample.size());
        auto node = go_last_node(sample);

        std::vector<T> distances;
        std::vector<int> idx_distances;
        bool is_border_the_closer = true;
        
        while(node) {
            //auto node_data = node->get_data_sliced();
            // indexes and the data are in the same order
            auto node_indexes = node->get_indexes(); 
            auto node_point = node->get_parent_point();
            auto node_split_axe = node->get_parent_split_axe();
            
            //~ std::cout << "node_split_axe=" << node_split_axe << "\n";
            //~ std::cout << "node_point=\n" << node_point << "\n";
            
            // compute the distance from the current node border
            std::vector<T> node_point_masked(node_point.size(), 0);
            std::vector<T> sample_masked(node_point.size(), 0);
            node_point_masked[node_split_axe] = node_point[node_split_axe];
            sample_masked[node_split_axe] = sample[node_split_axe];
            auto d_sample_border = distance_metric(sample_masked, node_point_masked);
            
            for(int i=0; i<node_indexes.size(); ++i) {
                // do not compute the distance twice
                if(std::find(std::begin(distances), std::end(distances), node_indexes[i]) == std::end(distances)) {
                    auto d = distance_metric(sample, (*_original_data)[node_indexes[i]]);
                    //~ std::cout << "d=" << d << ", d_border=" << d_sample_border << "\n";
                    if(d < d_sample_border)
                        is_border_the_closer = false;
                    distances.push_back(d);
                    idx_distances.push_back(node_indexes[i]);
                }
            }
            if(is_border_the_closer)
                node = node->go_back();
            else
                break;
        };
        
        auto sorted_indexes = sort_indexes(distances);
        std::vector<std::vector<T>> nearest_samples(k);
        for(int i=0; i<k; ++i) {
            nearest_samples[i] = (*_original_data)[idx_distances[sorted_indexes[i]]];
        }
        return nearest_samples;
    }
};
#endif
