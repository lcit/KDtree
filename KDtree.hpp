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

//~ #define DEBUG 1

#include "metrics.hpp"
#include <iostream>
#include <numeric>
#include <algorithm>
#include <memory>
#include <vector>
#include <utility>
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
using Indexes = std::vector<int>;

template<typename Vector>
Indexes sort_indexes(const Vector& v) {
    Indexes idxs(v.size());
    std::iota(std::begin(idxs), std::end(idxs), 0);
    std::sort(std::begin(idxs), std::end(idxs), [&v](int i1, int i2) {return v[i1] < v[i2];});
    return idxs;
}

template<typename Vector>
Vector extract(const Vector& src, const int start, const int end) {
    Vector dest(end-start);
    for (int i = 0; i < dest.size(); ++i)
        dest[i] = src[i+start];
    return dest;
}

template<typename Vector1, typename Vector2>
Vector1 extract(const Vector1& src, const Vector2& idxs) {
    Vector1 dest(idxs.size());
    for (int i = 0; i < dest.size(); ++i)
        dest[i] = src[idxs[i]];
    return dest;
}

template<typename T, size_t ROWS, size_t COLS>
class KDnode {
private:
    const std::array<std::array<T,COLS>,ROWS>* _original_data;  ///< a pointer to the original data
    Indexes                 _indexes;                           ///< the indexes of the rows that belong to the current node
    KDnode<T,ROWS,COLS>*    _parent;                            ///< pointer to the parent node
    const int               _n_samples_split;                   ///< the minimum number of samples required to split a node
    std::vector<T>          _data_sliced_vector;                ///< vector used to construct the Eigen::Map (view)
    MatrixView<T>           _data_sliced = MatrixView<T>(nullptr,0,0); ///< a view (no copy) of a portion of the original data
    T                       _split_point;                       ///< the splitting threshold on the axe given by _split_axe
    std::vector<T>          _split_points;                      ///< collection of the previous _split_point
    const int               _split_axe;                         ///< index of the column used to decide where to split
    std::vector<int>        _split_axes;                        ///< collection of the previous _split_axes
    std::unique_ptr<KDnode<T,ROWS,COLS>> _left, _right;         ///< pointer to the next nodes
    const bool              _root_node;                         ///< true if node == root node
    bool                    _terminal_node;                     ///< true if node == terminal node
    const int               _depth;                             ///< node tree depth, the root node has a depth of 0

public:
    KDnode(const std::array<std::array<T,COLS>,ROWS>* data, const Indexes& indexes, KDnode<T,ROWS,COLS>* parent, const int split_axe,
    const bool root_node, const int n_samples_split, const int depth, const std::vector<T>& split_points = {}, const std::vector<int>& split_axes = {})
        : _original_data(data), _indexes(indexes), _parent(parent), _split_axe(split_axe), _n_samples_split(n_samples_split),
        _root_node(root_node), _terminal_node(false), _split_points(split_points), _split_axes(split_axes), _depth(depth){

        _split_axes.push_back(_split_axe);

        build_view();

        // split node until we have less than the desired number of samples per split
        if(indexes.size() > _n_samples_split) {
            split_node();
        } else {
            _terminal_node = true;
            _split_points = {};
        }
    }
    ~KDnode() {
        _parent = nullptr;
    }

    auto get_indexes() const                    { return _indexes; }
    auto get_split_axe() const                  { return _split_axe; }
    auto get_parent_split_axe() const           { return _parent? _parent->get_split_axe() : -1; }
    auto get_data_sliced() const                { return _data_sliced; }
    auto get_data_sliced_vector() const         { return _data_sliced_vector; }
    auto get_split_point() const                { return _split_point; }
    auto get_parent_split_point() const         { return _parent? _parent->get_split_point() : T();}
    auto get_branch_split_points() const        { return _split_points; }
    auto get_parent_branch_split_points() const { return _parent? _parent->get_branch_split_points() : std::vector<T>{};}
    auto get_branch_split_axes() const          { return _split_axes; }
    auto get_depth() const                      { return _depth; }
    auto go_left() const                        { return _left.get(); }
    auto go_right() const                       { return _right.get(); }
    auto go_back() const                        { return _parent; }
    auto is_root() const                        { return _root_node; }
    auto is_end() const                         { return _terminal_node; }


private:
    /// build the view of the original data
    void build_view() {
        int rows = _indexes.size();

        // concatenate all the necessary rows without making copies
        _data_sliced_vector.reserve(rows*COLS);
        for(int i=0; i<rows; ++i)
            std::move((*_original_data)[_indexes[i]].begin(), (*_original_data)[_indexes[i]].end(), std::back_inserter(_data_sliced_vector));

        // build the view using Eigen::Map:
        // Eigen::Map require to be initialised at compile-time, therefore, we are obliged to init the Map using
        // a nullptr. During run-time, we make use of the C++ "placement new" syntax to chnage the Map as explained here:
        // https://eigen.tuxfamily.org/dox/group__TutorialMapClass.html#TutorialMapPlacementNew
        new (&_data_sliced) MatrixView<T>(&_data_sliced_vector[0], rows, COLS);
    }

    /// create a new left and new right nodes
    void split_node() {
        // sort the column used to decide where to split
        VectorRow<T> feature_to_split = _data_sliced.col(_split_axe);
        auto sorted_indices = sort_indexes(feature_to_split);
        _indexes = extract(_indexes, sorted_indices);

        // get the new indexes of the left and right nodes
        int indexes_size = _indexes.size();
        int median_pos = indexes_size/2;
        int new_split_axe = (_split_axe+1)%(*_original_data)[0].size();
        Indexes indices_left = extract(_indexes, 0, median_pos);
        Indexes indices_right = extract(_indexes, median_pos, indexes_size);

        // calculate the split point
        if(indexes_size%2 == 0) {
            T p1 = _data_sliced(sorted_indices[median_pos-1],_split_axe);
            T p2 = _data_sliced(sorted_indices[median_pos],_split_axe);
            _split_point = (p1 + p2)/2.0;
        } else {
            _split_point = _data_sliced(sorted_indices[median_pos],_split_axe);
        }
        //~ if(indexes_size%2 == 0) {
            //~ _split_point = _data_sliced(sorted_indices[median_pos-1],_split_axe);
        //~ } else {
            //~ _split_point = _data_sliced(sorted_indices[median_pos],_split_axe);
        //~ }
        _split_points.push_back(_split_point);

        // make a new left and a new right nodes
        _left = std::make_unique<KDnode<T,ROWS,COLS>>(_original_data, indices_left, this, new_split_axe,
                                                        false, _n_samples_split, _depth+1, _split_points, _split_axes);
        _right = std::make_unique<KDnode<T,ROWS,COLS>>(_original_data, indices_right, this, new_split_axe,
                                                        false, _n_samples_split, _depth+1, _split_points, _split_axes);
    }
};

template<typename T, size_t ROWS, size_t COLS>
class KDtree {
private:
    const std::array<std::array<T,COLS>,ROWS>* _original_data;  ///< a pointer to the original data
    std::unique_ptr<KDnode<T,ROWS,COLS>> node0;                 ///< pointer to the root node
    const int _n_samples_split;                                 ///< the minimum number of samples required to split a node

public:
    KDtree(const std::array<std::array<T,COLS>,ROWS>* original_data, const int n_samples_split = 1)
        : _original_data(original_data), _n_samples_split(n_samples_split) {
        Indexes indexes0(_original_data->size());
        std::iota(std::begin(indexes0), std::end(indexes0), 0);
        node0 = std::make_unique<KDnode<T,ROWS,COLS>>(_original_data, indexes0, nullptr, 0, true, _n_samples_split, 0);
    }
    ~KDtree() {}

    auto get_original_data() const     { return _original_data; }
    auto get_n_samples_split() const   { return _n_samples_split; }
    auto get_node0() const             { return node0.get(); }

    auto go_last_node(const std::array<T,COLS>& sample) const {
        auto node = get_node0();
        while(!node->is_end()) {
            auto split_axe = node->get_split_axe();
            if(sample[split_axe] > node->get_split_point())
                node = node->go_right();
            else
                node = node->go_left();
        }
        return node;
    }

    template<template<typename> class Dist>
    std::vector<int> find_k_nearest(const int k, std::array<T,COLS>& sample, const int approx = 0) const {

        Dist<T> distance_metric;

        // go to the last node that contains the sample
        auto node = go_last_node(sample);

        // go backward until we reach a node with enough samples to start the searching algorithm
        auto node_indexes = node->get_indexes();
        while(node_indexes.size() < k) {
            node = node->go_back();
            if(!node) {
                std::cerr << "Error: The number of nearest neighbours k is greater than the dataset size!\n";
                exit(-1);
            }
            node_indexes = node->get_indexes();
        }
        auto branch_split_points = node->get_parent_branch_split_points();
        auto branch_split_axes = node->get_branch_split_axes();
#if DEBUG
        std::cout << "depth=" << node->get_depth() << "\n";
#endif
        // compute the distance between the sample and all the borders up to the root node
        std::vector<T> border_distances(branch_split_points.size());
        for(int i=0; i<border_distances.size(); ++i) {
            border_distances[i] = distance_metric(sample[branch_split_axes[i]], branch_split_points[i]);
        }
#if DEBUG
        std::cout << "border_distances.size()=" << border_distances.size() << "\n";
#endif
        // find the k nearest neighbours
        std::pair<Indexes,std::vector<T>>  k_nearest = retrieve_k_nearest(k, sample, node_indexes, distance_metric);

        // compare the farthest of the k nearest point with the border distances. If one of the borders
        // is closer to the test point than the farthest of the k nearest point it means that we have to go backward
        // in the tree. The node in which the border was created defines the number of backward steps we have to do
        // to reach the appropriate node.
        int go_back_n_times = 0;
        T farthest_samples_distance = k_nearest.second[k-1];
        for(int i=0; i<border_distances.size(); ++i) {
#if DEBUG
            std::cout << "farthest_samples_distance=" << farthest_samples_distance << " border_distances=" << border_distances[i] << "\n";
#endif
            if(farthest_samples_distance>border_distances[i] ) {
                go_back_n_times = border_distances.size()-i;
                break;
            }
        }
#if DEBUG
        std::cout << "go_back_n_times=" << go_back_n_times << "\n";
#endif
        // reduce the searching space to reduce execution time.
        // WARNING: If approx > 0 the KDtree might not find the exact nearest sample
        go_back_n_times -= approx;

        // skip this part if we stay in the same node
        if(go_back_n_times != 0) {
            // go back in the tree and re-find the k nearest neighbours
            for(int i=0; i<go_back_n_times; ++i)
                node = node->go_back();

            node_indexes = node->get_indexes();
#if DEBUG
            std::cout << "depth=" << node->get_depth() << "\n";
            std::cout << "node_indexes.size()=" << node_indexes.size() << "\n";
#endif
            k_nearest = retrieve_k_nearest(k, sample, node_indexes, distance_metric);

        }
        return k_nearest.first;
    }

public:

    template<template<typename> class Dist>
    inline std::pair<Indexes,std::vector<T>> retrieve_k_nearest(const int k, const std::array<T,COLS>& sample,
                                                            const Indexes& indexes, Dist<T>& distance_metric) const {

        //~ // special simpler case when indexes.size() == 1
        if(indexes.size() == 1) {
            std::vector<T> k_nearest_distances = {distance_metric(sample, (*_original_data)[indexes[0]])};
            Indexes k_nearest_absolute_idx = {indexes[0]};
            return std::make_pair<Indexes,std::vector<T>>(std::move(k_nearest_absolute_idx), std::move(k_nearest_distances));
        } else {
            std::vector<T> distances(indexes.size());
            std::vector<T> k_nearest_distances(k);
            Indexes k_nearest_relative_idx(k);
            Indexes k_nearest_absolute_idx(k);

            T d0 = distance_metric(sample, (*_original_data)[indexes[0]]);
            distances[0] = d0;
            k_nearest_distances[0] = d0;
            k_nearest_absolute_idx[0] = indexes[0];
            k_nearest_relative_idx[0] = 0;

            for(int i=1; i<indexes.size(); ++i) {
                T d = distance_metric(sample, (*_original_data)[indexes[i]]);
                distances[i] = d;
                if(d < d0) {
                    k_nearest_distances[0] = d;
                    k_nearest_absolute_idx[0] = indexes[i];
                    k_nearest_relative_idx[0] = i;
                    d0 = d;
#if DEBUG
                    std::cout << "d=" << d << "\n";
#endif
                }
            }
#if DEBUG
            std::cout << "k_nearest_distances[0]=" << k_nearest_distances[0] << "\n";
#endif
            if(k>1) {
                distances[k_nearest_relative_idx[0]] = -1;

                for(int kk=1; kk<k; ++kk) {
                    int not_used_yet;
                    for(int i=0; i<indexes.size(); ++i) {
                        if(distances[i] != -1) {
                            not_used_yet = i;
                            break;
                        }
                    }
                    k_nearest_distances[kk] = distances[not_used_yet];
                    k_nearest_absolute_idx[kk] = indexes[not_used_yet];
                    k_nearest_relative_idx[kk] = not_used_yet;

                    for(int i=1; i<indexes.size(); ++i) {
                        if(distances[i] < k_nearest_distances[kk] && distances[i] != -1) {
                            k_nearest_distances[kk] = distances[i];
                            k_nearest_absolute_idx[kk] = indexes[i];
                            k_nearest_relative_idx[kk] = i;
                        }
                    }
                    distances[k_nearest_relative_idx[kk]] = -1;
                }
            }
            return std::make_pair<Indexes,std::vector<T>>(std::move(k_nearest_absolute_idx), std::move(k_nearest_distances));
        }
    }
};
#endif
