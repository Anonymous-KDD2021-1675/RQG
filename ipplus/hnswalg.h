#pragma once
#include "visited_list_pool.h"
#include "hnswlib.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <unordered_set> // don't use set, use unordered_set instead

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

template <typename T>
void writeBinaryPOD(std::ostream& out, const T& podRef) {
  out.write((char*)&podRef, sizeof(T));
}

template <typename T>
static void readBinaryPOD(std::istream& in, T& podRef) {
  in.read((char*)&podRef, sizeof(T));
}

#define DEBUG_LIB 1
namespace hnswlib {
  typedef unsigned int tableint;
  typedef unsigned int linklistsizeint;
  template <typename dist_t> class HierarchicalNSW : public AlgorithmInterface<dist_t> {
  public:
    HierarchicalNSW(SpaceInterface<dist_t> *s) {

    }
    HierarchicalNSW(SpaceInterface<dist_t> *s, const string &location, bool nmslib = false) {
      LoadIndex(location, s);
    }
    HierarchicalNSW(SpaceInterface<dist_t> *s, size_t maxElements, size_t M, size_t cos_M, size_t efConstruction, size_t cos_efConstruction) :
      ll_locks(maxElements), elementLevels(maxElements) {
      maxelements_ = maxElements;

      data_size_ = s->get_data_size();
      fstdistfunc_ = s->get_dist_func();
      dist_func_param_ = s->get_dist_func_param();
      M_ = M;
      maxM_ = M_;   // max_M_ is no longer needed as we don't have inner product graphs on upper layers
      maxM0_ = M_ * 2;
      efConstruction_ = efConstruction;
      ef_ = 7;      // ?????

      cos_M_ = cos_M;
      cos_maxM_ = cos_M_;
      cos_maxM0_ = cos_M_ * 2;
      cos_efConstruction_ = cos_efConstruction;
      cos_ef_ = 1;


      size_links_level0_ip_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
      size_links_level0_cos_ = cos_maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
      size_links_level0_ = size_links_level0_ip_ + size_links_level0_cos_;
      size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(labeltype);
      offsetData_ = size_links_level0_;
      label_offset_ = size_links_level0_ + data_size_;
      //cout << offsetData_ << "\t" << label_offset_ << "\n";
      //cout << size_links_level0_ << "\t" << data_size_ << "\t" << sizeof(labeltype) << "\n";

      data_level0_memory_ = (char *)malloc(maxelements_*size_data_per_element_);

      size_t predicted_size_per_element = size_data_per_element_ + sizeof(void*) + 8 + 8 + 2 * 8;
      //cout << "size_mb=" << maxelements_*(predicted_size_per_element) / (1000 * 1000) << "\n";
      cur_element_count = 0;

      visitedlistpool = new VisitedListPool(1, maxElements);



      //initializations for special treatment of the first node
      enterpoint_node = -1;
      maxlevel_ = -1;

      linkLists_ = (char **)malloc(sizeof(void *) * maxelements_);
      size_links_per_element_ = cos_maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
      mult_ = 1 / log(1.0 * M_);
      revSize_ = 1.0 / mult_;
    }
    ~HierarchicalNSW() {

      free(data_level0_memory_);
      for (tableint i = 0; i < cur_element_count; i++) {
        if (elementLevels[i] > 0)
          free(linkLists_[i]);
      }
      free(linkLists_);
      delete visitedlistpool;
    }
    size_t maxelements_;
    size_t cur_element_count;
    size_t size_data_per_element_;
    size_t size_links_per_element_;

    size_t M_;
    size_t maxM_;
    size_t maxM0_;
    size_t efConstruction_;
    int delaunay_type_;
    double mult_, revSize_;
    int maxlevel_;

    size_t cos_M_, cos_maxM_, cos_maxM0_;
    size_t cos_efConstruction_;
    size_t cos_ef_;

    VisitedListPool *visitedlistpool;
    mutex cur_element_count_guard_;
    mutex MaxLevelGuard_;
    vector<mutex> ll_locks;
    tableint enterpoint_node;

    size_t dist_calc;
    size_t size_links_level0_;
    size_t offsetData_;

    size_t size_links_level0_ip_;
    size_t size_links_level0_cos_;

    char *data_level0_memory_;
    char **linkLists_;
    vector<int> elementLevels;
    vector<float> elementNorms;



    size_t data_size_;
    size_t label_offset_;
    DISTFUNC<dist_t> fstdistfunc_;
    void *dist_func_param_;
    std::default_random_engine generator = std::default_random_engine(100);

    inline labeltype getExternalLabel(tableint internal_id) {
      return *((labeltype *)(data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_));
    }
    inline labeltype *getExternalLabeLp(tableint internal_id) {
      return (labeltype *)(data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_);
    }
    inline char *getDataByInternalId(tableint internal_id) {
      return (data_level0_memory_ + internal_id * size_data_per_element_ + offsetData_);
    }
    int getRandomLevel(double revSize)
    {
      std::uniform_real_distribution<double> distribution(0.0, 1.0);
      double r = -log(distribution(generator)) * revSize;
      //cout << revSize;
      return (int)r;
    }

    std::priority_queue< std::pair< dist_t, tableint  >> searchBaseLayer(tableint ep, void *datapoint, int layer) {
      VisitedList *vl = visitedlistpool->getFreeVisitedList();
      vl_type *massVisited = vl->mass;
      vl_type currentV = vl->curV;

      std::priority_queue< std::pair< dist_t, tableint  >> topResults;
      std::priority_queue< std::pair< dist_t, tableint >> candidateSet;
      dist_t dist = fstdistfunc_(datapoint, getDataByInternalId(ep), dist_func_param_) / elementNorms[getExternalLabel(ep)];

      topResults.emplace(dist, ep);
      candidateSet.emplace(-dist, ep);
      massVisited[ep] = currentV;
      dist_t lowerBound = dist;

      while (!candidateSet.empty()) {

        std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();

        if ((-curr_el_pair.first) > lowerBound) {
          break;
        }
        candidateSet.pop();

        tableint curNodeNum = curr_el_pair.second;

        unique_lock<mutex> lock(ll_locks[curNodeNum]);

        int *data;// = (int *)(linkList0_ + curNodeNum * size_links_per_element0_);
        if (layer == 0)
          data = (int *)(data_level0_memory_ + curNodeNum * size_data_per_element_ + size_links_level0_ip_);
        else
          data = (int *)(linkLists_[curNodeNum] + (layer - 1) * size_links_per_element_);
        int size = *data;
        tableint *datal = (tableint *)(data + 1);
        _mm_prefetch((char *)(massVisited + *(data + 1)), _MM_HINT_T0);
        _mm_prefetch((char *)(massVisited + *(data + 1) + 64), _MM_HINT_T0);
        _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
        _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);

        for (int j = 0; j < size; j++) {
          tableint tnum = *(datal + j);
          _mm_prefetch((char *)(massVisited + *(datal + j + 1)), _MM_HINT_T0);
          _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
          if (!(massVisited[tnum] == currentV)) {
            massVisited[tnum] = currentV;
            char *currObj1 = (getDataByInternalId(tnum));

            dist_t dist = fstdistfunc_(datapoint, currObj1, dist_func_param_) / elementNorms[getExternalLabel(tnum)];
            if (topResults.top().first > dist || topResults.size() < cos_efConstruction_) {
              candidateSet.emplace(-dist, tnum);
              _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
              topResults.emplace(dist, tnum);
              if (topResults.size() > cos_efConstruction_) {
                topResults.pop();
              }
              lowerBound = topResults.top().first;
            }
          }
        }
      }
      visitedlistpool->releaseVisitedList(vl);

      return topResults;
    }
    /*
    struct CompareByFirst {
      constexpr bool operator()(pair<dist_t, tableint> const & a,
        pair<dist_t, tableint> const & b) const noexcept
      {
        return a.first < b.first;
      }
    };
    */
    std::priority_queue<std::pair<dist_t, tableint>> searchBaseLayerST(tableint ep, void *datapoint, size_t ef) {
      VisitedList *vl = visitedlistpool->getFreeVisitedList();
      vl_type *massVisited = vl->mass;
      vl_type currentV = vl->curV;

      std::priority_queue<std::pair<dist_t, tableint>> topResults;
      std::priority_queue<std::pair<dist_t, tableint>> candidateSet;
      dist_t dist = fstdistfunc_(datapoint, getDataByInternalId(ep), dist_func_param_) / elementNorms[getExternalLabel(ep)];
      dist_calc++;
      topResults.emplace(dist, ep);
      candidateSet.emplace(-dist, ep);
      massVisited[ep] = currentV;
      dist_t lowerBound = dist;

      while (!candidateSet.empty()) {

        std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();

        if ((-curr_el_pair.first) > lowerBound) {
          break;
        }
        candidateSet.pop();

        tableint curNodeNum = curr_el_pair.second;
        int *data = (int *)(data_level0_memory_ + curNodeNum * size_data_per_element_ + size_links_level0_ip_);
        int size = *data;
        _mm_prefetch((char *)(massVisited + *(data + 1)), _MM_HINT_T0);
        _mm_prefetch((char *)(massVisited + *(data + 1) + 64), _MM_HINT_T0);
        _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
        _mm_prefetch((char *)(data + 2), _MM_HINT_T0);

        for (int j = 1; j <= size; j++) {
          int tnum = *(data + j);
          _mm_prefetch((char *)(massVisited + *(data + j + 1)), _MM_HINT_T0);
          _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);////////////
          if (!(massVisited[tnum] == currentV)) {

            massVisited[tnum] = currentV;

            char *currObj1 = (getDataByInternalId(tnum));
            dist_t dist = fstdistfunc_(datapoint, currObj1, dist_func_param_) / elementNorms[getExternalLabel(tnum)];
            dist_calc++;
            if (topResults.top().first > dist || topResults.size() < ef) {
              candidateSet.emplace(-dist, tnum);
              _mm_prefetch(data_level0_memory_ + candidateSet.top().second * size_data_per_element_ + size_links_level0_ip_,///////////
                _MM_HINT_T0);////////////////////////

              topResults.emplace(dist, tnum);

              if (topResults.size() > ef) {
                topResults.pop();
              }
              lowerBound = topResults.top().first;
            }
          }
        }
      }

      visitedlistpool->releaseVisitedList(vl);
      return topResults;
    }

    std::priority_queue<std::pair<dist_t, tableint>> searchBaseLayerST_inner_product(
                vector<pair<dist_t, tableint>> starting_points,
                void *datapoint,
                size_t ef) {
      VisitedList *vl = visitedlistpool->getFreeVisitedList();
      vl_type *massVisited = vl->mass;
      vl_type currentV = vl->curV;

      std::priority_queue<std::pair<dist_t, tableint>> topResults;
      std::priority_queue<std::pair<dist_t, tableint>> precandidateSet;
      std::priority_queue<std::pair<dist_t, tableint>> candidateSet;

      for (auto& ele : starting_points) {
        topResults.emplace(ele.first, ele.second);
        precandidateSet.emplace(-ele.first, ele.second);
        massVisited[ele.second] = currentV;
      }

      // dist_t dist = fstdistfunc_(datapoint, getDataByInternalId(ep), dist_func_param_);
      // dist_calc++;
      dist_t lowerBound = topResults.top().first;

      while (!precandidateSet.empty()) {

        pair<dist_t, tableint> curr_el_pair = precandidateSet.top();
        precandidateSet.pop();

        tableint curNodeNum = curr_el_pair.second;
        int *data = (int *)(data_level0_memory_ + curNodeNum * size_data_per_element_);
        int size = *data;
        _mm_prefetch((char *)(massVisited + *(data + 1)), _MM_HINT_T0);
        _mm_prefetch((char *)(massVisited + *(data + 1) + 64), _MM_HINT_T0);
        _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
        _mm_prefetch((char *)(data + 2), _MM_HINT_T0);

        for (int j = 1; j <= size; j++) {
          int tnum = *(data + j);
          _mm_prefetch((char *)(massVisited + *(data + j + 1)), _MM_HINT_T0);
          _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);////////////
          if (!(massVisited[tnum] == currentV)) {

            massVisited[tnum] = currentV;

            char *currObj1 = (getDataByInternalId(tnum));
            dist_t dist = fstdistfunc_(datapoint, currObj1, dist_func_param_);
            dist_calc++;
            if (topResults.top().first > dist || topResults.size() < ef) {
              candidateSet.emplace(-dist, tnum);
              _mm_prefetch(data_level0_memory_ + candidateSet.top().second * size_data_per_element_,///////////
                _MM_HINT_T0);////////////////////////

              topResults.emplace(dist, tnum);

              if (topResults.size() > ef) {
                topResults.pop();
              }
              lowerBound = topResults.top().first;
            }
          }
        }
      }

      while (!candidateSet.empty()) {

        std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();

        if ((-curr_el_pair.first) > lowerBound) {
          break;
        }
        candidateSet.pop();

        tableint curNodeNum = curr_el_pair.second;
        int *data = (int *)(data_level0_memory_ + curNodeNum * size_data_per_element_);
        int size = *data;
        _mm_prefetch((char *)(massVisited + *(data + 1)), _MM_HINT_T0);
        _mm_prefetch((char *)(massVisited + *(data + 1) + 64), _MM_HINT_T0);
        _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
        _mm_prefetch((char *)(data + 2), _MM_HINT_T0);

        for (int j = 1; j <= size; j++) {
          int tnum = *(data + j);
          _mm_prefetch((char *)(massVisited + *(data + j + 1)), _MM_HINT_T0);
          _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);////////////
          if (!(massVisited[tnum] == currentV)) {

            massVisited[tnum] = currentV;

            char *currObj1 = (getDataByInternalId(tnum));
            dist_t dist = fstdistfunc_(datapoint, currObj1, dist_func_param_);
            dist_calc++;
            if (topResults.top().first > dist || topResults.size() < ef) {
              candidateSet.emplace(-dist, tnum);
              _mm_prefetch(data_level0_memory_ + candidateSet.top().second * size_data_per_element_,///////////
                _MM_HINT_T0);////////////////////////

              topResults.emplace(dist, tnum);

              if (topResults.size() > ef) {
                topResults.pop();
              }
              lowerBound = topResults.top().first;
            }
          }
        }
      }

      visitedlistpool->releaseVisitedList(vl);
      return topResults;
    }


    void getNeighborsByHeuristic2(tableint cur_c, std::priority_queue< std::pair< dist_t, tableint>> &topResults, const int NN)
    {
      if (topResults.size() < NN) {
        return;
      }
      std::priority_queue< std::pair< dist_t, tableint>> resultSet;
      std::priority_queue< std::pair< dist_t, tableint>> templist;
      vector<std::pair< dist_t, tableint>> returnlist;
      while (topResults.size() > 0) {
        resultSet.emplace(-topResults.top().first, topResults.top().second);
        topResults.pop();
      }

      while (resultSet.size()) {
        if (returnlist.size() >= NN)
          break;
        std::pair< dist_t, tableint> curen = resultSet.top();
        dist_t dist_to_query = -curen.first / elementNorms[getExternalLabel(cur_c)];
        resultSet.pop();
        bool good = true;
        for (std::pair< dist_t, tableint> curen2 : returnlist) {
          dist_t curdist =
            fstdistfunc_(getDataByInternalId(curen2.second), getDataByInternalId(curen.second), dist_func_param_)
            / elementNorms[getExternalLabel(curen2.second)] / elementNorms[getExternalLabel(curen.second)];
          // coeff * curdist < dist_to_query, we can adjust the coefficient here to get different performance
          if (curdist < dist_to_query) {
            good = false;
            break;
          }
        }
        if (good) {
          returnlist.push_back(curen);
        }
      }

      for (std::pair< dist_t, tableint> curen2 : returnlist) {

        topResults.emplace(-curen2.first, curen2.second);
      }
    }
    linklistsizeint *get_linklist0(tableint cur_c) {
      return (linklistsizeint *)(data_level0_memory_ + cur_c * size_data_per_element_);
    };
    linklistsizeint *get_linklist(tableint cur_c, int level) {
      return (linklistsizeint *)(linkLists_[cur_c] + (level - 1) * size_links_per_element_);
    };

    void mutuallyConnectNewElement(void *datapoint, tableint cur_c, std::priority_queue< std::pair< dist_t, tableint  >> topResults, int level) {

      size_t Mcurmax = level ? cos_maxM_ : cos_maxM0_;
      getNeighborsByHeuristic2(cur_c, topResults, cos_M_);
      while (topResults.size() > cos_M_) {
        throw exception();
        topResults.pop();
      }
      vector<tableint> rez;
      rez.reserve(cos_M_);
      while (topResults.size() > 0) {
        rez.push_back(topResults.top().second);
        topResults.pop();
      }
      {
        linklistsizeint *ll_cur;
        if (level == 0)
          ll_cur = (linklistsizeint *)(data_level0_memory_ + cur_c * size_data_per_element_ + size_links_level0_ip_);
        else
          ll_cur = (linklistsizeint *)(linkLists_[cur_c] + (level - 1) * size_links_per_element_);
        if (*ll_cur) {
          cout << *ll_cur << "\n";
          cout << elementLevels[cur_c] << "\n";
          cout << level << "\n";
          throw runtime_error("Should be blank");
        }
        *ll_cur = rez.size();
        tableint *data = (tableint *)(ll_cur + 1);


        for (int idx = 0; idx < rez.size(); idx++) {
          if (data[idx])
            throw runtime_error("Should be blank");
          if (level > elementLevels[rez[idx]])
            throw runtime_error("Bad level");

          data[idx] = rez[idx];
        }
      }
      for (int idx = 0; idx < rez.size(); idx++) {

        unique_lock<mutex> lock(ll_locks[rez[idx]]);

        if (rez[idx] == cur_c)
          throw runtime_error("Connection to the same element");
        linklistsizeint *ll_other;
        if (level == 0)
          ll_other = (linklistsizeint *)(data_level0_memory_ + rez[idx] * size_data_per_element_ + size_links_level0_ip_);
        else
          ll_other = (linklistsizeint *)(linkLists_[rez[idx]] + (level - 1) * size_links_per_element_);
        if (level > elementLevels[rez[idx]])
          throw runtime_error("Bad level");
        int sz_link_list_other = *ll_other;

        if (sz_link_list_other > Mcurmax || sz_link_list_other < 0)
          throw runtime_error("Bad sz_link_list_other");

        if (sz_link_list_other < Mcurmax) {
          tableint *data = (tableint *)(ll_other + 1);
          data[sz_link_list_other] = cur_c;
          *ll_other = sz_link_list_other + 1;
        }
        else {
          // finding the "weakest" element to replace it with the new one
          tableint *data = (tableint *)(ll_other + 1);
          dist_t d_max = fstdistfunc_(getDataByInternalId(cur_c), getDataByInternalId(rez[idx]), dist_func_param_)
                            / elementNorms[getExternalLabel(cur_c)] ;
          // Heuristic:
          std::priority_queue< std::pair< dist_t, tableint>> candidates;
          candidates.emplace(d_max, cur_c);

          for (int j = 0; j < sz_link_list_other; j++) {
            candidates.emplace(fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(rez[idx]), dist_func_param_) / elementNorms[getExternalLabel(data[j])], data[j]);
          }

          getNeighborsByHeuristic2(rez[idx], candidates, Mcurmax);

          int indx = 0;
          while (candidates.size() > 0) {
            data[indx] = candidates.top().second;
            candidates.pop();
            indx++;
          }
          *ll_other = indx;
          // Nearest K:
          /*int indx = -1;
          for (int j = 0; j < sz_link_list_other; j++) {
              dist_t d = fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(rez[idx]), dist_func_param_);
              if (d > d_max) {
                  indx = j;
                  d_max = d;
              }
          }
          if (indx >= 0) {
              data[indx] = cur_c;
          } */
        }

      }
    }
 
    void mips_mutuallyConnectNewElement(void *datapoint, tableint cur_c, std::priority_queue< std::pair< dist_t, tableint  >> topResults) {

      size_t Mcurmax = maxM0_;
      while (topResults.size() > maxM_) {
        topResults.pop();
      }
      vector<tableint> rez;
      rez.reserve(maxM_);
      while (topResults.size() > 0) {
        rez.push_back(topResults.top().second);
        topResults.pop();
      }
      {
        linklistsizeint *ll_cur;
        ll_cur = (linklistsizeint *)(data_level0_memory_ + cur_c * size_data_per_element_);
        if (*ll_cur) {
          cout << *ll_cur << "\n";
          throw runtime_error("Should be blank");
        }
        *ll_cur = rez.size();
        tableint *data = (tableint *)(ll_cur + 1);

        for (int idx = 0; idx < rez.size(); idx++) {
          if (data[idx])
            throw runtime_error("Should be blank");

          data[idx] = rez[idx];
        }
      }
      for (int idx = 0; idx < rez.size(); idx++) {

        // ?????????????????? 20190515
        unique_lock<mutex> lock(ll_locks[rez[idx]]);

        if (rez[idx] == cur_c)
          throw runtime_error("Connection to the same element");
        linklistsizeint *ll_other;
        ll_other = (linklistsizeint *)(data_level0_memory_ + rez[idx] * size_data_per_element_);
        int sz_link_list_other = *ll_other;

        if (sz_link_list_other > Mcurmax || sz_link_list_other < 0)
          throw runtime_error("Bad sz_link_list_other");

        if (sz_link_list_other < Mcurmax) {
          tableint *data = (tableint *)(ll_other + 1);
          data[sz_link_list_other] = cur_c;
          *ll_other = sz_link_list_other + 1;
        }
        else {
          // finding the "weakest" element to replace it with the new one
          tableint *data = (tableint *)(ll_other + 1);
          dist_t d_max = fstdistfunc_(getDataByInternalId(cur_c), getDataByInternalId(rez[idx]), dist_func_param_);
          // Heuristic:
          std::priority_queue< std::pair< dist_t, tableint>> candidates;
          candidates.emplace(d_max, cur_c);

          for (int j = 0; j < sz_link_list_other; j++) {
            candidates.emplace(fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(rez[idx]), dist_func_param_), data[j]);
          }

          // getNeighborsByHeuristic2(rez[idx], candidates, Mcurmax);
          while (candidates.size() > Mcurmax) {
            candidates.pop();
          }

          int indx = 0;
          while (candidates.size() > 0) {
            data[indx] = candidates.top().second;
            candidates.pop();
            indx++;
          }
          *ll_other = indx;
          // Nearest K:
          /*int indx = -1;
          for (int j = 0; j < sz_link_list_other; j++) {
              dist_t d = fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(rez[idx]), dist_func_param_);
              if (d > d_max) {
                  indx = j;
                  d_max = d;
              }
          }
          if (indx >= 0) {
              data[indx] = cur_c;
          } */
        }
      }
    }
    mutex global;
    size_t ef_;
    void setEf(size_t ef) {
      ef_ = ef;
    }

    void setCosEf(size_t cos_ef) {
      cos_ef_ = cos_ef;
    }

    void addPoint(void *datapoint, labeltype label, int level = -1) {

      tableint cur_c = 0;
      {
        unique_lock<mutex> lock(cur_element_count_guard_);
        if (cur_element_count >= maxelements_)
        {
          cout << "The number of elements exceeds the specified limit\n";
          throw runtime_error("The number of elements exceeds the specified limit");
        };
        cur_c = cur_element_count;
        cur_element_count++;
        if (cur_element_count % 10000 == 0) cout << "count : " << cur_element_count << endl;
      }
      unique_lock<mutex> lock_el(ll_locks[cur_c]);
      int curlevel = getRandomLevel(mult_);
      if (level > 0)
        curlevel = level;
      elementLevels[cur_c] = curlevel;



      unique_lock<mutex> templock(global);
      int maxlevelcopy = maxlevel_;
      if (curlevel <= maxlevelcopy)
        templock.unlock();
      tableint currObj = enterpoint_node;


      memset(data_level0_memory_ + cur_c * size_data_per_element_, 0, size_data_per_element_);
      // Initialisation of the data and label            
      memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));
      memcpy(getDataByInternalId(cur_c), datapoint, data_size_);


      if (curlevel) {
        linkLists_[cur_c] = (char*)malloc(size_links_per_element_*curlevel);
        memset(linkLists_[cur_c], 0, size_links_per_element_*curlevel);
      }
      if (currObj != -1) {

        if (curlevel < maxlevelcopy) {

          // 20190515 I am not sure if it should be divided by a single norm(id) or both norms (getting actual cosine value)
          dist_t curdist = fstdistfunc_(datapoint, getDataByInternalId(currObj), dist_func_param_) / elementNorms[getExternalLabel(currObj)];
          for (int level = maxlevelcopy; level > curlevel; level--) {
            bool changed = true;
            while (changed) {
              changed = false;
              int *data;
              unique_lock<mutex> lock(ll_locks[currObj]);
              data = (int *)(linkLists_[currObj] + (level - 1) * size_links_per_element_);
              int size = *data;
              tableint *datal = (tableint *)(data + 1);
              for (int i = 0; i < size; i++) {
                tableint cand = datal[i];
                if (cand<0 || cand>maxelements_)
                  throw runtime_error("cand error");
                dist_t d = fstdistfunc_(datapoint, getDataByInternalId(cand), dist_func_param_) / elementNorms[getExternalLabel(cand)];
                if (d < curdist) {
                  curdist = d;
                  currObj = cand;
                  changed = true;
                }
              }
            }
          }
        }

        for (int level = min(curlevel, maxlevelcopy); level >= 0; level--) {
          if (level > maxlevelcopy || level < 0)
            throw runtime_error("Level error");

          // currObj may be changed during the iteration; worth a trial
          std::priority_queue< std::pair< dist_t, tableint  >> topResults = searchBaseLayer(currObj, datapoint, level);
          mutuallyConnectNewElement(datapoint, cur_c, topResults, level);

          if (level == 0) {
            // put elements in topResults and their neighbors (even neighbors of mips neighbors) into MIPS queue
            vector<pair<dist_t, tableint>> cos_res;
            cos_res.reserve(cos_M_);
            while (topResults.size() > 0) {
              tableint tmp_id = topResults.top().second;
              dist_t tmp_ip = topResults.top().first * elementNorms[getExternalLabel(tmp_id)];
              cos_res.push_back(make_pair(tmp_ip, tmp_id));
              topResults.pop();
            }
            std::priority_queue<std::pair<dist_t, tableint>> mips_topResults = searchBaseLayerST_inner_product(cos_res, datapoint, efConstruction_);
            // I don't need heuristic for mips inside the following function
            mips_mutuallyConnectNewElement(datapoint, cur_c, mips_topResults);
          }
        }
      }

      else {
        // Do nothing for the first element
        enterpoint_node = 0;
        maxlevel_ = curlevel;

      }

      //Releasing lock for the maximum level
      if (curlevel > maxlevelcopy) {
        enterpoint_node = cur_c;
        maxlevel_ = curlevel;
      }
    };
    std::priority_queue< std::pair< dist_t, labeltype >> searchKnn(void *query_data, int k) {
                       
      tableint currObj = enterpoint_node;
      dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node), dist_func_param_) / elementNorms[getExternalLabel(enterpoint_node)];
      dist_calc++;
      for (int level = maxlevel_; level > 0; level--) {
        bool changed = true;
        while (changed) {
          changed = false;
          int *data;
          data = (int *)(linkLists_[currObj] + (level - 1) * size_links_per_element_);
          int size = *data;
          tableint *datal = (tableint *)(data + 1);
          for (int i = 0; i < size; i++) {
            tableint cand = datal[i];
            if (cand<0 || cand>maxelements_)
              throw runtime_error("cand error");
            dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_) / elementNorms[getExternalLabel(cand)];
            dist_calc++;
            if (d < curdist) {
              curdist = d;
              currObj = cand;
              changed = true;
            }
          }
        }
      }


      // here I set cos_ef_ to cos_M_
      // zzzzz
      std::priority_queue<std::pair< dist_t, tableint>> topResults = searchBaseLayerST(currObj, query_data, cos_ef_);
            
      vector<pair<dist_t, tableint>> cos_res;

      // cos_res.reserve(cos_M_);


//      printf("size = %d cos_ef = %d\n",cos_ef_, topResults.size());
      while (topResults.size() > 0) {
        tableint tmp_id = topResults.top().second;
//        tmp_id = 0;
        dist_t tmp_ip = topResults.top().first * elementNorms[getExternalLabel(tmp_id)];
        cos_res.push_back(make_pair(tmp_ip, tmp_id));
        topResults.pop();
      }
      std::priority_queue<std::pair<dist_t, tableint>> mips_topResults = searchBaseLayerST_inner_product(cos_res, query_data, ef_);


      std::priority_queue< std::pair< dist_t, labeltype >> results;
      while (mips_topResults.size() > k) {
        mips_topResults.pop();
      }
      while (mips_topResults.size() > 0) {
        std::pair<dist_t, tableint> rez = mips_topResults.top();
        results.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
        mips_topResults.pop();
      }
      return results;
    };

    std::priority_queue< std::pair< dist_t, tableint>> searchKnnInternal(void *query_data, int k) {
      tableint currObj = enterpoint_node;
      dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node), dist_func_param_);
      dist_calc++;
      for (int level = maxlevel_; level > 0; level--) {
        bool changed = true;
        while (changed) {
          changed = false;
          int *data;
          data = (int *)(linkLists_[currObj] + (level - 1) * size_links_per_element_);
          int size = *data;
          tableint *datal = (tableint *)(data + 1);
          for (int i = 0; i < size; i++) {
            tableint cand = datal[i];
            if (cand<0 || cand>maxelements_)
              throw runtime_error("cand error");
            dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);
            dist_calc++;
            if (d < curdist) {
              curdist = d;
              currObj = cand;
              changed = true;
            }
          }
        }
      }

      //std::priority_queue< std::pair< dist_t, tableint  >> topResults = searchBaseLayer(currObj, query_data, 0);
      std::priority_queue< std::pair< dist_t, tableint  >> topResults = searchBaseLayerST(currObj, query_data, ef_);
      while (topResults.size() > k) {
        topResults.pop();
      }
      return topResults;
    };
    void       SaveIndex(const string &location)
    {

      cout << "Saving index to " << location.c_str() << "\n";
      std::ofstream output(location, std::ios::binary);
      streampos position;

      writeBinaryPOD(output, maxelements_);
      writeBinaryPOD(output, size_data_per_element_);
	  output.write(data_level0_memory_, maxelements_*size_data_per_element_);
      writeBinaryPOD(output, maxM0_);
	  writeBinaryPOD(output, offsetData_);
      writeBinaryPOD(output, label_offset_);
	  //----------------------------------------------
	  for (size_t i = 0; i < maxelements_; ++i) {
        writeBinaryPOD(output, elementNorms[i]);
      }
	
	  writeBinaryPOD(output, cur_element_count);
      //writeBinaryPOD(output, label_offset_);
      writeBinaryPOD(output, maxlevel_);
      writeBinaryPOD(output, enterpoint_node);
      writeBinaryPOD(output, maxM_);

      writeBinaryPOD(output, M_);
      writeBinaryPOD(output, mult_);
      writeBinaryPOD(output, efConstruction_);


      writeBinaryPOD(output, cos_M_);
      writeBinaryPOD(output, cos_efConstruction_);

     // output.write(data_level0_memory_, maxelements_*size_data_per_element_);

      for (size_t i = 0; i < maxelements_; i++) {
        unsigned int linkListSize = elementLevels[i] > 0 ? size_links_per_element_*elementLevels[i] : 0;
        writeBinaryPOD(output, linkListSize);
        if (linkListSize)
          output.write(linkLists_[i], linkListSize);
      }
      output.close();
    }

    void       LoadIndex(const string &location, SpaceInterface<dist_t> *s)
    {

//------------------------------

      std::ifstream input(location, std::ios::binary);
      readBinaryPOD(input, maxelements_);
      readBinaryPOD(input, size_data_per_element_);
      data_level0_memory_ = (char *)malloc(maxelements_*size_data_per_element_);
      input.read(data_level0_memory_, maxelements_*size_data_per_element_);
      readBinaryPOD(input, maxM0_);
      readBinaryPOD(input, offsetData_);
      readBinaryPOD(input, label_offset_);

      for (size_t i = 0; i < maxelements_; ++i) {
        float temp;
        readBinaryPOD(input, temp);
        elementNorms.push_back(temp);
      }
//------------------------

      //cout << "Loading index from " << location;
     // std::ifstream input(location, std::ios::binary);
      streampos position;

 //     readBinaryPOD(input, maxelements_);
      readBinaryPOD(input, cur_element_count);
  //    readBinaryPOD(input, size_data_per_element_);
  //    readBinaryPOD(input, label_offset_);
  //    readBinaryPOD(input, offsetData_);
      readBinaryPOD(input, maxlevel_);
      readBinaryPOD(input, enterpoint_node);

      readBinaryPOD(input, maxM_);
  //    readBinaryPOD(input, maxM0_);
      readBinaryPOD(input, M_);
      readBinaryPOD(input, mult_);
      readBinaryPOD(input, efConstruction_);
      cout << efConstruction_ << "\n";

      readBinaryPOD(input, cos_M_);
      readBinaryPOD(input, cos_efConstruction_);
/*
      for (size_t i = 0; i < maxelements_; ++i) {
        float temp;
        readBinaryPOD(input, temp);
        elementNorms.push_back(temp);
      }
*/
      printf("label_offset_ = %d\n", label_offset_);
      data_size_ = s->get_data_size();
      fstdistfunc_ = s->get_dist_func();
      dist_func_param_ = s->get_dist_func_param();

    //  data_level0_memory_ = (char *)malloc(maxelements_*size_data_per_element_);
    //  input.read(data_level0_memory_, maxelements_*size_data_per_element_);


      cos_maxM_ = cos_M_;
      cos_maxM0_ = 2 * cos_M_;
      size_links_level0_ip_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
      size_links_level0_cos_ = cos_maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
      size_links_level0_ = offsetData_;

      size_links_per_element_ = cos_maxM_ * sizeof(tableint) + sizeof(linklistsizeint);

      visitedlistpool = new VisitedListPool(1, maxelements_);


      linkLists_ = (char **)malloc(sizeof(void *) * maxelements_);
      cout << maxelements_ << "\n";
      elementLevels = vector<int>(maxelements_);
      revSize_ = 1.0 / mult_;
      ef_ = 10;
      for (size_t i = 0; i < maxelements_; i++) {
        unsigned int linkListSize;
        readBinaryPOD(input, linkListSize);
        if (linkListSize == 0) {
          elementLevels[i] = 0;

          linkLists_[i] = nullptr;
        }
        else {
          elementLevels[i] = linkListSize / size_links_per_element_;
          linkLists_[i] = (char *)malloc(linkListSize);
          input.read(linkLists_[i], linkListSize);
        }
      }


      input.close();
      size_t predicted_size_per_element = size_data_per_element_ + sizeof(void*) + 8 + 8 + 2 * 8;
      cout << "Loaded index, predicted size=" << maxelements_*(predicted_size_per_element) / (1000 * 1000) << "\n";
      return;
    }
  };

}
