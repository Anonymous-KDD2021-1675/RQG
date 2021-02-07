#pragma once

#include "visited_list_pool.h"
#include "hnswlib.h"
#include <random>
#include <stdlib.h>
#include <unordered_set>
#include <list>
#include <chrono>
#include <ctime>

//#define X 16

float quan_dist(unsigned char* id, float** quan_book, unsigned fid){
	float dist = 0;
	for(int i = 0; i < 32; i++){
		dist += quan_book[i][ id[i] ];
	}
	return dist;
}

float fstl2func_( unsigned char* a, unsigned char* b, float*** dist_book, int size){
	float dist = 0;
	for(int i = 0; i < size; i++){
		dist += dist_book[i][a[i]][b[i]];
	}
	return dist;
}

    float compare(const float* a, const float* b, unsigned size) {
      float result = 0;
#ifdef __GNUC__
#ifdef __AVX__
      #define AVX_DOT(addr1, addr2, dest, tmp1, tmp2) \
          tmp1 = _mm256_loadu_ps(addr1);\
          tmp2 = _mm256_loadu_ps(addr2);\
          tmp1 = _mm256_mul_ps(tmp1, tmp2); \
          dest = _mm256_add_ps(dest, tmp1);

	  __m256 sum;
   	  __m256 l0, l1;
   	  __m256 r0, r1;
      unsigned D = (size + 7) & ~7U;
      unsigned DR = D % 16;
      unsigned DD = D - DR;
   	  const float *l = a;
   	  const float *r = b;
      const float *e_l = l + DD;
   	  const float *e_r = r + DD;
      float unpack[8] __attribute__ ((aligned (32))) = {0, 0, 0, 0, 0, 0, 0, 0};

      sum = _mm256_loadu_ps(unpack);
      if(DR){AVX_DOT(e_l, e_r, sum, l0, r0);}

      for (unsigned i = 0; i < DD; i += 16, l += 16, r += 16) {
	    AVX_DOT(l, r, sum, l0, r0);
	    AVX_DOT(l + 8, r + 8, sum, l1, r1);
      }
      _mm256_storeu_ps(unpack, sum);
      result = unpack[0] + unpack[1] + unpack[2] + unpack[3] + unpack[4] + unpack[5] + unpack[6] + unpack[7];

#else
#ifdef __SSE2__
      #define SSE_DOT(addr1, addr2, dest, tmp1, tmp2) \
          tmp1 = _mm128_loadu_ps(addr1);\
          tmp2 = _mm128_loadu_ps(addr2);\
          tmp1 = _mm128_mul_ps(tmp1, tmp2); \
          dest = _mm128_add_ps(dest, tmp1);
      __m128 sum;
      __m128 l0, l1, l2, l3;
      __m128 r0, r1, r2, r3;
      unsigned D = (size + 3) & ~3U;
      unsigned DR = D % 16;
      unsigned DD = D - DR;
      const float *l = a;
      const float *r = b;
      const float *e_l = l + DD;
      const float *e_r = r + DD;
      float unpack[4] __attribute__ ((aligned (16))) = {0, 0, 0, 0};

      sum = _mm_load_ps(unpack);
      switch (DR) {
          case 12:
          SSE_DOT(e_l+8, e_r+8, sum, l2, r2);
          case 8:
          SSE_DOT(e_l+4, e_r+4, sum, l1, r1);
          case 4:
          SSE_DOT(e_l, e_r, sum, l0, r0);
        default:
          break;
      }
      for (unsigned i = 0; i < DD; i += 16, l += 16, r += 16) {
          SSE_DOT(l, r, sum, l0, r0);
          SSE_DOT(l + 4, r + 4, sum, l1, r1);
          SSE_DOT(l + 8, r + 8, sum, l2, r2);
          SSE_DOT(l + 12, r + 12, sum, l3, r3);
      }
      _mm_storeu_ps(unpack, sum);
      result += unpack[0] + unpack[1] + unpack[2] + unpack[3];
#else

      float dot0, dot1, dot2, dot3;
      const float* last = a + size;
      const float* unroll_group = last - 3;

      /* Process 4 items with each loop for efficiency. */
      while (a < unroll_group) {
          dot0 = a[0] * b[0];
          dot1 = a[1] * b[1];
          dot2 = a[2] * b[2];
          dot3 = a[3] * b[3];
          result += dot0 + dot1 + dot2 + dot3;
          a += 4;
          b += 4;
      }
      /* Process last 0-3 pixels.  Not needed for standard vector lengths. */
      while (a < last) {
          result += *a++ * *b++;
      }
#endif
#endif
#endif
      return result;
    }

namespace hnswlib {
    typedef unsigned int tableint;
    typedef unsigned int linklistsizeint;

    template<typename dist_t>
    class HierarchicalNSW : public AlgorithmInterface<dist_t> {
    public:

        HierarchicalNSW(SpaceInterface<dist_t> *s) {

        }

        HierarchicalNSW(SpaceInterface<dist_t> *s, const std::string &location, int num, const std::string &location2, bool nmslib = false, size_t max_elements=0) {
			    loadIndex(num, location, location2, s, max_elements);
        }

        HierarchicalNSW(SpaceInterface<dist_t> *s, size_t max_elements, int* ring_elements, int num_ring, size_t M = 16, size_t ef_construction = 500, size_t random_seed = 100) :
                link_list_locks_(max_elements) {
            max_elements_ = max_elements;
            fstdistfunc_ = s->get_dist_func();
            dist_func_param_ = s->get_dist_func_param();
			
            M_ = M;
            maxM_ = M_;
            maxM0_ = 2 * M_;
            ef_construction_ = std::max(ef_construction,M_);
            ef_ = 10;
			quan_size_ = 32;


           if(ring_elements != NULL){
			trans_quan = new int[max_elements];

			num_ring_ = num_ring;
			ring_elements_ = new int[num_ring];
			for(int i = 0; i < num_ring; i++) 
				ring_elements_[i] = ring_elements[i] + 1;
			
            offsetData_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint) + quan_size_ + sizeof(float);
            size_quan_per_element_ = 2 * (offsetData_);
			
			offsetData2_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint); 		
			quan_level_memory_ = (char **) malloc(sizeof(void *) * num_ring_);
					
			for(int i = 0; i < num_ring_; i++){
				quan_level_memory_[i] =  (char *) malloc(ring_elements_[i] * size_quan_per_element_);
				if(quan_level_memory_[i] == nullptr)
					throw std::runtime_error("Not enough memory");
			}
			
            cur_element_count = 0;

            visited_list_pool_ = new VisitedListPool(1, max_elements);

            enterpoint_node_ = -1;
			maxlevel_ = num_ring_ - 1;
            }
        }

        struct CompareByFirst {
            constexpr bool operator()(std::pair<dist_t, tableint> const &a,
                                      std::pair<dist_t, tableint> const &b) const noexcept {
                return a.first < b.first;
            }
        };

        ~HierarchicalNSW() {
            for (tableint i = 0; i < num_ring_; i++) {
                free(quan_level_memory_[i]);
            }
			free(quan_level_memory_);
            delete visited_list_pool_;
        }

        size_t max_elements_;
        size_t cur_element_count;
		size_t size_quan_per_element_;
        size_t size_data_per_element_;
		
        size_t M_;
        size_t maxM_;
        size_t maxM0_;
        size_t ef_construction_;
        size_t ef_;
		int num_ring_;
        int maxlevel_;
		int quan_size_;
        int* ring_elements_;
		char** quan_level_memory_;
		char* data_level0_memory_;

                size_t label_offset2;
		int* trans_quan;
		int* trans_;
		DISTFUNC<dist_t> fstdistfunc_;
        void *dist_func_param_;
	        std::vector<float> elementNorms;
		double dist_calc;
		size_t size_links_level0_ip_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
			
        VisitedListPool *visited_list_pool_;
        std::mutex cur_element_count_guard_;

        std::vector<std::mutex> link_list_locks_;
        tableint enterpoint_node_;

        size_t offsetData_, offsetData2_, offsetLevel0_;
		size_t offsetData0_;

         inline labeltype getExternalLabel(tableint internal_id) const{
               return *( (labeltype*) (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset2) ); 

         }
         		
		inline char *getDataByInternalId(tableint internal_id) const{
             return (data_level0_memory_ + internal_id * size_data_per_element_ + offsetData0_);
        }
		
        inline float *getNormByInternalId(tableint internal_id, int i) const {  
			return (float*)(quan_level_memory_[i] + internal_id * size_quan_per_element_);
        }
		
        inline char *getQuanByInternalId(tableint internal_id, int i) const {  
			return (quan_level_memory_[i] + internal_id * size_quan_per_element_ + sizeof(float));
        }		

        inline linklistsizeint *getLinkByInternalId(tableint internal_id, int i) const { 
			return (linklistsizeint *)(quan_level_memory_[i] + internal_id * size_quan_per_element_ + sizeof(float) + quan_size_);
        }			
	    
			
        inline float *getNormByInternalId2(tableint internal_id, int i) const {  
			return (float*)(quan_level_memory_[i] + internal_id * size_quan_per_element_ + offsetData_);
        }
		
        inline char *getQuanByInternalId2(tableint internal_id, int i) const {  
			return (quan_level_memory_[i] + internal_id * size_quan_per_element_ + offsetData_ + sizeof(float));
        }		

        inline linklistsizeint *getLinkByInternalId2(tableint internal_id, int i) const { 
			return (linklistsizeint *)(quan_level_memory_[i] + internal_id * size_quan_per_element_ + offsetData_ + sizeof(float) + quan_size_);
        }			
		
		
	 	static inline int InsertIntoPool (Neighbor *addr, unsigned K, Neighbor nn) {
        // find the location to insert
            int left=0,right=K-1;
            if(addr[left].distance>nn.distance){
                memmove((char *)&addr[left+1], &addr[left],K * sizeof(Neighbor));
                addr[left] = nn;
                return left;
            }
            if(addr[right].distance<nn.distance){
                addr[K] = nn;
                return K;
            }
            while(left<right-1){
                int mid=(left+right)/2;
                if(addr[mid].distance>nn.distance)right=mid;
                else left=mid;
            }
            //check equal ID

            while (left > 0){
                if (addr[left].distance < nn.distance) break;
                if (addr[left].id == nn.id) return K + 1;
                left--;
            }
            if(addr[left].id == nn.id||addr[right].id==nn.id)return K+1;
            memmove((char *)&addr[right+1], &addr[right],(K-right) * sizeof(Neighbor));
            addr[right]=nn;
            return right;
        }    		
		
	 	static inline int findinPool(Neighbor *addr, unsigned K, float val) {
        // find the location to insert
            int left=0,right=K-1;
            if(addr[left].distance>val){
                return left;
            }
            if(addr[right].distance<val){
                return K;
            }
            while(left<right-1){
                int mid=(left+right)/2;
                if(addr[mid].distance>val)right=mid;
                else left=mid;
            }
            return right;
        }		

    std::priority_queue<std::pair<dist_t, tableint>> searchBaseLayerST_inner_product(
                std::vector<std::pair<dist_t, tableint>> starting_points,
                const void *datapoint,
                size_t ef) {
      VisitedList *vl = visited_list_pool_->getFreeVisitedList();
      vl_type *massVisited = vl->mass;
      vl_type currentV = vl->curV;

      std::priority_queue<std::pair<dist_t, tableint>> topResults;
      std::priority_queue<std::pair<dist_t, tableint>> precandidateSet;
      std::priority_queue<std::pair<dist_t, tableint>> candidateSet;

      for (auto& ele : starting_points) {
        topResults.emplace(ele.first, ele.second);
        candidateSet.emplace(-ele.first, ele.second);
        massVisited[ele.second] = currentV;
      }

      dist_t lowerBound = topResults.top().first;

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
        _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData0_, _MM_HINT_T0);
        _mm_prefetch((char *)(data + 2), _MM_HINT_T0);

        for (int j = 1; j <= size; j++) {
          int tnum = *(data + j);
          _mm_prefetch((char *)(massVisited + *(data + j + 1)), _MM_HINT_T0);
          _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData0_, _MM_HINT_T0);////////////
          if (!(massVisited[tnum] == currentV)) {

            massVisited[tnum] = currentV;

            char *currObj1 = (getDataByInternalId(tnum));
            dist_t dist = fstdistfunc_(datapoint, currObj1, dist_func_param_);
            dist_calc += 1;
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

      visited_list_pool_->releaseVisitedList(vl);
      return topResults;
    }

		
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
        searchBaseLayer(tableint ep_id, const void *data_point, int level, bool flag, float norm, float*** dist_book, labeltype j2) {
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;

            dist_t lowerBound;
			dist_t dist;
			float curnorm;

			if(flag == true)
                dist = fstl2func_((unsigned char*) data_point, (unsigned char*)getQuanByInternalId(ep_id, level), dist_book, quan_size_);
            else{
				curnorm =  *(getNormByInternalId2(ep_id, level)) * norm;
				dist = fstl2func_((unsigned char*)data_point, (unsigned char*)getQuanByInternalId2(ep_id, level), dist_book, quan_size_) * curnorm;
			}
				
			top_candidates.emplace(dist, ep_id);
            lowerBound = dist;
            candidateSet.emplace(-dist, ep_id);

            visited_array[ep_id] = visited_array_tag;

            while (!candidateSet.empty()) {
                std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
                if ((-curr_el_pair.first) > lowerBound) {
                    break;
                }
                candidateSet.pop();

                tableint curNodeNum = curr_el_pair.second;

                std::unique_lock <std::mutex> lock(link_list_locks_[curNodeNum]);

              unsigned int *data;// = (int *)(linkList0_ + curNodeNum * size_links_per_element0_);				
				if(flag == true)
				    data = getLinkByInternalId(curNodeNum, level);
				else {data = getLinkByInternalId2(curNodeNum, level);}
				
                size_t size = getListCount((linklistsizeint*)data);
                tableint *datal = (tableint *) (data + 1);
				
/*			
#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
                _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif
*/
                dist_t dist1;
				char *currObj1;
                for (size_t j = 0; j < size; j++) {
                    tableint candidate_id = *(datal + j);
//                    if (candidate_id == 0) continue;

/*
#ifdef USE_SSE
                    _mm_prefetch((char *) (visited_array + *(datal + j + 1)), _MM_HINT_T0);
                    _mm_prefetch(getQuanByInternalId(*(datal + j + 1), level), _MM_HINT_T0);
#endif
 */            
                    if (visited_array[candidate_id] == visited_array_tag) continue;
                    visited_array[candidate_id] = visited_array_tag;
                    
					if(flag == true){
				       	currObj1 = getQuanByInternalId(candidate_id, level);
                        dist1 = fstl2func_((unsigned char*)data_point, (unsigned char*)currObj1, dist_book, quan_size_);
					}
					else{
						currObj1 = getQuanByInternalId2(candidate_id, level);
					    curnorm = *(getNormByInternalId2(candidate_id, level)) * norm;
					    dist1 = fstl2func_((unsigned char*)data_point, (unsigned char*)currObj1, dist_book, quan_size_) * curnorm;                                        	
					}
					
					
                    if (top_candidates.size() < ef_construction_ || lowerBound > dist1) {
                        candidateSet.emplace(-dist1, candidate_id);
						
/*						
#ifdef USE_SSE
                        _mm_prefetch(getQuanByInternalId(candidateSet.top().second, level), _MM_HINT_T0);
#endif
*/

                        //if (!isMarkedDeleted(candidate_id))
                            top_candidates.emplace(dist1, candidate_id);

                        if (top_candidates.size() > ef_construction_)
                            top_candidates.pop();

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
            visited_list_pool_->releaseVisitedList(vl);

            return top_candidates;
        }

        void getNeighborsByHeuristic2(
                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
                const size_t M, int level, float*** dist_book) {
            if (top_candidates.size() < M) {
                return;
            }
            std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
            std::vector<std::pair<dist_t, tableint>> return_list;
            while (top_candidates.size() > 0) {
                queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
                top_candidates.pop();
            }

            while (queue_closest.size()) {
                if (return_list.size() >= M)
                    break;
                std::pair<dist_t, tableint> curent_pair = queue_closest.top();
                dist_t dist_to_query = -curent_pair.first;
                queue_closest.pop();
                bool good = true;
                for (std::pair<dist_t, tableint> second_pair : return_list) {
                    dist_t curdist =
                            fstl2func_((unsigned char*)getQuanByInternalId(second_pair.second, level),
                                         (unsigned char*)getQuanByInternalId(curent_pair.second, level),
                                         dist_book, quan_size_);;
                    if (curdist < dist_to_query) {
                        good = false;
                        break;
                    }
                }
                if (good) {
                    return_list.push_back(curent_pair);
                }


            }

            for (std::pair<dist_t, tableint> curent_pair : return_list) {

                top_candidates.emplace(-curent_pair.first, curent_pair.second);
            }
        }
		
    void getNeighborsByHeuristic3(
	        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &topResults,
 			const size_t NN, int level, float*** dist_book)
    {
      if (topResults.size() < NN) {
        return;
      }
      std::priority_queue< std::pair< dist_t, tableint>> resultSet;
      std::priority_queue< std::pair< dist_t, tableint>> templist;
      std::vector<std::pair< dist_t, tableint>> returnlist;
      while (topResults.size() > 0) {
        resultSet.emplace(-topResults.top().first, topResults.top().second);
        topResults.pop();
      }

      while (resultSet.size()) {
        if (returnlist.size() >= NN)
          break;
        std::pair< dist_t, tableint> curen = resultSet.top();
        dist_t dist_to_query = -curen.first;
        resultSet.pop();
        bool good = true;
        for (std::pair< dist_t, tableint> curen2 : returnlist) {
		  float curnorm = *(getNormByInternalId2(curen2.second, level)) * *(getNormByInternalId2(curen.second, level));
          dist_t curdist =
            curnorm * fstl2func_((unsigned char*)getQuanByInternalId2(curen2.second, level), (unsigned char*)getQuanByInternalId2(curen.second, level), 
			        dist_book, quan_size_);;
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

        void mutuallyConnectNewElement(const void *data_point, tableint cur_c,
                                       std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates,
									   std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> ip_candidates,
                                       int level, float*** dist_book, labeltype label) {

            size_t Mcurmax = maxM_;

            getNeighborsByHeuristic2(top_candidates, M_, level, dist_book);		
			getNeighborsByHeuristic3(ip_candidates, M_, level, dist_book);
			
            if (top_candidates.size() > M_ || ip_candidates.size() > M_)
                throw std::runtime_error("Should be not be more than M_ candidates returned by the heuristic");

            std::vector<tableint> selectedNeighbors;
            selectedNeighbors.reserve(M_);
            while (top_candidates.size() > 0) {
                selectedNeighbors.push_back(top_candidates.top().second);
                top_candidates.pop();
            }
			
			std::vector<tableint> rez;
            rez.reserve(M_);
            while (ip_candidates.size() > 0) {
                rez.push_back(ip_candidates.top().second);
                ip_candidates.pop();
            }			

            {
                linklistsizeint *ll_cur = getLinkByInternalId(cur_c, level);

                if (*ll_cur) {
                    throw std::runtime_error("The newly inserted element should have blank link list");
                }
                setListCount(ll_cur,selectedNeighbors.size());
                tableint *data = (tableint *) (ll_cur + 1);


                for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
                    if (data[idx])
                        throw std::runtime_error("Possible memory corruption");
                    data[idx] = selectedNeighbors[idx];

                }
				
				//-------------------
                ll_cur = getLinkByInternalId2(cur_c, level);
                if (*ll_cur) {
                    throw std::runtime_error("The newly inserted element should have blank link list");
                }

				setListCount(ll_cur, rez.size());
                tableint *data2 = (tableint *) (ll_cur + 1);


                for (size_t idx = 0; idx < rez.size(); idx++) {
                    if (data2[idx])
                        throw std::runtime_error("Possible memory corruption");
                    data2[idx] = rez[idx];

                }	
            }

            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {

                std::unique_lock <std::mutex> lock(link_list_locks_[selectedNeighbors[idx]]);


                linklistsizeint *ll_other = getLinkByInternalId(selectedNeighbors[idx], level);
                size_t sz_link_list_other = getListCount(ll_other);

                if (sz_link_list_other > Mcurmax)
                    throw std::runtime_error("Bad value of sz_link_list_other");
                if (selectedNeighbors[idx] == cur_c)
                    throw std::runtime_error("Trying to connect an element to itself");

                tableint *data = (tableint *) (ll_other + 1);
                if (sz_link_list_other < Mcurmax) {
                    data[sz_link_list_other] = cur_c;
                    setListCount(ll_other, sz_link_list_other + 1);
                } else {
                    // finding the "weakest" element to replace it with the new one
                    dist_t d_max = fstl2func_((unsigned char*)getQuanByInternalId(cur_c, level), (unsigned char*)getQuanByInternalId(selectedNeighbors[idx], level),
                                                dist_book, quan_size_);
                    // Heuristic:
                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                    candidates.emplace(d_max, cur_c);

                    for (size_t j = 0; j < sz_link_list_other; j++) {
                        candidates.emplace(
                                fstl2func_((unsigned char*)getQuanByInternalId(data[j], level), (unsigned char*)getQuanByInternalId(selectedNeighbors[idx], level),
                                             dist_book, quan_size_), data[j]);
                    }

                    getNeighborsByHeuristic2(candidates, Mcurmax, level, dist_book);

                    int indx = 0;
                    while (candidates.size() > 0) {
                        data[indx] = candidates.top().second;
                        candidates.pop();
                        indx++;
                    }
                    setListCount(ll_other, indx);
                }

            }
						
            for (size_t idx = 0; idx < rez.size(); idx++) {

                std::unique_lock <std::mutex> lock(link_list_locks_[rez[idx]]);


                linklistsizeint *ll_other = getLinkByInternalId2(rez[idx], level);
                size_t sz_link_list_other = getListCount(ll_other);

                if (sz_link_list_other > Mcurmax)
                    throw std::runtime_error("Bad value of sz_link_list_other");
                if (rez[idx] == cur_c)
                    throw std::runtime_error("Trying to connect an element to itself");

                tableint *data = (tableint *) (ll_other + 1);
                if (sz_link_list_other < Mcurmax) {
                    data[sz_link_list_other] = cur_c;
                    setListCount(ll_other, sz_link_list_other + 1);
                } else {
                    // finding the "weakest" element to replace it with the new one
					float norm = *(getNormByInternalId2(cur_c, level)) * *(getNormByInternalId2(rez[idx], level));
                    dist_t d_max = norm * fstl2func_((unsigned char*)getQuanByInternalId2(cur_c, level), (unsigned char*)getQuanByInternalId2(rez[idx], level), 
					        dist_book, quan_size_);
                    // Heuristic:
                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                    candidates.emplace(d_max, cur_c);

                    for (size_t j = 0; j < sz_link_list_other; j++) {
						
						norm = *(getNormByInternalId2(data[j], level)) * *(getNormByInternalId2(rez[idx], level));
						
                        candidates.emplace(
                                norm * fstl2func_((unsigned char*)getQuanByInternalId2(data[j], level), (unsigned char*)getQuanByInternalId2(rez[idx], level),
                                            dist_book, quan_size_), data[j]);
                    }

                    getNeighborsByHeuristic3(candidates, Mcurmax, level, dist_book);

                    int indx = 0;
                    while (candidates.size() > 0) {
                        data[indx] = candidates.top().second;
                        candidates.pop();
                        indx++;
                    }
                    setListCount(ll_other, indx);
                }

            }			
        }

        std::mutex global;
        //size_t ef_;

        void setEf(size_t ef) {
            ef_ = ef;
        }

        void saveIndex(const std::string &location) {
            std::ofstream output(location, std::ios::binary);
            std::streampos position;

            writeBinaryPOD(output, max_elements_);
            writeBinaryPOD(output, cur_element_count);
            writeBinaryPOD(output, size_quan_per_element_);
            
            writeBinaryPOD(output, offsetData_);
            writeBinaryPOD(output, maxlevel_);
            writeBinaryPOD(output, enterpoint_node_);
            writeBinaryPOD(output, maxM_);

            writeBinaryPOD(output, maxM0_);
            writeBinaryPOD(output, M_);
            
            writeBinaryPOD(output, ef_construction_);

			for(int i = 0; i < num_ring_; i++){
                writeBinaryPOD(output, ring_elements_[i]);
				output.write(quan_level_memory_[i], ring_elements_[i] * size_quan_per_element_);
			}
			output.write((char *) trans_quan, 4 * max_elements_);
			
            output.close();
        }

        void loadIndex(int num, const std::string &location, const std::string &location2, SpaceInterface<dist_t> *s, size_t max_elements_i=0) {
          
            dist_calc = 0;
            quan_size_ = 32;
            std::ifstream input(location, std::ios::binary);
            std::ifstream input2(location2, std::ios::binary);
            if (!input.is_open() || !input2.is_open())
                throw std::runtime_error("Cannot open file");
            
            readBinaryPOD(input, max_elements_);
            readBinaryPOD(input, cur_element_count);

            trans_quan = new int[max_elements_];

            size_t max_elements=max_elements_i;
            if(max_elements < cur_element_count)
                max_elements = max_elements_;
            max_elements_ = max_elements;
            readBinaryPOD(input, size_quan_per_element_);
            
            readBinaryPOD(input, offsetData_);
            readBinaryPOD(input, maxlevel_);
            readBinaryPOD(input, enterpoint_node_);

            readBinaryPOD(input, maxM_);
            readBinaryPOD(input, maxM0_);
            readBinaryPOD(input, M_);
            
            readBinaryPOD(input, ef_construction_);          
            fstdistfunc_ = s->get_dist_func();
            dist_func_param_ = s->get_dist_func_param();

			num_ring_ = maxlevel_ + 1;

			ring_elements_ = new int[num_ring_];
			quan_level_memory_ = (char **) malloc(sizeof(void *) * num_ring_);
					
			for(int i = 0; i < num_ring_; i++){
				readBinaryPOD(input, ring_elements_[i]);
				quan_level_memory_[i] =  (char *) malloc(ring_elements_[i] * size_quan_per_element_);
				if(quan_level_memory_[i] == nullptr)
					throw std::runtime_error("Not enough memory");				
				input.read(quan_level_memory_[i], ring_elements_[i] * size_quan_per_element_);
			}
			input.read((char *) trans_quan, 4 * max_elements_);
			
            input.close();

            if(num == 1){   //ipnswplus
			    readBinaryPOD(input2, max_elements_);
			    readBinaryPOD(input2, size_data_per_element_);

                data_level0_memory_ = (char *) malloc(max_elements * size_data_per_element_);
                if (data_level0_memory_ == nullptr)
                    throw std::runtime_error("Not enough memory: loadIndex failed to allocate level0");
                input2.read(data_level0_memory_, max_elements * size_data_per_element_);

                readBinaryPOD(input2, maxM0_);
			    readBinaryPOD(input2, offsetData0_);		    
			    readBinaryPOD(input2, label_offset2);                           
               }
            else{	        //ipnsw		
                size_t temp0 = 0;
                double mult0;
                int maxlel; 
                tableint eenter;
			    readBinaryPOD(input2, temp0);
			    readBinaryPOD(input2, max_elements_);
			    readBinaryPOD(input2, temp0);
			    readBinaryPOD(input2, size_data_per_element_);
			    readBinaryPOD(input2, label_offset2);
			    readBinaryPOD(input2, offsetData0_);
			    readBinaryPOD(input2, maxlel);
			    readBinaryPOD(input2, eenter);
			    readBinaryPOD(input2, temp0);
			
                readBinaryPOD(input2, maxM0_);
			    readBinaryPOD(input2, temp0);
			    readBinaryPOD(input2, mult0);
			    readBinaryPOD(input2, temp0);

                data_level0_memory_ = (char *) malloc(max_elements_ * size_data_per_element_);
                if (data_level0_memory_ == nullptr)
                    throw std::runtime_error("Not enough memory: loadIndex failed to allocate level0");
                input2.read(data_level0_memory_, max_elements_ * size_data_per_element_);
			}

			size_links_level0_ip_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
			
            std::vector<std::mutex>(max_elements).swap(link_list_locks_);

            visited_list_pool_ = new VisitedListPool(1, max_elements);

            ef_ = 10;

            offsetData2_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);

            input2.close();
	    trans_ = new int[max_elements];
            for(int i = 0; i < max_elements; i++){
		trans_[ getExternalLabel(i) ] = i;
	    }
			
            return;
        }

        unsigned short int getListCount(linklistsizeint * ptr) const {
            return *((unsigned short int *)ptr);
        }

        void setListCount(linklistsizeint * ptr, unsigned short int size) const {
            *((unsigned short int*)(ptr))=*((unsigned short int *)&size);
        }

        void connect(const void *data_point, int* result, float*** dist_book, int deg_) {
            tableint ep_id = enterpoint_node_;
		int level = maxlevel_;
									
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;

            dist_t lowerBound;
			dist_t dist;
            dist = fstl2func_((unsigned char*) data_point, (unsigned char*)getQuanByInternalId(ep_id, level), dist_book, quan_size_);
	
			top_candidates.emplace(dist, ep_id);
            lowerBound = dist;
            candidateSet.emplace(-dist, ep_id);

            visited_array[ep_id] = visited_array_tag;

            while (!candidateSet.empty()) {
                std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
                if ((-curr_el_pair.first) > lowerBound) {
                    break;
                }
                candidateSet.pop();

                tableint curNodeNum = curr_el_pair.second;

                std::unique_lock <std::mutex> lock(link_list_locks_[curNodeNum]);

                unsigned int *data;// = (int *)(linkList0_ + curNodeNum * size_links_per_element0_);				
				data = getLinkByInternalId(curNodeNum, level);
				
                size_t size = getListCount((linklistsizeint*)data);
                tableint *datal = (tableint *) (data + 1);

                dist_t dist1;
				char *currObj1;
                for (size_t j = 0; j < size; j++) {
                    tableint candidate_id = *(datal + j);
          
                    if (visited_array[candidate_id] == visited_array_tag) continue;
                    visited_array[candidate_id] = visited_array_tag;
                    
				    currObj1 = getQuanByInternalId(candidate_id, level);
                    dist1 = fstl2func_((unsigned char*)data_point, (unsigned char*)currObj1, dist_book, quan_size_);
						
                    if (top_candidates.size() < ef_construction_ || lowerBound > dist1) {
                        candidateSet.emplace(-dist1, candidate_id);
						
                            top_candidates.emplace(dist1, candidate_id);

                        if (top_candidates.size() > ef_construction_)
                            top_candidates.pop();

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
            visited_list_pool_->releaseVisitedList(vl);
            //return top_candidates;
            while(top_candidates.size() > deg_){
				top_candidates.pop();
			}
			for(int i = deg_ - 1; i >= 0; i--){
				result[i] = top_candidates.top().second;
				top_candidates.pop();
			}
        }		
		
        void addPoint(const void *quan_point, labeltype label, int layer, float norm, float*** dist_book) {
            addPoint2(quan_point, label, layer, norm, dist_book);
        }

        tableint addPoint2(const void *quan_point, labeltype label, int layer, float norm, float*** dist_book) {
            
            tableint cur_c = 0;
            {
                std::unique_lock <std::mutex> lock(cur_element_count_guard_);
                if (cur_element_count >= max_elements_) {
                    throw std::runtime_error("The number of elements exceeds the specified limit");
                };

                cur_c = cur_element_count;
                cur_element_count++;
				trans_quan[cur_c] = label;
            }

            std::unique_lock <std::mutex> lock_el(link_list_locks_[cur_c]);
			int curlevel = layer;
      
            std::unique_lock <std::mutex> templock(global);
            int maxlevelcopy = maxlevel_;
            if (curlevel <= maxlevelcopy)
                templock.unlock();
		
			//int maxlevelcopy = maxlevel_;
            tableint currObj0 = enterpoint_node_;
			tableint currObj1 = enterpoint_node_;
            tableint enterpoint_copy = enterpoint_node_;

            for(int i = 0; i <= layer; i++){
                memset(quan_level_memory_[i] + cur_c * size_quan_per_element_, 0, size_quan_per_element_);
				
				memcpy(getNormByInternalId(cur_c, i), &norm, sizeof(float));
			    memcpy(getQuanByInternalId(cur_c, i), quan_point, quan_size_);

			    memcpy(getNormByInternalId2(cur_c, i), &norm, sizeof(float));
			    memcpy(getQuanByInternalId2(cur_c, i), quan_point, quan_size_);

			}	


            if ((signed)currObj0 != -1) {

                if (curlevel < maxlevelcopy) { // only for outest ring 

                    dist_t curdist = fstl2func_((unsigned char*)quan_point, (unsigned char*)getQuanByInternalId(currObj0, maxlevelcopy), dist_book, quan_size_);
                    for (int level = maxlevelcopy; level > curlevel; level--) {

                        bool changed = true;
                        while (changed) {
                            changed = false;
                            unsigned int *data;
                            std::unique_lock <std::mutex> lock(link_list_locks_[currObj0]);
                            data = getLinkByInternalId(currObj0,level);
                            int size = getListCount(data);

                            tableint *datal = (tableint *) (data + 1);
                            for (int i = 0; i < size; i++) {
                                tableint cand = datal[i];
                                if (cand < 0 || cand > max_elements_)
                                    throw std::runtime_error("cand error");
                                dist_t d = fstl2func_((unsigned char*)quan_point, (unsigned char*)getQuanByInternalId(cand, level), dist_book, quan_size_);
                                if (d < curdist) {
                                    curdist = d;
                                    currObj0 = cand;
                                    changed = true;
                                }
                            }
                        }
                    }
					
					float curnorm = *(getNormByInternalId2(currObj1, maxlevelcopy)) * norm;
                    curdist = fstl2func_((unsigned char*)quan_point, (unsigned char*)getQuanByInternalId2(currObj1, maxlevelcopy), dist_book, quan_size_) * curnorm;
                    for (int level = maxlevelcopy; level > curlevel; level--) {

                        bool changed = true;
                        while (changed) {
                            changed = false;
                            unsigned int *data;
                            std::unique_lock <std::mutex> lock(link_list_locks_[currObj1]);
                            data = getLinkByInternalId2(currObj1,level);
                            int size = getListCount(data);
							

                            tableint *datal = (tableint *) (data + 1);
                            for (int i = 0; i < size; i++) {
                                tableint cand = datal[i];
                                if (cand < 0 || cand > max_elements_)
                                    throw std::runtime_error("cand error");
								
								float tmp_norm = *(getNormByInternalId2(cand, level)) * norm;
                                dist_t d = fstl2func_((unsigned char*)quan_point, (unsigned char*)getQuanByInternalId2(cand, level), dist_book, quan_size_) * tmp_norm;
                                if (d < curdist) {
                                    curdist = d;
                                    currObj1 = cand;
                                    changed = true;
                                }
                            }
                        }
                    }				
                }

                //bool epDeleted = isMarkedDeleted(enterpoint_copy);
                for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--) {
					
                    if (level > maxlevelcopy || level < 0)  // possible?
                        throw std::runtime_error("Level error");

                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = searchBaseLayer(
                            currObj0, quan_point, level, true, norm, dist_book, label);
				
                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> ip_candidates = searchBaseLayer(
                            currObj1, quan_point, level, false, norm, dist_book, label);

                    mutuallyConnectNewElement(quan_point, cur_c, top_candidates, ip_candidates, level, dist_book, label);

                    currObj0 = top_candidates.top().second;
					currObj1 = ip_candidates.top().second;
                }

            } else {
                // Do nothing for the first element
                enterpoint_node_ = 0;
                maxlevel_ = curlevel;
            }

            return cur_c;
        };

        //std::priority_queue<std::pair<dist_t, labeltype >>
		std::vector<Neighbor>
        searchKnn(const void *query_data, size_t K, float** quan_book, float* max_norm, int vecdim,  double* avg_dist) {
            float quan_cal = (float) quan_size_ / vecdim /2;
			
	        int L = ef_;
	        int L_copy = L;
		    VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;
			
	        std::vector<Neighbor> retset(L + 1);
		    std::vector<Neighbor> retset2(L + 1);
			
            int L2 = 0;
            int L_ip = 0;
					   	                                             
            for (unsigned i = 0; i < 1; i++) {
                float *x = (float *)(getNormByInternalId(0, maxlevel_));
                float norm_x = *x;
                x++;
                float dist = quan_dist((unsigned char*) x, quan_book, 0);
                retset[i] = Neighbor(0, dist, true);
				retset2[i] = Neighbor(0, norm_x * dist, true);             
				visited_array[0] = visited_array_tag;
                L2++;
				L_ip++;
            }
           
			std::sort( retset.begin(), retset.begin()+ L2); 
            std::sort( retset2.begin(), retset2.begin()+ L_ip); 			
						
			for (int level = maxlevel_; level >= 0; level--) {			
                               
                if(level < maxlevel_){
                         bool change_ = true; 
                         unsigned cur_obj = retset[0].id;
                         float cur_dist = retset[0].distance;
                         float cur_ip = cur_dist * (* getNormByInternalId(cur_obj, level) );                         

                         while(change_ == true){
                              change_ = false;                            
                             _mm_prefetch(getLinkByInternalId(cur_obj, level), _MM_HINT_T0);
                             unsigned *neighbors = (unsigned *)(getLinkByInternalId(cur_obj, level));
                            unsigned MaxM = *neighbors;

                             neighbors++;
                            for (unsigned m = 0; m < MaxM; ++m)
                                _mm_prefetch( (char *)getNormByInternalId(neighbors[m], level), _MM_HINT_T0);
                            for (unsigned m = 0; m < MaxM; ++m) {
                               unsigned id = neighbors[m];
                               if (visited_array[id] == visited_array_tag) continue;
                               visited_array[id] = visited_array_tag;
							
                               float *data = getNormByInternalId(neighbors[m], level);
                               float norm = *data;
                               data++;
                               float dist = quan_dist( (unsigned char*) data, quan_book, id);
							   dist_calc += quan_cal;
							   
			                   float ip = norm * dist;
                               if(dist < cur_dist) {change_ = true; cur_obj = id; cur_dist = dist; cur_ip = ip;}

			       if (L_ip == L_copy && ip >= retset2[L_copy - 1].distance) continue;
                               Neighbor nn2(id, ip, true);
                               InsertIntoPool(retset2.data(), L_ip, nn2);

                           }

                        }																							
                       
                         
                         if( retset2[0].distance < cur_dist * max_norm[level] )  
                           continue;                     
                        
                         change_ = true; 
                         while(change_ == true){
                              change_ = false;                            
                             _mm_prefetch(getLinkByInternalId2(cur_obj, level), _MM_HINT_T0);
                             unsigned *neighbors = (unsigned *)(getLinkByInternalId2(cur_obj, level));
                            unsigned MaxM = *neighbors;

                             neighbors++;
                            for (unsigned m = 0; m < MaxM; ++m)
                                _mm_prefetch( (char *)getNormByInternalId2(neighbors[m], level), _MM_HINT_T0);
                            for (unsigned m = 0; m < MaxM; ++m) {
                               unsigned id = neighbors[m];
                               if (visited_array[id] == visited_array_tag) continue;
                               visited_array[id] = visited_array_tag;
							
                               float *data = getNormByInternalId2(neighbors[m], level);
                               float norm = *data;
                               data++;
                               float dist = quan_dist( (unsigned char*) data, quan_book, id);
							   dist_calc += quan_cal;
			       float ip = norm * dist;

                               if(ip < cur_ip) {change_ = true;  cur_obj = id; cur_ip = ip;}
			       if (L_ip == L_copy && ip >= retset2[L_copy - 1].distance) continue;
                               Neighbor nn2(id, ip, true);
                               InsertIntoPool(retset2.data(), L_ip, nn2);

                           }

                        }
                         
                        
                  for(int i = 0; i < L_ip; i++)
                        retset2[i].flag = true;																							

	         int   k = 0;
                              
                while (k < L) {
                    int nk = L_ip;

                    if (retset2[k].flag) {
                        retset2[k].flag = false;
                        unsigned n = retset2[k].id;

                        _mm_prefetch(getLinkByInternalId2(n, level), _MM_HINT_T0);
                        unsigned *neighbors = (unsigned *)(getLinkByInternalId2(n, level));
                        unsigned MaxM = *neighbors;
                        neighbors++;
                        for (unsigned m = 0; m < MaxM; ++m)
                            _mm_prefetch(getNormByInternalId2(neighbors[m], level), _MM_HINT_T0);
                        for (unsigned m = 0; m < MaxM; ++m) {
                            unsigned id = neighbors[m];
                            if (visited_array[id] == visited_array_tag) continue;
                            visited_array[id] = visited_array_tag;
                            float *data = getNormByInternalId2(neighbors[m], level);
                            float norm = *data;
                            data++;
                            float dist = quan_dist( (unsigned char*) data, quan_book, id);
							dist_calc += quan_cal;
							float ip = norm * dist;
	                                                   						
							if (L_ip == L_copy && ip >= retset2[L_copy - 1].distance) continue;
                            Neighbor nn2(id, ip, true);
                            int r = InsertIntoPool(retset2.data(), L_ip, nn2);
		                    if(L_ip < L_copy) {L_ip++;}
                            if (r < nk) nk = r;
                        }
                    }
                    if (nk <= k)
                        k = nk;
                    else
                        ++k;
                } 
                continue; 
			
                }

	            int k = 0;    
                         
                while (k <  L) {
                    int nk = L2;

                    if (retset[k].flag) {
                        retset[k].flag = false;
                        unsigned n = retset[k].id;

                        _mm_prefetch(getLinkByInternalId(n, level), _MM_HINT_T0);
                        unsigned *neighbors = (unsigned *)(getLinkByInternalId(n, level));
                        unsigned MaxM = *neighbors;

                        neighbors++;
                        for (unsigned m = 0; m < MaxM; ++m)
                            _mm_prefetch( (char *)getNormByInternalId(neighbors[m], level), _MM_HINT_T0);
                        for (unsigned m = 0; m < MaxM; ++m) {
                            unsigned id = neighbors[m];
                            if (visited_array[id] == visited_array_tag) continue;
                            visited_array[id] = visited_array_tag;
							
                            float *data = getNormByInternalId(neighbors[m], level);
                            float norm = *data;
                            data++;
                            float dist = quan_dist( (unsigned char*) data, quan_book, id);
							dist_calc += quan_cal;    
							float ip = norm * dist;
					
                                         	
                            if (L2 == L_copy && dist >= retset[L_copy - 1].distance) continue;
                            Neighbor nn(id, dist, true);
                            int r = InsertIntoPool(retset.data(), L2, nn);
		                    if(L2 < L_copy) {L2++;}
							
							if (L_ip == L_copy && ip >= retset2[L_copy - 1].distance) continue;
                            Neighbor nn2(id, ip, true);
                            InsertIntoPool(retset2.data(), L_ip, nn2);
		                    if(L_ip < L_copy) {L_ip++;}

                            if (r < nk) nk = r;
                        }
                    }
                    if (nk <= k)
                        k = nk;
                    else
                        ++k;
                }
                
	            k = 0;
                              
                while (k < L) {
                    int nk = L_ip;

                    if (retset2[k].flag) {
                        retset2[k].flag = false;
                        unsigned n = retset2[k].id;

                        _mm_prefetch(getLinkByInternalId2(n, level), _MM_HINT_T0);
                        unsigned *neighbors = (unsigned *)(getLinkByInternalId2(n, level));
                        unsigned MaxM = *neighbors;
                        neighbors++;
                        for (unsigned m = 0; m < MaxM; ++m)
                            _mm_prefetch(getNormByInternalId2(neighbors[m], level), _MM_HINT_T0);
                        for (unsigned m = 0; m < MaxM; ++m) {
                            unsigned id = neighbors[m];
                            if (visited_array[id] == visited_array_tag) continue;
                            visited_array[id] = visited_array_tag;
                            float *data = getNormByInternalId2(neighbors[m], level);
                            float norm = *data;
                            data++;
                            float dist = quan_dist( (unsigned char*) data, quan_book, id);
							dist_calc += quan_cal;
							float ip = norm * dist;
	                                                   						
							if (L_ip == L_copy && ip >= retset2[L_copy - 1].distance) continue;
                            Neighbor nn2(id, ip, true);
                            int r = InsertIntoPool(retset2.data(), L_ip, nn2);
		                    if(L_ip < L_copy) {L_ip++;}
                            if (r < nk) nk = r;
                        }
                    }
                    if (nk <= k)
                        k = nk;
                    else
                        ++k;
                } 
                          					
			}
               			
			visited_list_pool_->releaseVisitedList(vl);		

          //--------------test----------------------
/*                   
                for(int i = 0; i < L_ip; i++){
                    retset2[i].id = trans_quan[retset2[i].id];
                }

	    	
                for(int i = 0; i < L_ip; i++){
                     retset2[i].id = trans_quan[retset2[i].id];
                     int tmp_id = trans_[retset2[i].id];
                    char *currObj1 = (getDataByInternalId(tmp_id));
                    retset2[i].distance = fstdistfunc_(query_data, currObj1, dist_func_param_);
                }
	       std::sort( retset2.begin(), retset2.begin()+ L2); 
*/	  	
        //-----------------------------------------------

        std::vector<std::pair<dist_t, tableint>> cos_res;
        
        for(int y = 0; y < L_ip; y++){
            int tmp_id = trans_[ trans_quan[retset2[y].id] ];  //ext->int

            char *currObj1 = (getDataByInternalId(tmp_id));
            dist_t dist = fstdistfunc_(query_data, currObj1, dist_func_param_);
            cos_res.push_back(std::make_pair(dist, tmp_id));			
        }		
		
		  std::priority_queue<std::pair<dist_t, tableint>> mips_topResults = searchBaseLayerST_inner_product(cos_res, query_data, ef_);

        std::priority_queue< std::pair< dist_t, labeltype >> results;
        while (mips_topResults.size() > K) {
            mips_topResults.pop();
        }
        

        for (int i = K - 1; i >= 0; i--) {
            std::pair<dist_t, tableint> rez = mips_topResults.top();
			retset2[i].id = getExternalLabel(rez.second);
            mips_topResults.pop();
        }

	    *avg_dist = dist_calc;
		return retset2;
       }

         void rotation(int qsize, float* R, int D, float** query, float** query2, int M, int L, float*** qip, int d, float*** vec){
             size_t a = D;
             size_t b = d;          
             for(int i = 0; i < qsize; i++){
                for(int j = 0; j < D; j++){
                   query2[i][j] = -1 * fstdistfunc_(&R[j*D], query[i], (void *) &a); 
                }

             }
         } 
		 
         float query_sample(float* query, float** data, int d, int n, float* norm){
		    float max_sum = -1;
			size_t a = d;
            for(int i = 0; i < n; i++){
                float val = -1 * fstdistfunc_(query, data[i], (void *) &a); 
				if(i == 0)
				    max_sum = val;
				else if(val > max_sum)
					max_sum = val;
				
				if(max_sum >= norm[i])
					break;
            }            
			 
			return max_sum; 
			 
         } 		 
		 
		
       };

}
