#include <iostream>
#include <fstream>
#include <queue>
#include <chrono>
#include "hnswlib/hnswlib.h"
//#include <chrono>

#include <unordered_set>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include"opencv2/imgproc/imgproc.hpp"

#ifdef _MSC_VER
#include <intrin.h>
#include <stdexcept>

#define  __builtin_popcount(t) __popcnt(t)
#else
#include <x86intrin.h>
#endif
#define USE_AVX
#if defined(__GNUC__)
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#else
#define PORTABLE_ALIGN32 __declspec(align(32))
#endif

using namespace std;
using namespace hnswlib;
using namespace cv;
class StopW {
    std::chrono::steady_clock::time_point time_begin;
public:
    StopW() {
        time_begin = std::chrono::steady_clock::now();
    }

    float getElapsedTimeMicro() {
        std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
        return (std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count());
    }

    void reset() {
        time_begin = std::chrono::steady_clock::now();
    }

};

// -----------------------------------------------------------------------------
float uniform(						// r.v. from Uniform(min, max)
	float min,							// min value
	float max)							// max value
{
	int   num  = rand();
	float base = (float) RAND_MAX - 1.0F;
	float frac = ((float) num) / base;

	return (max - min) * frac + min;
}

// -----------------------------------------------------------------------------
//	Given a mean and a standard deviation, gaussian generates a normally 
//		distributed random number.
//
//	Algorithm:  Polar Method, p.104, Knuth, vol. 2
// -----------------------------------------------------------------------------
float gaussian(						// r.v. from Gaussian(mean, sigma)
	float mean,							// mean value
	float sigma)						// std value
{
	float v1 = -1.0f;
    float v2 = -1.0f;
	float s  = -1.0f;
	float x  = -1.0f;

	do {
		v1 = 2.0F * uniform(0.0F, 1.0F) - 1.0F;
		v2 = 2.0F * uniform(0.0F, 1.0F) - 1.0F;
		s = v1 * v1 + v2 * v2;
	} while (s >= 1.0F);
	x = v1 * sqrt (-2.0F * log (s) / s);

	x = x * sigma + mean; 			// x is distributed from N(0, 1)
	return x;
}


void Mul_W(CvMat* A, float* w, CvMat* B){
	float temp;
	for (int j = 0; j < A->cols; j++) {
        for (int i = 0; i < A->rows; i++) {
            temp = cvmGet (A, i, j);
            cvmSet (B, i, j, temp * w[j]);
		}	
    }
		
}

void Normalize(CvMat* M){
	float temp, sum;
	for (int j = 0; j < M->cols; j++) {
		sum = 0;
        for (int i = 0; i < M->rows; i++) {
            temp = cvmGet (M, i, j);
			sum += temp*temp;
        }
		sum = sqrt(sum);
        for (int i = 0; i < M->rows; i++) {
            temp = cvmGet (M, i, j);
            cvmSet (M, i, j, temp/sum);
		}	
    }
		
}


    float compare2(const float* a, const float* b, unsigned size) {
      float result = 0;
     //  printf("check1\n");
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
       //   printf("check2\n");
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
     // printf("check3\n");
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

struct elem{
	int id;
	float ip;
};

void rotation_(
    float* R,
    int n,
	int D,
	float** data,
    int* Layer,
    elem* norm	
)
{
	float sum = 0;
	float** data2 = new float* [n];
	for(int i = 0; i < n; i++){
		data2[i] = new float[D];
	}
	for(int i = 0; i < n; i++){
		for(int j = 0; j < D; j++){
			data2[i][j] = data[i][j];
		}
	}
	
	for(int i = 0; i < n; i++){
          if(Layer[i] < 1)
		  break;
		for(int j = 0; j < D; j++){
			data[ norm[i].id ][j] = 0;
            data[ norm[i].id ][j] = compare2( &(R[j*D]), data2[ norm[i].id ], D);
		}
	}
	
	for(int i = 0; i < n; i++){
		delete[] data2[i];
	}
	delete[] data2; data2 = NULL;
}


/*
* Author:  David Robert Nadeau
* Site:    http://NadeauSoftware.com/
* License: Creative Commons Attribution 3.0 Unported License
*          http://creativecommons.org/licenses/by/3.0/deed.en_US
*/

#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))

#include <unistd.h>
#include <sys/resource.h>

#if defined(__APPLE__) && defined(__MACH__)
#include <mach/mach.h>

#elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
#include <fcntl.h>
#include <procfs.h>

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)

#endif

#else
#error "Cannot define getPeakRSS( ) or getCurrentRSS( ) for an unknown OS."
#endif


/**
* Returns the peak (maximum so far) resident set size (physical
* memory use) measured in bytes, or zero if the value cannot be
* determined on this OS.
*/
static size_t getPeakRSS() {
#if defined(_WIN32)
    /* Windows -------------------------------------------------- */
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
    return (size_t)info.PeakWorkingSetSize;

#elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
    /* AIX and Solaris ------------------------------------------ */
    struct psinfo psinfo;
    int fd = -1;
    if ((fd = open("/proc/self/psinfo", O_RDONLY)) == -1)
        return (size_t)0L;      /* Can't open? */
    if (read(fd, &psinfo, sizeof(psinfo)) != sizeof(psinfo))
    {
        close(fd);
        return (size_t)0L;      /* Can't read? */
    }
    close(fd);
    return (size_t)(psinfo.pr_rssize * 1024L);

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))
    /* BSD, Linux, and OSX -------------------------------------- */
    struct rusage rusage;
    getrusage(RUSAGE_SELF, &rusage);
#if defined(__APPLE__) && defined(__MACH__)
    return (size_t)rusage.ru_maxrss;
#else
    return (size_t) (rusage.ru_maxrss * 1024L);
#endif

#else
    /* Unknown OS ----------------------------------------------- */
    return (size_t)0L;          /* Unsupported. */
#endif
}


/**
* Returns the current resident set size (physical memory use) measured
* in bytes, or zero if the value cannot be determined on this OS.
*/
static size_t getCurrentRSS() {
#if defined(_WIN32)
    /* Windows -------------------------------------------------- */
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
    return (size_t)info.WorkingSetSize;

#elif defined(__APPLE__) && defined(__MACH__)
    /* OSX ------------------------------------------------------ */
    struct mach_task_basic_info info;
    mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
        (task_info_t)&info, &infoCount) != KERN_SUCCESS)
        return (size_t)0L;      /* Can't access? */
    return (size_t)info.resident_size;

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
    /* Linux ---------------------------------------------------- */
    long rss = 0L;
    FILE *fp = NULL;
    if ((fp = fopen("/proc/self/statm", "r")) == NULL)
        return (size_t) 0L;      /* Can't open? */
    if (fscanf(fp, "%*s%ld", &rss) != 1) {
        fclose(fp);
        return (size_t) 0L;      /* Can't read? */
    }
    fclose(fp);
    return (size_t) rss * (size_t) sysconf(_SC_PAGESIZE);

#else
    /* AIX, BSD, Solaris, and Unknown OS ------------------------ */
    return (size_t)0L;          /* Unsupported. */
#endif
}


static void
get_gt(unsigned int *massQA, float *massQ, size_t vecsize, size_t qsize, InnerProductSpace &l2space,
       size_t vecdim, vector<std::priority_queue<std::pair<float, labeltype >>> &answers, size_t K) {


    (vector<std::priority_queue<std::pair<float, labeltype >>>(qsize)).swap(answers);
    //DISTFUNC<float> fstdistfunc_ = l2space.get_dist_func();
   // cout << qsize << "\n";
    for (int i = 0; i < qsize; i++) {
        for (int j = 0; j < K; j++) {
            answers[i].emplace(0.0f, massQA[100 * i + j]);
        }
    }
}

static float
test_approx(float *massQ, size_t vecsize, size_t qsize, HierarchicalNSW<float> &appr_alg, size_t vecdim,
            vector<std::priority_queue<std::pair<float, labeltype >>> &answers, size_t K, float*** qip, float* max_norm,
			 double* avg_dist) {
    size_t correct = 0;
    size_t total = 0;
    //uncomment to test in parallel mode:
    //#pragma omp parallel for
    //--------test----------------------
    // printf("massQ[300] = %.5f, [301] = %.5f, dim = %d, qsize = %d, K =%d\n",massQ[300],massQ[301],vecdim,qsize,K);
    //-------------------------------
    for (int i = 0; i < qsize; i++) {           
	     std::vector<Neighbor> result = appr_alg.searchKnn(massQ + vecdim * i, K, qip[i], max_norm, vecdim, avg_dist);
    	  
        std::priority_queue<std::pair<float, labeltype >> gt(answers[i]);
        unordered_set<labeltype> g;
        total += gt.size();

        while (gt.size()) {

            g.insert(gt.top().second);
            gt.pop();
        }
		for(int i = 0; i < K; i++){
			if (g.find(result[i].id) != g.end()) {
                correct++;
            }
		}   
    }
    return 1.0f * correct / total;  
}

static void
test_vs_recall(float *massQ, size_t vecsize, size_t qsize, HierarchicalNSW<float> &appr_alg, size_t vecdim,
               vector<std::priority_queue<std::pair<float, labeltype >>> &answers, size_t K, float*** qip, float* max_norm) {
    vector<size_t> efs;// = { 10,10,10,10,10 };
    
   /*
    for (int i = K; i < 100; i += 100 ) {
        efs.push_back(i);
    }
    for (int i = 10; i < 100; i += 10) {
        efs.push_back(i);
    }
    */
    for (int i = K; i <= 3000; i += 100) {
        efs.push_back(i);
    }

    for (size_t ef : efs) {
        appr_alg.setEf(ef);
        appr_alg.dist_calc = 0;
        StopW stopw = StopW();
	
		double avg_dist = 0;
              
        float recall = test_approx(massQ, vecsize, qsize, appr_alg, vecdim, answers, K, qip, max_norm,  &avg_dist);
        float time_us_per_query = stopw.getElapsedTimeMicro() / qsize;
		
		avg_dist = avg_dist / qsize;

        cout << ef << "\t" << recall << "\t" << time_us_per_query << " us" << "\n";
        if (recall > 1.0) {
            cout << recall << "\t" << time_us_per_query << " us\n";
            break;
        }
    }
}

inline bool exists_test(const std::string &name) {
    ifstream f(name.c_str());
    return f.good();
}

/*
struct elem{
	int id;
	float ip;
};
*/

int Elemcomp(const void*a,const void*b)
{  
  elem x1 = *((elem*) b);
  elem x2 = *((elem*) a);

  if(x1.ip > x2 .ip)
       return 1;
  else {return -1;}
 // return x1.ip - x2.ip;
 // return (*(elem*)b).ip - (*(elem*)a).ip;
}


int Intcomp(const void*a,const void*b)
{

  return  *(int*)a - *(int*)b;
}



void sift_test1B(
  char* data_set,
  int vecsize_,
  int vecdim_,
  int qsize_,
  char* mode_,
  char* path_info,
  char* path_index_,
  char* path_index2_,
  char* path_gt,
  int topk
) {
	
	const float FLOATZERO = 1e-6F;
	//-------------------------specify--------------------------------
    size_t vecsize = vecsize_;
    size_t vecdim = vecdim_;
    size_t qsize = qsize_;
    char *path_data = data_set;    	
	
	int efConstruction = 200;
	int M = 32;
    int maxk = 100;
	int rat_ = 4;
		
    int d;
    if( vecdim % M == 0)
       d = vecdim / M;
    else{
       d = vecdim / M + 1;
    }
         
    char path_index[1024];
    char path_index2[1024];
	//char path_index2[1024];
        //  char path_gt[1024];
         
    sprintf(path_index, path_index_);
    sprintf(path_index2, path_index2_);
	//sprintf(path_index2, "outgraph.hnsw");

	int n2 = 20000;
	int M2 = 16;   //the number of codebooks 
	int MIN_NUM = (int) (0.01 * vecsize);  // the minimum size of the subset in the first ring
	int qn = 100; //the number of query samples
	int K = topk; // top-K
	
	int L =256;
	int ROUND1 = 10;
    int ROUND2 = 20;
	int ROUND3 = 10;
	//----------------------------------------
	
        int ind1, ind2, sub;
        sub = M * d - vecdim;

    float max_temp;
	float min_temp;
	int min_vec;
	InnerProductSpace l2space(vecdim);
	
	float*** vec = new float** [M];
	for(int i = 0; i < M; i++){
		vec[i] = new float* [L];
	} 
    for(int i = 0; i < M; i++){
		for(int l = 0; l < L; l++){ 
	        vec[i][l] = new float[d];     
		}
	}
		
	int D = vecdim;
    float* R = new float[M * d * M * d];
	int num_ring = 0;
         
    int temp_n = vecsize;
    while(temp_n >= MIN_NUM){
        temp_n = temp_n / rat_;
        num_ring++;
    }   
 
	int* r_num = new int[num_ring];
      	     
    elem* norm = new elem[vecsize]; 

	HierarchicalNSW<float> *appr_alg;
    if (exists_test(path_index)) {
        cout << "Loading index from " << path_index << ":\n";
		if( std::string(mode_) == "ipnsw")
            appr_alg = new HierarchicalNSW<float>(&l2space, path_index, 0, path_index2, false);
		else if (std::string(mode_) == "ipnswplus")
			appr_alg = new HierarchicalNSW<float>(&l2space,  path_index, 1, path_index2, false);
		else{
			printf("incorrect inputs\n");
			exit(0);
		}
       
    } else {
	int n = vecsize;
		
	double temp;
	float temp2;
	float lambda_;
	
	elem* norm_t = new elem[n2];
	float* diag_ = new float [M];
	float* diag0_ = new float [M];
	
	float** data_org = new float*[n];
     
	ifstream inputD(path_data, ios::binary);
	for (int i = 0; i < n; i++) data_org[i] = new float[D];
	
    for (int i = 0; i < n; i++) {
        int t;
        inputD.read((char *) &t, 4);
        inputD.read((char *) (data_org[i]), 4 * D);
    }
		
	for (int i = 0; i < n; i++){
		norm[i].id = i;
		norm[i].ip = 0;
		for(int j = 0; j < D; j++){
		    norm[i].ip += data_org[i][j] * data_org[i][j];
		}
		norm[i].ip = sqrt(norm[i].ip);
	}

	qsort(norm, n, sizeof(elem), Elemcomp);  //in descending order
	
	float* norm_temp = new float[n];
	for(int i = 0; i < n; i++)
		norm_temp[i] = norm[i].ip;
        
	float* r_val = new float[num_ring];
	int old_num;
	
	int pow_2 = 1;
	for(int i = 0; i < num_ring; i++){
		pow_2 = rat_ * pow_2;
		int pos2 = n / pow_2;
		r_val[i] = norm[pos2].ip;
		r_num[i] = pos2;
	}

	float** data = new float*[n];
	for (int i = 0; i < n; i++) data[i] = new float[M*d];	
			
	for (int i = 0; i < n; i++){  //adjust the diemsion and resort the data
		ind1 = 0; ind2 = 0;
		for(int j = 0; j < M; j++){
			for(int l = 0; l < d-1; l++){
				data[i][ind1] = data_org[norm[i].id][ind2];
			    ind1++; ind2++;
			}
			if(sub != ind1 - ind2){
			    data[i][ind1] = 0;
			    ind1 ++;
			}
            else{
				data[i][ind1] = data_org[norm[i].id][ind2];
			    ind1++; ind2++;	
			}			
		}
		if(ind1 != M * d || ind2 != D) {printf("error\n"); exit(0);}
	}
      
	D = M * d;  //update dimension
        
	float** q_sample = new float* [qn];
	for(int i = 0; i < qn; i++){
		q_sample[i] = new float[D];
	}

    float q_sum;
    for(int i = 0; i < qn; i++){  // normalize vec
        q_sum = 0;
        for(int j = 0; j < D; j++){  		
            q_sample[i][j] = gaussian(0.0f, 1.0f);
            q_sum += q_sample[i][j] * q_sample[i][j]; 
		}
              
        for(int j = 0; j < D; j++){
            q_sample[i][j] = q_sample[i][j]/sqrt(q_sum);
        }
    }	
	
	float sum = 0;
	elem* norm2 = new elem[n];
	elem* q_norm = new elem[qn];
	HierarchicalNSW<float> * appr_alg0 = new HierarchicalNSW<float>(&l2space, vecsize, NULL, -1, M2, efConstruction);
	
	for(int i = 0; i < qn; i++){
		q_norm[i].id = i;
	    q_norm[i].ip = appr_alg0 -> query_sample(q_sample[i], data, D, n, norm_temp);
	}
	qsort(q_norm, qn, sizeof(elem), Elemcomp); //in decreasing order	

	D = vecdim;
	for (int i = 0; i < n; i++){  //adjust the diemsion and resort the data
		ind1 = 0; ind2 = 0;
		for(int j = 0; j < M; j++){
			for(int l = 0; l < d-1; l++){
				data[i][ind1] = data_org[i][ind2];
			    ind1++; ind2++;
			}
			if(sub != ind1 - ind2){
			    data[i][ind1] = 0;
			    ind1 ++;
			}
            else{
				data[i][ind1] = data_org[i][ind2];
			    ind1++; ind2++;	
			}			
		}
		if(ind1 != M * d || ind2 != D) {printf("error\n"); exit(0);}
	}  
	D = M * d;  //update dimension
	
	//----------------------------------
	float** train = new float*[n2];
	for(int i = 0; i < n2; i++) train[i] = new float[D];
	int inv = (n - n2/2) / (n2/2);	
	int ind = n2/2 - inv;
	
	for(int i = 0; i < n2/2; i++)
		for(int j = 0; j < D; j++)
			train[i][j] = data[ norm[i].id ][j];
	
	for(int i = n2/2; i < n2; i++){
		ind += inv;
		for(int j = 0; j < D; j++)
			train[i][j] = data[ norm[ind].id ][j];
	}
	
	for(int i = 0; i < n2; i++){ 
		norm_t[i].id = i;
		norm_t[i].ip = 0;
		for(int j = 0; j <D; j++){
			norm_t[i].ip += train[i][j] * train[i][j]; 
		}
		norm_t[i].ip = sqrt(norm_t[i].ip);		
	}	
	qsort(norm_t, n2, sizeof(elem), Elemcomp);
	
	StopW stopw1 = StopW();
	int* pvec = new int[n2];
	float* dvec = new float[n2];
	float* w = new float[n2];
	float sum_tol = 0; 
	
	for(int j = 0; j < n2; j++){ // compute weights(before normalization)
		temp = 0;
		for(int l = 0; l < D; l++){
			temp += train[j][l] * train[j][l];
		}
		w[j] = sqrt(temp);
	}
	
	CvMat* M_D = cvCreateMat(D, n2, CV_32FC1);
	//CvMat* M_W = cvCreateMat(n2, n2, CV_32FC1);
	CvMat* M_X = cvCreateMat(D, n2, CV_32FC1);
	CvMat* M_XW = cvCreateMat(D, n2, CV_32FC1);
	CvMat* M_Y = cvCreateMat(D, n2, CV_32FC1);
    CvMat* M_YW = cvCreateMat(D, n2, CV_32FC1);
	CvMat* M_YWT = cvCreateMat(n2, D, CV_32FC1);
	CvMat* M_R = cvCreateMat(D, D, CV_32FC1);
	CvMat* M_RC = cvCreateMat(D, D, CV_32FC1);
	CvMat* M_RX = cvCreateMat(D, n2, CV_32FC1);
	CvMat* M_U = cvCreateMat(D, D, CV_32FC1);
	CvMat* M_UY = cvCreateMat(D, n2, CV_32FC1);

	CvMat* ABt   = cvCreateMat(D, D, CV_32FC1);
	CvMat* ABt_D = cvCreateMat(D, D, CV_32FC1);
	CvMat* ABt_U = cvCreateMat(D, D, CV_32FC1);
	CvMat* ABt_VT = cvCreateMat(D, D, CV_32FC1);
	CvMat* M_RT = cvCreateMat(D,D,CV_32FC1);

    for (int i = 0; i < M_U->rows; i++) {
        for (int j = 0; j < M_U->cols; j++) {
			if(i != j)
			{cvmSet (M_U, i, j, 0);}
        }
    }
 
    bool** zero_  = new bool* [M];
	for(int i = 0; i < M; i++)
		zero_[i] = new bool[n2];
	
	for(int i = 0; i < M; i++){
		for(int j = 0; j < n2; j++){
			zero_[i][j] = false;
		}
	}
   
    for(int i = 0; i < n2; i++){
        temp = 0;
        for(int j = 0; j < D; j++){
            temp += train[i][j] * train[i][j];
        }       
        temp = sqrt(temp);
            for(int j = 0; j < D; j++)
                train[i][j] = train[i][j] / temp; 
    }

    for (int i = 0; i < M_X->rows; i++) {
        for (int j = 0; j < M_X->cols; j++) {
            cvmSet (M_X, i, j, train[j][i]);
        }
    }

    //-------------------PCA--------------------------
	CvMat* M_X2 = cvCreateMat(D, n2, CV_32FC1);
    CvMat* pMean = cvCreateMat(1, D, CV_32FC1);
    CvMat* pEigVals = cvCreateMat(1, D, CV_32FC1);
    CvMat* pEigVecs = cvCreateMat(D, D, CV_32FC1);
    cvCalcPCA(M_X, pMean, pEigVals, pEigVecs, CV_PCA_DATA_AS_COL);
    CvMat* PCA_R = cvCreateMat(D, D, CV_32FC1);

    int* ord = new int[M];
    int* ord2 = new int[D];
    elem* prod = new elem[M];
    float ssum;

    for(int i = 0; i < D; i++){
        if(i < M){
            prod[i].ip = cvmGet(pEigVals,0,i);
            prod[i].id = i;
            ord[i] = 1;
            ord2[i] = i * d;
        }  
		
        if(i >= M){
            ssum = cvmGet(pEigVals,0,i);
            qsort(prod, M, sizeof(elem), Elemcomp);

            for(int j = 0; j < M; j++){
                if(ord[prod[j].id] < d ){
                    ord2[i] = prod[j].id * d + ord[prod[j].id];
                    ord[prod[j].id]++;
                    prod[j].ip *= ssum;
                    break;
                }
            }
        }
    }
	
    float* pca_arr = new float[D];
    for(int i = 0; i < D; i++){
        for(int j = 0; j < D; j++){
            pca_arr[j] = cvmGet(pEigVecs,i,j);  
        } 
        ssum = 0;
        for(int j = 0; j < D; j++){
            ssum += pca_arr[j] * pca_arr[j];
        }
        for(int j = 0; j < D; j++){
            cvmSet(PCA_R, ord2[i], j, pca_arr[j]/sqrt(ssum));
        }
    }
 
    cvMatMul( PCA_R, M_X, M_X2);
    for(int i = 0; i < n2; i++){
        for(int j = 0; j < D; j++){
            train[i][j] = cvmGet(M_X2,j,i);
        }
    }

    for(int i = 0; i < M_X->rows; i++) {
        for (int j = 0; j < M_X->cols; j++) {
            cvmSet (M_X, i, j, train[j][i]);
        }
    }
     
	int rad_ = 1;
    int fi_ = 0;
    int* rad_arr = new int[L];
    bool flag_ = false;
    for(int i = 0; i < M; i++){
        fi_ = 0;
    for(int l = 0; l < L; l++){                             
        srand(rad_);
        rad_++;
        int ind_ = rand()% 10000;
        for(int s = 0; s < d; s++){
		    vec[i][l][s] = train[ind_][i*d+s];
		}
        flag_ = false;
            for(int y = 0; y < fi_; y++){ 
                if(ind_ == rad_arr[fi_]){
                    l--; 
					flag_ = true; 
					break;
                } 
            }
            if(flag_ == false){
                rad_arr[fi_] = ind_; 
				fi_++; 	
            }
		}
	}	

	for(int i = 0; i < M; i++){
		for(int j = 0; j < n2; j++){ //normalize train samples;
			temp = 0;
			for(int x = 0; x < d; x++){
			    temp += train[j][i*d+x] * train[j][i*d+x]; 
			}
			
            temp = sqrt(temp);
			if(fabs(temp) <= FLOATZERO){
				zero_[i][j] = true;
				for(int x = 0; x < d; x++){
			        train[j][i*d+x] = 0; 
			    }
			}
			else{
			    for(int x = 0; x < d; x++){
			        train[j][i*d+x] = train[j][i*d+x] / temp; 
			    }
			}	
		}	
	}

	float sqrt_M = sqrt(M);
	float sqrt_diag_;
	for(int i = 0; i < M; i++) diag_[i] = 1 / float(M);
	for(int i = 0; i < M; i++) diag0_[i] = 1 / float(M);	
	double* err = new double[n2];
    double err_sum = 0;
    float diag2_;        
		
    for(int k1 = 0; k1 < ROUND1; k1++){
	//    printf("k1=%d\n",k1);
	
        err_sum = 0;
		float quan, err1, err2;
		cvMatMul(M_U, M_Y, M_UY);
		for (int i = 0; i < M_RX->cols; i++) {
		    quan = 0;
            for (int j = 0; j < M_RX->rows; j++) {
			    err1 = cvmGet (M_RX, j, i);
			    err2 = cvmGet (M_Y, j, i);
			    quan += (err1-err2) * (err1-err2);
            }
		    err[i] = quan;
                    err_sum += err[i];
                    
        }
        
	    for(int i = 0; i < M; i++){  //spherical-K-means
		    sqrt_diag_ = sqrt(diag_[i]);
		    for(int k2 = 0; k2 < ROUND2; k2++){
		        for(int j = 0; j < n2; j++){  //assignment of train vectors			

			        for(int l = 0; l < L; l++){
				        temp = 0;
                        for(int x = 0; x < d; x++){
			                temp += train[j][i*d+x] * vec[i][l][x];
                        }
                        if( l == 0) {max_temp = temp; min_vec = 0;}
                        else if(temp > max_temp) {max_temp = temp; min_vec = l;}				
		            }
                    pvec[j] = min_vec;			
		        }
                    	
		        for(int j = 0; j < n2; j++){
			        for(int x = 0; x < d; x++){
			            vec[i][ pvec[j] ][x] = 0; 
			        }	
		        }
                    
		        for(int j = 0; j < n2; j++){  // compute new vectors
			        for(int x = 0; x < d; x++){
			            vec[i][ pvec[j] ][x] += w[j] * w[j] * train[j][i*d+x]; 
			        }	
		        }

		        for(int l = 0; l < L; l++){  // normalization of vec
			        temp = 0;
			        for(int x = 0; x < d; x++){
			            temp += vec[i][l][x] * vec[i][l][x]; 
			        }
			        temp = sqrt(temp);
			        if(fabs(temp) > FLOATZERO){                
			            for(int x = 0; x < d; x++){
			                vec[i][l][x] = vec[i][l][x] * sqrt_diag_ / temp; 
			            }					
		            }
			        else{
			            for(int x = 0; x < d; x++){
			                vec[i][l][x] = gaussian(0.0f, 1.0f);	
			            }
			            temp = 0;
			            for(int x = 0; x < d; x++){
			                temp += vec[i][l][x] * vec[i][l][x]; 
			            }
			
                        temp = sqrt(temp);
			            for(int x = 0; x < d; x++){
			                vec[i][l][x] = vec[i][l][x] * sqrt_diag_/ temp; 
			            }	
		            }	
		        }			
		    }
		
		    for(int j = 0; j < n2; j++){
                for(int x = 0; x < d; x++){
				    cvmSet (M_Y, i*d+x, j, vec[i][pvec[j]][x]);
			    }
		    }	
	    }  

        float test_sum = 0;
        for(int i = 0; i < M; i++){
            for(int j = 0; j < d; j++){
                test_sum += vec[i][0][j] * vec[i][0][j];   
            }
        }
                   
        err_sum = 0;
		cvMatMul(M_U, M_Y, M_UY);
		for (int i = 0; i < M_RX->cols; i++) {
		    quan = 0;
            for (int j = 0; j < M_RX->rows; j++) {
			    err1 = cvmGet (M_RX, j, i);
			    err2 = cvmGet (M_Y, j, i);
			    quan += (err1-err2) * (err1-err2);
            }
		    err[i] = quan;
            err_sum += err[i];                    
        }
           
        if(k1 == ROUND1 - 1) 
	        break;
	
	    if(k1 == 0){
	        Mul_W(M_X, w, M_XW); //XW=X*W
	    }
	    else{Mul_W (M_RX, w, M_XW);}

	    Mul_W(M_Y, w, M_YW); //YW=Y*W
	
	    cvTranspose(M_YW, M_YWT); // transpose
         
	    cvMatMul(M_XW, M_YWT, ABt);

	    cvSVD(ABt, ABt_D, ABt_U, ABt_VT, CV_SVD_V_T); //SVD
	    cvMatMul( ABt_U, ABt_VT, M_R );
	    cvTranspose(M_R, M_RT) ;
    
	    if(k1 == 0){
		    for (int i = 0; i < M_RC->rows; i++) {
                for (int j = 0; j < M_RC->cols; j++) {
				    if(i == j)
                        cvmSet (M_RC, i, j, 1);
			        else{
				        cvmSet (M_RC, i, j, 0);	
				    }
                }
            }	
	    }
        cvMatMul( M_RT, M_RC, M_RC );
	    cvMatMul( M_RC, M_X, M_RX );
        Mul_W(M_RX, w, M_XW);

        err_sum = 0;
		cvMatMul(M_U, M_Y, M_UY);
		for (int i = 0; i < M_RX->cols; i++) {
		    quan = 0;
            for (int j = 0; j < M_RX->rows; j++) {
			    err1 = cvmGet (M_RX, j, i);
			    err2 = cvmGet (M_Y, j, i);
			    quan += (err1-err2) * (err1-err2);
            }
		    err[i] = quan;
                    err_sum += err[i];
                    
        }
	    //-----------adjust norms in different subspacess-----------
	    cvMatMul(M_XW, M_YWT, ABt);
	
	    temp  = 0;
	    lambda_ = 0;
	    for (int i = 0; i < M; i++) {
		    temp2 = 0;
            for (int j = 0; j < d; j++){
			    temp2 += cvmGet(ABt, i*d+j, i*d+j);
		    }
            temp += temp2 * temp2 / diag_[i];		
        }
   
	    for (int i = 0; i < M; i++) {
		    temp2 = 0;
            for (int j = 0; j < d; j++){
			    temp2 += cvmGet(ABt, i*d+j, i*d+j);
		    }
		    diag0_[i] = temp2 / sqrt(temp) / diag_[i];
            diag_[i] = temp2 * temp2 / temp / diag_[i];		
        }	
		
	    for (int i = 0; i < M; i++) {
            for (int j = 0; j < d; j++){
			    cvmSet(M_U, i*d+j, i*d+j, sqrt(diag0_[i]));
		    }		
        }
   
        err_sum = 0;
		cvMatMul(M_U, M_Y, M_UY);
		for (int i = 0; i < M_RX->cols; i++) {
		    quan = 0;
            for (int j = 0; j < M_RX->rows; j++) {
			    err1 = cvmGet (M_RX, j, i);
			    err2 = cvmGet (M_UY, j, i);
			    quan += (err1-err2) * (err1-err2);
            }
		    err[i] = quan;
            err_sum += err[i];                   
        }

	    cvMatMul(M_U, M_Y, M_UY);
				
	    for(int i = 0; i < n2; i++){
		    for(int j = 0; j < D; j++){
			    train[i][j] = cvmGet (M_RX, j, i);
		    }
	    }
	
	    for(int i = 0; i < M; i++){
		    for(int j = 0; j < n2; j++){ //normalize train samples;
			    temp = 0;
			    for(int x = 0; x < d; x++){
			        temp += train[j][i*d+x] * train[j][i*d+x]; 
			    }
            
			    temp = sqrt(temp);
			    if(fabs(temp) <= FLOATZERO){
				    for(int x = 0; x < d; x++){
			            train[j][i*d+x] = 0; 
			        }
			    }
			    else{
			        for(int x = 0; x < d; x++){
			            train[j][i*d+x] = train[j][i*d+x] / temp; 
			        }
			    }	
		    }	
	    }
    }		

	int end, start, mid_t, mid;
	mid_t = n2;
	mid = n-1;
	float tmp_val;
	int* hat_J; int* acc_J;
	float* hat_Q; float* acc_Q; float*delta;
	float MIN_VAL = norm[MIN_NUM].ip;
	int cur = 0;

	for(int i = 0; i < 1; i++){
		mid_t = n2;
	    mid = n-1;
		if(num_ring <= 3) break;  //at least three rings
		
		for(int j = 0; j < num_ring; j++){
		    if(j == 0){
				for(int ii = n2-1; ii >= 0; ii--){
				    if(norm_t[ii].ip >= r_val[j]){
						end = ii;
						break;
					}	
				}

				for(int ii = end; ii >= 0; ii--){
				    if(norm_t[ii].ip >= r_val[j+1]){
						start = ii + 1;
						break;
					}					
				}

				for(int ii = qn - 1; ii>=0; ii--){
					if(q_norm[ii].ip >= norm_t[end].ip ){
						cur = ii;
						break;
					}
				}				
			}
			else if( r_num[j-1] / rat_ < MIN_NUM){ //between MIN_NUM and 2*MIN_NUM
				num_ring = j;
				break;
			}
			else if(j == num_ring-1){
				tmp_val = norm[r_num[j-1] / rat_].ip;
				for(int ii = mid_t-1; ii >= 0; ii--){
				    if(norm_t[ii].ip >= tmp_val){
						end = ii;
						break;
					}	
				}
				for(int ii = end; ii >= 0; ii--){
				    if(norm_t[ii].ip >= MIN_VAL){
						start = ii + 1;
						break;
					}					
				}						
			}			
            else{
				
				if( r_num[j+1] >= r_num[j-1] / rat_ ){
					r_num[j] = r_num[j-1]/rat_;
					r_val[j] = norm[r_num[j-1] / rat_].ip;
					continue;
				}
				
				tmp_val = norm[r_num[j-1] / rat_].ip;
				for(int ii = mid_t-1; ii >= 0; ii--){
				    if(norm_t[ii].ip >= tmp_val){
						end = ii;
						break;
					}	
				}
				for(int ii = end; ii >= 0; ii--){
				    if(norm_t[ii].ip >= r_val[j+1]){
						start = ii + 1;
						break;
					}					
				}				
			}	
			
            int diff = end - start;
			tmp_val = norm_t[end].ip;
			if(diff == 0){
				mid_t = end;
				if(j == 0){
				    for(int ii = n-1; ii >= 0; ii--){
					    if(norm[ii].ip >= tmp_val){
							mid = ii;
							r_num[j] = mid;
							r_val[j] = norm[mid].ip;
							break;
						}
					}
				}
				else{					
				    for(int ii = mid; ii >= 0; ii--){
					    if(norm[ii].ip >= tmp_val){
							mid = ii;
							r_num[j] = mid;
							r_val[j] = norm[mid].ip;							
							break;						
						}						
					}	
				}
                continue;				
			}
			else if(diff == -1){
				continue;
			}
		    
			hat_J = new int[diff];  //check if 0
			acc_J = new int[diff];
			hat_Q = new float[diff+1];
			acc_Q = new float[diff+1];
			delta = new float[diff+1];
				
			memset(hat_J, 0 ,sizeof(int)*diff);
			memset(acc_J, 0 ,sizeof(int)*diff);
			memset(hat_Q, 0 ,sizeof(float)*(diff+1));
			memset(acc_Q, 0 ,sizeof(float)*(diff+1));
            memset(delta, 0 ,sizeof(float)*(diff+1));				
			
			for(int ii = cur; ii >= 0; ii--){
				if(q_norm[ii].ip >= norm_t[end].ip){
					cur = ii;
					break;
				}	
			}
			
            for(int l = end-1; l >= start; l--){ 					
				for(int ii = cur; ii >= 0; ii--){
					if(q_norm[ii].ip >= norm_t[l].ip){
						hat_J[l-start] = cur - ii;
						cur = ii;
						break;
					}	
				}
			}
                
            for(int l = 0; l < diff; l++){
				if(l == 0) acc_J[l] = hat_J[0];
				else{
					acc_J[l] = acc_J[l-1] + hat_J[l];
				}	
			}
			
            for(int l = end; l >= start; l--){
				if(l == end) acc_Q[l-start] = err[end];
				else{
					acc_Q[l-start] = acc_Q[l-start+1] + err[l];
				}
			}
			
            for(int l = end; l >= start; l--){
				hat_Q[l-start] = err[l];
			}			

			float min_sum2 = 0;
			int min_id2;
            for(int l = end; l >= start + 1; l--){
				if(l == end) {min_id2 = end-start; continue;}
				int diff2 = l - start;
				delta[diff2] = -1 *acc_J[diff2-1] * hat_Q[diff2] + acc_Q[diff2+1] * hat_J[diff2] + delta[l - start + 1];
				if(delta[diff2] < min_sum2){
					min_sum2 = delta[diff2];
                    min_id2 = diff2;
				}       
			}
			
			mid_t = min_id2 + start;
			tmp_val = norm_t[mid_t].ip;
			
			if(j == 0)
			    old_num = r_num[j];
            else			
		        old_num = r_num[j-1]/rat_;
		
		    for(int ii = mid; ii >= 0; ii--){
			    if(norm[ii].ip >= tmp_val){
					mid = ii;
					if(mid < old_num){			
					    r_num[j] = mid;
					    r_val[j] = norm[mid].ip;
                    }
                    else{
					    r_num[j] = old_num;
					    r_val[j] = norm[old_num].ip;						
					}
						
					break;						
				}						
			}		
		}
	}
      	
	cout << "Training time:" << 1e-6 * stopw1.getElapsedTimeMicro() << "  seconds\n";
	
	StopW stopw2 = StopW();
    L2Space l2space(vecdim);	
	int cur_ring = 0;
	
	//-------------determine layer number-----------
	int* Layer = new int[n];
	for(int i = n-1; i>=0; i--){
		if(cur_ring < num_ring && i < r_num[cur_ring]){
			cur_ring++;
			while(cur_ring < num_ring && i < r_num[cur_ring]){
                cur_ring++;
            }
		}	
		Layer[i] = cur_ring; //from 0	
	}

	cvMatMul( M_RC, PCA_R, M_RC);
	unsigned char** dip = new unsigned char* [n];
	for(int i = 0; i < n; i++){
		dip[i] = new unsigned char[M];
	}
	
	for(int i = 0; i < D; i++){
		for(int j = 0; j < D; j++){ 
            R[i * D + j] = cvmGet (M_RC, i, j);
		}
	}

	rotation_(R, n, D, data, Layer, norm);

	float max_sum;
	int num;

    float temp_sum;
	for(int i = 0; i < n; i++){
        temp_sum = 0;
		if(Layer[i] < 1)
		    break;
		for(int j = 0; j < M; j++){
			for(int l = 0; l < L; l++){
			    sum = 0;
			    for(int s = 0; s < d; s++){
				    sum += data[norm[i].id][j*d+s] * vec[j][l][s];
			    }
			    if(l == 0) {max_sum = sum; num = 0;}
				else if(sum > max_sum) {max_sum = sum; num = l;}				
			}
			dip[norm[i].id][j] = num;                    
		}
	}

	float*** dist_book = new float**[M];
	for(int i = 0; i < M; i++)
		dist_book[i] = new float*[L];
	for(int i = 0; i < M; i++){
		for(int j = 0; j < L; j++){
			dist_book[i][j] = new float[L];
			memset(dist_book[i][j], 0 ,sizeof(float)*L);
		}
	}
	
    for(int i = 0; i < M; i++){
		for(int j = 0; j < L; j++){
            for(int l = 0; l < L; l++){
                for(int x = 0; x < d; x++)				
		            dist_book[i][j][l] += -1 * (vec[i][j][x] * vec[i][l][x]);
			}
	    }
	}
	cout << "Rotation and quantization time:" << 1e-6 * stopw2.getElapsedTimeMicro() << "  seconds\n";
    cout << "Building index:\n";
     appr_alg = new HierarchicalNSW<float>(&l2space, vecsize, r_num, num_ring, M2, efConstruction);


		if(Layer[0] < 1) {printf("error\n"); exit(0);}
        appr_alg->addPoint((void *) (dip[ norm[0].id ]), (size_t) (norm[0].id), Layer[0]-1, norm[0].ip,  dist_book);
 
        int j1 = 0;
        StopW stopw = StopW();
        StopW stopw_full = StopW();
        size_t report_every = 100000;
#pragma omp parallel for
        for (int i = 1; i < vecsize; i++) {
	   float* mass = new float[vecdim];
            int j2=0;
#pragma omp critical
            {
                				
                j1++;
                j2=j1;
                if (j1 % report_every == 0) {
                    cout << j1 / (0.01 * vecsize) << " %, "
                         << report_every / (1000.0 * 1e-6 * stopw.getElapsedTimeMicro()) << " kips " << " Mem: "
                         << getCurrentRSS() / 1000000 << " Mb \n";
                    stopw.reset();
                }
            }
			if(Layer[j2] >= 1){
                appr_alg->addPoint((void *) (dip[ norm[j2].id ]), (size_t) (norm[j2].id), Layer[j2]-1, norm[j2].ip, dist_book);
            }                   
        }

        cout << "Build time:" << 1e-6 * stopw_full.getElapsedTimeMicro() << "  seconds\n";
        appr_alg->saveIndex(path_index);
		
		//-----write info--------------------------------		
		FILE* fv = fopen(path_info, "w");		
		
	    for(int i = 0; i < M; i++){
		    for(int l = 0; l < L; l++){  		
		        for(int x = 0; x < d; x++){
	                fprintf(fv, "%f ", vec[i][l][x]);
			    }
			    fprintf(fv, "\n");
		    }
	    }
	
	    for(int i = 0; i < D; i++){
		   for(int j = 0; j < D; j++){ 
                fprintf(fv, "%f ", cvmGet (M_RC, i, j));
		    }
		    fprintf(fv, "\n");
	    }
		
		fprintf(fv, "%d ", num_ring);
		for(int i = 0; i < num_ring; i++){ 
            fprintf(fv, "%d ", r_num[i]);
            fprintf(fv, "%f ", norm[r_num[i]].ip);
		}
        fclose(fv);
        return;
    }
	
	//-------------read info---------------------------------------

	int D2 = M * d;	
	FILE* fv2 = fopen(path_info, "r");
	
	for(int i = 0; i < M; i++){
		for(int l = 0; l < L; l++){  		
		    for(int x = 0; x < d; x++){
	            fscanf(fv2, "%f ", &vec[i][l][x]);
			}
			fscanf(fv2, "\n");
		}
	}
	
	for(int i = 0; i < D2; i++){
		for(int j = 0; j < D2; j++){ 
            fscanf(fv2, "%f ", &R[i*D2+j]);
		}
		fscanf(fv2, "\n");
	}
	fscanf(fv2, "%d ", &num_ring);

        
	float* max_norm = new float[num_ring];
	for(int i = 0; i < num_ring; i++){
        float x; 
        fscanf(fv2, "%d ", &r_num[i]);
        fscanf(fv2, "%f ", &x);
        max_norm[i] = x;
    }

	//----------------compute distance book-------------------------
	
	float*** qip = new float** [qsize];
	for(int i = 0; i < qsize; i++){
		qip[i] = new float* [M];
	}
	for(int i = 0; i < qsize; i++){
		for(int j = 0; j < M; j++){
		    qip[i][j] = new float[L];
		    memset(qip[i][j], 0 ,sizeof(float)*L);	
		}	
	}
	
	float** query2  = new float*[qsize];
	for(int i = 0; i < qsize; i++){
		query2[i] = new float[D2];
		memset(query2[i], 0 ,sizeof(float)*D2);
	}
	
		
	ifstream inputQ2(path_data, ios::binary);
        float** query_org2 = new float* [qsize];
	for (int i = 0; i < qsize; i++) query_org2[i] = new float[vecdim];
	
        for (int i = 0; i < qsize; i++) {
             int t;
             inputQ2.read((char *) &t, 4);
             inputQ2.read((char *) (query_org2[i]), 4 * vecdim);
        }

	float** query = new float*[qsize];
	for (int i = 0; i < qsize; i++) query[i] = new float[D2];	
			
	for (int i = 0; i < qsize; i++){  //adjust the diemsion and resort the data
		ind1 = 0; ind2 = 0;
		for(int j = 0; j < M; j++){
			for(int l = 0; l < d-1; l++){
		            query[i][ind1] = query_org2[i][ind2];
			    ind1++; ind2++;
			}
			if(sub != ind1 - ind2){
			    query[i][ind1] = 0;
			    ind1 ++;
			}
            else{
				query[i][ind1] = query_org2[i][ind2];
			    ind1++; ind2++;	
			}			
		}
		if(ind1 != M * d || ind2 != vecdim) {printf("error\n"); exit(0);}
	}


			
    StopW stopw2 = StopW();

	for(int cn = 0; cn < qsize; cn++){
		for(int j = 0; j < D2; j++){			
			query2[cn][j] = compare2( &(R[j*D2]), query[cn], D2);
	    }		
	
		for(int j = 0; j < M; j++){
			for(int l = 0; l < L; l++){                                                                                 
                for(int z = 0; z < d; z++)
                    qip[cn][j][l] += query2[cn][j*d +z] * vec[j][l][z];

                    qip[cn][j][l] = -1 * qip[cn][j][l];                                                                                            		
                }	
		    } 
        } 
  
        float time_us_per_query = stopw2.getElapsedTimeMicro() / qsize;
        printf("time_us_per_query = %.5f\n", time_us_per_query);

	cout << "Loading GT:\n";
    ifstream inputGT(path_gt, ios::binary);
    unsigned int *massQA = new unsigned int[qsize * maxk];
    for (int i = 0; i < qsize; i++) {
        int x;
        inputGT.read((char *) &x, 4);
        inputGT.read((char *) (massQA + 100 * i), 4 * maxk);
    }
    inputGT.close();
	
	float *massb = new float[vecdim];
	
    cout << "Loading queries:\n";
    float *massQ = new float[qsize * vecdim];
    inputQ2.seekg(0, ios::beg);

    for (int i = 0; i < qsize; i++) {
        int in = 0;
        inputQ2.read((char *) &in, 4);
        inputQ2.read((char *) massb, 4 * vecdim);
        for (int j = 0; j < vecdim; j++) {
            massQ[i * vecdim + j] = massb[j];
        }

    }
    inputQ2.close();
		
    vector<std::priority_queue<std::pair<float, labeltype >>> answers;
   // cout << "Parsing gt:\n";
    get_gt(massQA, massQ, vecsize, qsize, l2space, vecdim, answers, K);
   // cout << "Loaded gt\n";
    for (int i = 0; i < 1; i++)
        test_vs_recall(massQ, vecsize, qsize, *appr_alg, vecdim, answers, K, qip, max_norm);
    return;
}
