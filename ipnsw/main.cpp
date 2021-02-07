#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <queue>
#include <chrono>
#include "hnswlib.h"
#include <algorithm>
#include <ctime>
#include <omp.h>


const int defaultEfConstruction = 1024;
const int defaultEfSearch = 128;
const int defaultM = 32;
const int defaultTopK = 1;

void printHelpMessage()
{
    std::cerr << "Usage: main [OPTIONS]" << std::endl;
    std::cerr << "This tool supports two modes: to construct the graph index from the database and to retrieve top K maximum inner product vectors using the constructed index. Each mode has its own set of parameters." << std::endl;
    std::cerr << std::endl;

    std::cerr << "  --mode            " << "\"database\" or \"query\". Use \"database\" for" << std::endl;
    std::cerr << "                    " << "constructing index and \"query\" for top K" << std::endl;
    std::cerr << "                    " << "maximum inner product retrieval" << std::endl;
    std::cerr << std::endl;

    std::cerr << "Database mode supports the following options:" << std::endl;
    std::cerr << "  --database        " << "Database filename. Database should be stored in binary format." << std::endl;
    std::cerr << "                    " << "Vectors are written consecutive and numbers are" << std::endl;
    std::cerr << "                    " << "represented in 4-bytes floating poing format (float in C/C++)" << std::endl;
    std::cerr << "  --databaseSize    " << "Number of vectors in the database" << std::endl;
    std::cerr << "  --dimension       " << "Dimension of vectors" << std::endl;
    std::cerr << "  --outputGraph     " << "Filename for the output index graph" << std::endl;
    std::cerr << "  --efConstruction  " << "efConstruction parameter. Default: " << defaultEfConstruction << std::endl;
    std::cerr << "  --M               " << "M parameter. Default: " << defaultM << std::endl;
    std::cerr << std::endl;
    std::cerr << "Query mode supports the following options:" << std::endl;
    std::cerr << "  --query           " << "Query filename. Queries should be stored in binary format." << std::endl;
    std::cerr << "                    " << "Vectors are written consecutive and numbers are" << std::endl;
    std::cerr << "                    " << "represented in 4-bytes floating poing format (float in C/C++)" << std::endl;
    std::cerr << "  --querySize       " << "Number of queries" << std::endl;
    std::cerr << "  --dimension       " << "Dimension of vectors" << std::endl;
    std::cerr << "  --inputGraph      " << "Filename for the input index graph" << std::endl;
    std::cerr << "  --efSearch        " << "efSearch parameter. Default: " << defaultEfSearch << std::endl;
    std::cerr << "  --topK            " << "Top size for retrieval. Default: " << defaultTopK << std::endl;
    std::cerr << "  --output          " << "Filename to print the result. Default: " << "stdout" << std::endl;
    
}

void printError(std::string err)
{
    std::cerr << err << std::endl;
    std::cerr << std::endl;
    printHelpMessage();
}

int main(int argc, char** argv) {

    std::string mode;
    std::ifstream input;
    std::ifstream inputQ;
    int efConstruction = defaultEfConstruction;
    int efSearch = defaultEfSearch;
    int M = defaultM;
    int vecsize = -1;
    int qsize = -1;
    int vecdim = -1;
    std::string graphname;
    std::string outputname;
    int topK = defaultTopK;
    std::string dataname;
    std::string queryname;

    hnswlib::HierarchicalNSW<float> *appr_alg;
    
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--h" || std::string(argv[i]) == "--help") {
            printHelpMessage();
           return 0; 
        }
    }
    for (int i = 1; i < argc - 1; i++) {
        if (std::string(argv[i]) == "--m" || std::string(argv[i]) == "--mode") {
            if (std::string(argv[i + 1]) == "database") {
                mode = "database";
            } else if (std::string(argv[i + 1]) == "query") {
                mode = "query";
            } else {
                printError("Unknown running mode \"" + std::string(argv[i + 1]) + "\". Please use \"database\" or \"query\"");
                return 0;
            }
            break;
        }
    }
    if (mode.empty()) {
        printError("Running mode was not specified");
        return 0;
    }

    std::cout << "Mode: " << mode << std::endl;

    
    if (mode == "database") {
        for (int i = 1; i < argc - 1; i++) {
            if (std::string(argv[i]) == "--d" || std::string(argv[i]) == "--data" || std::string(argv[i]) == "--database") {
                input.open(argv[i + 1], std::ios::binary);
                if (!input.is_open()) {
                    printError("Cannot open file \"" + std::string(argv[i + 1]) + "\"");
                    return 0;
                } else {
                    dataname = std::string(argv[i + 1]);
                }
                break;
            }
        }
        if (!input.is_open()) {
            printError("Database file was not specified");
            return 0;
        }
        std::cout << "Database file: " << dataname << std::endl;



        for (int i = 1; i < argc - 1; i++) {
            if (std::string(argv[i]) == "--dataSize" || std::string(argv[i]) == "--dSize" || std::string(argv[i]) == "--databaseSize") {
                if (sscanf(argv[i + 1], "%d", &vecsize) != 1 || vecsize <= 0) {
                    printError("Inappropriate value for database size: \"" + std::string(argv[i + 1]) + "\"");
                    return 0;
                }
                break;
            }
        }
        if (vecsize == -1) {
            printError("Database size was not specified");
            return 0;
        }
        std::cout << "Database size: " << vecsize << std::endl;
        
        
        for (int i = 1; i < argc - 1; i++) {
            if (std::string(argv[i]) == "--dataDim" || std::string(argv[i]) == "--dimension" || std::string(argv[i]) == "--databaseDimension") {
                if (sscanf(argv[i + 1], "%d", &vecdim) != 1 || vecdim <= 0) {
                    printError("Inappropriate value for database dimension: \"" + std::string(argv[i + 1]) + "\"");
                    return 0;
                }
                break;
            }
        }
        if (vecdim == -1) {
            printError("Database dimension was not specified");
            return 0;
        }
        std::cout << "Database dimension: " << vecdim << std::endl;
        
        
        for (int i = 1; i < argc - 1; i++) {
            if (std::string(argv[i]) == "--outGraph" || std::string(argv[i]) == "--outputGraph") {
                std::ofstream outGraph(argv[i + 1]);
                if (!outGraph.is_open()) {
                    printError("Cannot create file \"" + std::string(argv[i + 1]) + "\"");
                    return 0;
                }
                outGraph.close();
                graphname = std::string(argv[i + 1]);
                break;
            }
        }
        if (graphname.empty()) {
            printError("Filename of the output graph was not specified");
            return 0;
        }
        std::cout << "Output graph: " << graphname << std::endl;


        
        for (int i = 1; i < argc - 1; i++) {
            if (std::string(argv[i]) == "--efConstruction") {
                if (sscanf(argv[i + 1], "%d", &efConstruction) != 1 || efConstruction <= 0) {
                    printError("Inappropriate value for efConstruction: \"" + std::string(argv[i + 1]) + "\"");
                    return 0;
                }
                break;
            }
        }
        std::cout << "efConstruction: " << efConstruction << std::endl;
        
        for (int i = 1; i < argc - 1; i++) {
            if (std::string(argv[i]) == "--M") {
                if (sscanf(argv[i + 1], "%d", &M) != 1 || M <= 0) {
                    printError("Inappropriate value for M: \"" + std::string(argv[i + 1]) + "\"");
                    return 0;
                }
                break;
            }
        }
        std::cout << "M: " << M << std::endl;
    


       
        hnswlib::L2Space l2space(vecdim);
        float *mass = new float[(size_t) vecsize * vecdim];
        //input.read((char *)mass, vecsize * vecdim * sizeof(float));
        for(size_t i = 0; i < vecsize; i++){
			int t;
			input.read((char *) &t, sizeof(float));
			input.read((char *) (mass + i * vecdim), vecdim * sizeof(float));
		}		
		input.close();
        
        appr_alg = new hnswlib::HierarchicalNSW<float>(&l2space, vecsize, M, efConstruction);
        std::cout << "Building index\n";
        double t1 = omp_get_wtime();
        for (int i = 0; i < 1; i++) {
            appr_alg->addPoint((void *)(mass + vecdim*i), (size_t)i);
        }
#pragma omp parallel for
        for (size_t i = 1; i < vecsize; i++) {
            appr_alg->addPoint((void *)(mass + vecdim*i), (size_t)i);
        }
        double t2 = omp_get_wtime();
 
        std::cout << "Index built, time=" << t2 - t1 << " s" << "\n";
        appr_alg->SaveIndex(graphname.data());
        delete appr_alg;
        delete mass;
    } else {
        for (int i = 1; i < argc - 1; i++) {
            if (std::string(argv[i]) == "--q" || std::string(argv[i]) == "--query") {
                inputQ.open(argv[i + 1], std::ios::binary);
                if (!inputQ.is_open()) {
                    printError("Cannot open file \"" + std::string(argv[i + 1]) + "\"");
                    return 0;
                } else {
                    queryname = std::string(argv[i + 1]);
                }
                break;
            }
        }
        if (!inputQ.is_open()) {
            printError("Query file was not specified");
            return 0;
        }
        std::cout << "Query filename: " << queryname << std::endl;



        for (int i = 1; i < argc - 1; i++) {
            if (std::string(argv[i]) == "--querySize" || std::string(argv[i]) == "--qSize") {
                if (sscanf(argv[i + 1], "%d", &qsize) != 1 || qsize <= 0) {
                    printError("Inappropriate value for query size: \"" + std::string(argv[i + 1]) + "\"");
                    return 0;
                }
                break;
            }
        }
        if (qsize == -1) {
            printError("Query size was not specified");
            return 0;
        }
        std::cout << "Query size: " << qsize << std::endl;
        
        
        for (int i = 1; i < argc - 1; i++) {
            if (std::string(argv[i]) == "--queryDim" || std::string(argv[i]) == "--dimension" || std::string(argv[i]) == "--queryDimension") {
                if (sscanf(argv[i + 1], "%d", &vecdim) != 1 || vecdim <= 0) {
                    printError("Inappropriate value for query dimension: \"" + std::string(argv[i + 1]) + "\"");
                    return 0;
                }
                break;
            }
        }
        if (vecdim == -1) {
            printError("Query dimension was not specified");
            return 0;
        }
        std::cout << "Query dimension: " << vecdim << std::endl;
        
        
        for (int i = 1; i < argc - 1; i++) {
            if (std::string(argv[i]) == "--inGraph" || std::string(argv[i]) == "--inputGraph") {
                std::ifstream inGraph(argv[i + 1]);
                if (!inGraph.is_open()) {
                    printError("Cannot open file \"" + std::string(argv[i + 1]) + "\"");
                    return 0;
                }
                inGraph.close();
                graphname = std::string(argv[i + 1]);
                break;
            }
        }
        if (graphname.empty()) {
            printError("Filename of the input graph was not specified");
            return 0;
        }
        std::cout << "Input graph: " << graphname << std::endl;


        
        for (int i = 1; i < argc - 1; i++) {
            if (std::string(argv[i]) == "--efSearch") {
                if (sscanf(argv[i + 1], "%d", &efSearch) != 1 || efSearch <= 0) {
                    printError("Inappropriate value for efSearch: \"" + std::string(argv[i + 1]) + "\"");
                    return 0;
                }
                break;
            }
        }
        std::cout << "efSearch: " << efSearch << std::endl;
        
        for (int i = 1; i < argc - 1; i++) {
            if (std::string(argv[i]) == "--topK") {
                if (sscanf(argv[i + 1], "%d", &topK) != 1 || topK <= 0) {
                    printError("Inappropriate value for top size: \"" + std::string(argv[i + 1]) + "\"");
                    return 0;
                }
                break;
            }
        }
        std::cout << "Top size: " << topK << std::endl;
        
        for (int i = 1; i < argc - 1; i++) {
            if (std::string(argv[i]) == "--output" || std::string(argv[i]) == "--out") {
                std::ofstream output(argv[i + 1]);
                if (!output.is_open()) {
                    printError("Cannot open file \"" + std::string(argv[i + 1]) + "\"");
                    return 0;
                }
                output.close();
                outputname = std::string(argv[i + 1]);
                break;
            }
        }
        if (outputname.empty()) {
            std::cout << "Output file: " << "stdout" << std::endl;
        } else {
            std::cout << "Output file: " << outputname << std::endl;
        }





        hnswlib::L2Space l2space(vecdim);
        float *massQ = new float[qsize * vecdim];
        //inputQ.read((char *)massQ, qsize * vecdim * sizeof(float));
		
		for(size_t i = 0; i < qsize; i++){
			int t;
			inputQ.read((char *) &t, sizeof(float));
			inputQ.read((char *) (massQ + i * vecdim), vecdim * sizeof(float));
		}		
        inputQ.close();

        std::priority_queue< std::pair< float, labeltype >> gt[qsize];
		
        appr_alg = new hnswlib::HierarchicalNSW<float>(&l2space, graphname.data(), false);
        appr_alg->setEf(efSearch);
        std::ofstream fres;
        if (!outputname.empty()) {
            fres.open(outputname);
        }
     
		//------------read ground truth------------------
        int MAXK = 100;
		
		int** truth_ = new int* [qsize];
		for(int i = 0; i < qsize; i++)
			truth_[i] = new int[MAXK];
		
		std::ifstream inputGT("truth.gt", std::ios::binary);
		for(int i = 0; i < qsize; i++){
			int t;
            inputGT.read((char *) &t, 4);
			inputGT.read( (char *) truth_[i], MAXK * sizeof(float) );
		}
        inputGT.close();		
		//------------------------------------------------
         std::vector<size_t> efs;// = { 10,10,10,10,10 };
   
        for (int i = 1; i < 10; i++) {
            efs.push_back(i);
        }
  
        for (int i = 10; i < 100; i += 10) {
            efs.push_back(i);
        }

        for (int i = 100; i <= 6000; i += 200) {
            efs.push_back(i);
        }

		int total = topK * qsize;
		int correct = 0;
		float recall;
	    for (size_t ef : efs) {
		//	appr_alg.setEf(ef);
                    appr_alg->ef_ = ef;
		    appr_alg->dist_calc = 0;
		    double avg_dist = 0;
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < qsize; i++) {
                gt[i] = appr_alg->searchKnn(massQ + vecdim*i, topK, &avg_dist);
            }
		
		    correct = 0;
            for (int i = 0; i < qsize; i++) {
                std::vector <int> res;
                 while (!gt[i].empty()) {
                    res.push_back(gt[i].top().second);
                    gt[i].pop();
                }
                std::reverse(res.begin(), res.end());
                for(int j = 0; j < topK; j++){
					for(int l = 0; l < topK; l++){
						if(res[j] == truth_[i][l]){
						    correct++;
							break;
						}
					}
				}
            }
			recall = 1.0f * correct / total;
			avg_dist = avg_dist / qsize;
			auto end = std::chrono::high_resolution_clock::now();
		//	printf("correct = %d total = %d\n", correct, total);
			std::cout << ef << "\t" << recall << "\t"  << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / (double)qsize << "ms" << "\t" << avg_dist << "\n";

        }
		
        /*		
        for (int i = 0; i < qsize; i++) {
            std::vector <int> res;
            while (!gt[i].empty()) {
                res.push_back(gt[i].top().second);
                gt[i].pop();
            }
            std::reverse(res.begin(), res.end());
            for (auto it: res) {
                if (!outputname.empty()) {
                    fres << it << ' ';
                } else {
                    std::cout << it << ' ';
                }
            }
            if (!outputname.empty()) {
                fres << std::endl;
            } else {
                std::cout << std::endl;
            }
        }
        */
        if (!outputname.empty()) {
            fres.close();
        }
       // std::cout << "Average query time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / (double)qsize << "ms" << std::endl;
        delete appr_alg;
        delete massQ;
    }
    return 0;
}
