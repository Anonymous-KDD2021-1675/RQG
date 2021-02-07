#include<stdlib.h>

void sift_test1B(char*, int, int, int, char*, char*, char*, char*, char*, int);
int main(int argc, char **argv) {
    char data_set[200];
    int vecsize_ = atoi(argv[2]);
    int vecdim_ = atoi(argv[3]);
    int qsize_ = atoi(argv[4]);
    int topk = atoi(argv[10]);
    
    sift_test1B(argv[1], vecsize_, vecdim_, qsize_, argv[5], argv[6], argv[7], argv[8], argv[9], topk);

    return 0;
};
