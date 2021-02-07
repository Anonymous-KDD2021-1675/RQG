# Project Title

RQG/RQG+

## Getting Started

### Prerequisites

* OpenCV 3.30+
* GCC 4.9+ with OpenMP
* CMake 2.8+
* Boost 1.55+
* TCMalloc

### Datasets (query sets, groundtruth sets)

* Yahoo!Music (https://drive.google.com/file/d/1NUZbnpJKrgUc1FB9cOnTPlVDlEaVo90P/view?usp=sharing)
* Word (https://drive.google.com/file/d/1kyMwFt-M9_R3r3NVH4f18AxDT0LqSvux/view?usp=sharing)
* ImageNet (https://drive.google.com/file/d/19MjZfwawJggGiTPo568lTZ--KM5yKUQW/view?usp=sharing)
* Tiny (https://drive.google.com/file/d/1yC1SVgIvqc6wtuEBjbLS-3mV56-qK6c3/view?usp=sharing)
* Gist (https://drive.google.com/file/d/1D0mFQAzuAi-cC_YI1AuopgMqD_m5Rz6I/view?usp=sharing)
* Glove (https://drive.google.com/file/d/1e8oa6hXsnqFhJ5eAcGdToCUjnGHxSyNT/view?usp=sharing)


### Compile On Ubuntu

Build folders for index (XXX is the dataset name)

```shell
$ mkdir data/
$ cd data/ && build XXX/
$ cd XXX/ && mkdir index/
```
* Put the dataset, query set and groundtruth set in /data/XXX/.
* The generated index is stroed in /data/XXX/index/.

Compile ip-NSW

```shell
$ cd ipnsw/
$ mkdir build/ && cd build/
$ cmake ..
$ make 
```

Compile ip-NSW+

```shell
$ cd ipplus/
$ mkdir build/ && cd build/
$ cmake ..
$ make
```

Compile RQG

```shell
$ cd RQG/
$ mkdir build/ && cd build/
$ cmake ..
$ make
```

## Run scripts (example: ImageNet)

* For other cases, change `n`(data size), `qn`(query size), `D`(dimension) and `K` accordingly.

### Run RQG
```shell
$ bash run_rqg_image.sh
```

### Run RQG+
```shell
$ bash run_rqg+_image.sh
```
