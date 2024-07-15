[![MBARI](https://www.mbari.org/wp-content/uploads/2014/11/logo-mbari-3b.png)](http://www.mbari.org)
[![semantic-release](https://img.shields.io/badge/%20%20%F0%9F%93%A6%F0%9F%9A%80-semantic--release-e10079.svg)](https://github.com/semantic-release/semantic-release)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/language-Python-blue.svg)](https://www.python.org/downloads/)
 
**sdcat** 

*Sliced Detection and Clustering Analysis Toolkit*

This repository processes images using a sliced detection and clustering workflow.
If your images look something like the image below, and you want to detect objects in the images, 
and optionally cluster the detections, then this repository is for you.

---
Drone/UAV
---
![](https://raw.githubusercontent.com/mbari-org/sdcat/main/docs/imgs/DSC00770_with_logo.png)
---
ISIIS Plankton Imager
---
![](https://raw.githubusercontent.com/mbari-org/sdcat/main/docs/imgs/CFE_ISIIS-081-2023-07-12_14-38-38.862_000058_with_logo.png) 

---
DeepSea Imaging System
---
![](https://raw.githubusercontent.com/mbari-org/sdcat/main/docs/imgs/1696956731236857_with_logo.png)
---
DINO + HDBSCAN Clustering
---
The clustering is done with a DINO Vision Transformer (ViT) model, and a cosine similarity metric with the HDBSCAN algorithm.
The DINO model is used to generate embeddings for the detections, and the HDBSCAN algorithm is used to cluster the detections.
To reduce the dimensionality of the embeddings, the t-SNE algorithm is used to reduce the embeddings to 2D.
The defaults are set to produce fine-grained clusters, but the parameters can be adjusted to produce coarser clusters.
The algorithm workflow looks like this:

![](https://raw.githubusercontent.com/mbari-org/sdcat/main/docs/imgs/cluster_workflow.png)
  
# Installation
 
Pip install the sdcat package including all [requirements](https://github.com/mbari-org/sdcat/blob/main/pyproject.toml) with:

```bash
pip install sdcat
```

Alternatively, [Docker](https://www.docker.com) can be used to run the code. A pre-built docker image is available at [Docker Hub](https://hub.docker.com/r/mbari/sdcat) with the latest version of the code.  
 
Detection
```shell
docker run -it -v $(pwd):/data mbari/sdcat detect --image-dir /data/images --save-dir /data/detections --model MBARI-org/uav-yolov5
```
Followed by clustering
```shell
docker run -it -v $(pwd):/data mbari/sdcat cluster detections --det-dir /data/detections/ --save-dir /data/detections --model MBARI-org/uav-yolov5
```

A GPU is recommended for clustering and detection.  If you don't have a GPU, you can still run the code, but it will be slower.
If running on a CPU, multiple cores are recommended and will speed up processing.

```shell
docker run -it --gpus all -v $(pwd):/data mbari/sdcat:latest-cuda12 detect --image-dir /data/images --save-dir /data/detections --model MBARI-org/uav-yolov5
```

# Commands

To get all options available, use the --help option.  For example:

```shell
sdcat --help
```
which will print out the following:
```shell
Usage: sdcat [OPTIONS] COMMAND [ARGS]...

  Process images from a command line.

Options:
  -V, --version  Show the version and exit.
  -h, --help     Show this message and exit.

Commands:
  cluster  Cluster detections.
  detect   Detect objects in images

```

To get details on a particular command, use the --help option with the command.  For example, with the **cluster** command:

```shell
 sdcat  cluster --help 
```

which will print out the following:
```shell
Usage: sdcat cluster [OPTIONS]

  Cluster detections from a single collection.

Options:
  --det-dir TEXT      Input folder with raw detection results
  --save-dir TEXT     Output directory to save clustered detection results
  --device TEXT       Device to use.
  -h, --help          Show this message and exit.

```

## File organization

The sdcat toolkit generates data in the following folders. Here, we assume both detection and clustering is output to the same root folder.:
 
```
/data/20230504-MBARI/
└── detections
    └── hustvl
        └── yolos-small                         # The model used to generate the detections
            ├── det_raw                         # The raw detections from the model
            │   └── csv                    
            │       ├── DSC01833.csv
            │       ├── DSC01859.csv
            │       ├── DSC01861.csv
            │       └── DSC01922.csv
            ├── det_filtered                    # The filtered detections from the model
            ├── det_filtered_clustered          # Clustered detections from the model
                ├── crops                       # Crops of the detections 
                ├── dino_vits8...date           # The clustering results - one folder per each run of the clustering algorithm
                ├── dino_vits8..exemplars.csv   # Exemplar embeddings - examples with the highest cosine similarity within a cluster
                ├── dino_vits8..detections.csv  # The detections with the cluster id
            ├── stats.txt                       # Statistics of the detections
            └── vizresults                      # Visualizations of the detections (boxes overlaid on images)
                ├── DSC01833.jpg
                ├── DSC01859.jpg
                ├── DSC01861.jpg
                └── DSC01922.jpg

```

## Process images creating bounding box detections with the YOLOv5 model.
The YOLOv5s model is not as accurate as other models, but is fast and good for detecting larger objects in images,
and good for experiments and quick results. 
**Slice size** is the size of the detection window.  The default is to allow the SAHI algorithm to determine the slice size;
a smaller slice size will take longer to process.

```shell
sdcat detect --image-dir <image-dir> --save-dir <save-dir> --model yolov5s --slice-size-width 900 --slice-size-height 900
```

## Cluster detections from the YOLOv5 model

Cluster the detections from the YOLOv5 model.  The detections are clustered using cosine similarity and embedding
features from a FaceBook Vision Transformer (ViT) model.   

```shell
sdcat cluster --det-dir <det-dir> --save-dir <save-dir> --model yolov5s
```
  

# Related work
* https://github.com/obss/sahi
* https://github.com/facebookresearch/dinov2
* https://arxiv.org/pdf/1911.02282.pdf HDBSCAN
* https://github.com/muratkrty/specularity-removal
