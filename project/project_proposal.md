# Project Name: Dataset Quality Metrics in MLOps Pipeline
[Loom video](https://www.loom.com/share/a10f12fe7f51497d8beecf495a91e312)

## Team:
- [Mahavir Dwivedi](https://www.linkedin.com/in/mahavir-dwivedi/): Software Developer at Greeman-Pedersen Inc, MYC, US
- [Changsin Lee](https://www.linkedin.com/in/changsin/): CTO at TestWorks, Seoul, Korea

# Project Goal:
The project goal is to establish a set metrics and processes that can be used to measure the quality of dataset.
The motivation came from the need to objectively measure both the quality of raw dataset and the annotation quality.
In the literature, there isn't an established process or consensus on how to compare the quality of different datasets.
Any problem with the dataset is detected manually and ad hoc basis.
Our goal is to come up with ways to tackle this problem in systematic and scalable ways.
To demonstrate how this can be done, we plan to use license plate data that Mahavir plans to release.
We will build a pipeline and a data flywheel template to process, train, and release an AI model.

# What dataset will you use, or how do you plan to collect data?
Mahavir's license plate dataset (to be released at the completion of the project)
Other publicly available license plate datasets: COCO, MNIST, etc.

# What baseline will you use?
We will start with public dataset metrics and then compare and contrast different approach in measure quality dataset metrics.

# What model architecture / loss function do you propose?
YOLO v3, Faster RCNN, and a few other object detection models to compare.

# What will the end result look like?
A license plate recognition App and the MLOps pipeline with dataset metrics integrated.

# What’s your stretch goal?
 - Few shot training for license plate recognition
 - Apply the metrics and processes to other ML datasets: e.g., classification, NLP.
 - Use the similar approach to do image captioning

# Timeline
| # | Item | Description | Owner | ETA  | Status |
|---|------|-------------|-------|------|--------|
| 1 | Project proposal |  Record and publish project proposal | Both |  4/2 | DONE  |
| 2 | Project management | Setup a repository & create a project timeline | Changsin | 4/2 | DONE |
| 3 | Web App POC | Create a Web App for license plate recognition | Mahavir | 4/11 | In Progress |
| 4 | Dataset Quality POC - Bayesian | Build a POC for estimating dataset quality using Bayesian networks | Changsin | 4/11 | In Progress |
| 5 | Data Pipeline - Weights and Biases | Integrate Weights and Biases | TBD | 4/18 | Not Started |
| 6 | Data Pipeline - Streamlit | Integrate Streamlit | TBD | 4/18 | Not Started |
| 7 | Fine turing | Iterate, test, and fix | All | 4/24 | Not Started |
| 8 | Presentation | Record the presentation video and write up the report | All | 4/26 | Not Started |

# Reference
Here are some of the aricles we start with.

1. [Quality and Relevance Metrics for Selection of Multimodal Pretraining Data](https://openaccess.thecvf.com/content_CVPRW_2020/html/w56/Rao_Quality_and_Relevance_Metrics_for_Selection_of_Multimodal_Pretraining_Data_CVPRW_2020_paper.html) (Rao, et al., 2020) CVPR
2. [Confident Learning: Estimating Uncertainty in Dataset Labels](https://arxiv.org/abs/1911.00068) (Northcutt, et al., 2018)
3. [Are We Done with ImageNet?](https://arxiv.org/abs/2006.07159) (Beyer, 2020)
4. [Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels](https://arxiv.org/pdf/1804.06872.pdf) (Bo Han, et al., 2018)
5. [Auto-correcting mislabeled data](https://medium.com/@yalcinmurat1986/auto-correcting-mislabeled-data-7a4098c77357) (Yalcin, 2020)
6. [A Recipe for Training Neural Networks](http://karpathy.github.io/2019/04/25/recipe/) (Karpathy, 2019)
