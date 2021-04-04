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
