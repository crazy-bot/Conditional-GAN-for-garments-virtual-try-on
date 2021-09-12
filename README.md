# Densepose based virtual try on

The goal of this project is to accomplish virtual try on from a single RGB image and target cloth

# Network architecture and pipeline:
We have extracted densepose given a RGB image using official [Densepose repo]()

The input to our pipeline is densepose IUV map and the target RGB cloth image as shown in the below figure. The network architecture is motivated from [StylePoseGAN]() and [StyleGAN2]()
![input-output-pipeline](/images/pipeline.png)

# Dataset
.* Test1: We collected 10000 pair images of cloth and person on that cloth from internet. 

.* Test2: We collected 100 pair images of same cloth with different person in very hard poses of soccer.

# Result
Below are the generated images followed by real images for Test1:
- ![real images](/images/reals_test1.png)
- ![fake images](/images/fakes_test1.png)

Below are the generated images followed by real images for Test2:
- ![real images](/images/reals_test2.png)
- ![fake images](/images/fakes_test2.png)

# Training and Testing:
The codebase is developed on top of official [StyleGAN2-ADA](). Kindly follow the similar steps for training and testing.

# Checkpoint
- model weights for test1 can be downloaded [here](stylegan2/runs/00003-Zalando-stylegan2-batch128-noaug/final.pkl)
- model weights for test2 can be downloaded [here](stylegan2/runs/00005-soccershirt-stylegan2-batch128-noaug-resumecustom/final.pkl)








