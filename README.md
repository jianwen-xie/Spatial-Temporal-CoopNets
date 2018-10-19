# Spatial-Temporal CoopNets

This repository contains a tensorflow implementation for ***spatial-temporal*** CoopNets, which is from Experiment 7 in the paper
"[Cooperative Training of Descriptor and Generator Networks](http://www.stat.ucla.edu/~jxie/CoopNets/CoopNets_files/doc/CoopNets_PAMI.pdf)"

## Reference
    @article{coopnets,
        author = {Xie, Jianwen and Lu, Yang and Gao, Ruiqi and Zhu, Song-Chun and Wu, Ying Nian},
        title = {Cooperative Training of Descriptor and Generator Networks},
        journal={IEEE transactions on pattern analysis and machine intelligence (PAMI)},
        year = {2018},
        publisher={IEEE}
    }


<p align="center">
    <img src="https://github.com/jianwen-xie/Spatial-Temporal-CoopNets/blob/master/demo/fire_pot.gif" width="350px"/>
    <img src="https://github.com/jianwen-xie/Spatial-Temporal-CoopNets/blob/master/demo/waterfall.gif" width="350px"/>
    <img src="https://github.com/jianwen-xie/Spatial-Temporal-CoopNets/blob/master/demo/vapour.gif" width="350px"/>
    <img src="https://github.com/jianwen-xie/Spatial-Temporal-CoopNets/blob/master/demo/flashing_light.gif" width="350px"/></p>

## Requirements
- Python 2.7 or Python 3.3+
- [Tensorflow r1.3+](https://www.tensorflow.org/install/)
- Install [FFmpeg](https://www.ffmpeg.org/download.html)
    ```bash
    sudo sh ffmpeg_installer.sh
    ```
- Install required Python libraries
    ```bash
    pip install -r requirements.txt
    ```

## How to run

- Clone this repo:
    ```bash
    git clone https://github.com/jianwen-xie/Spatial-Temporal-CoopNets
    cd Spatial-Temporal-CoopNets
    ```
- Put training data ***fire_pot.avi*** into path:
   ```bash
   ./trainingVideo/fire_pot/fire_pot.avi
   ```
- To train a model with ***fire_pot.avi*** video:
    ```bash
    $ python train_coop_video.py --category fire_pot \
                                 --num_epochs 1000 --num_chains 2 \
                                 --data_path ./trainingVideo \
                                 --output_dir ./output_coop_video
    ```
- Synthesized results will be saved in
    ```bash
    ./output_coop_video/fire_pot
    ```
## Other related references
    @inproceedings{coopnets,
       author = {Xie, Jianwen and Lu, Yang and Gao, Ruiqi and Wu, Ying Nian},
       title = {Cooperative Learning of Energy-Based Model and Latent Variable Model via MCMC Teaching},
       booktitle = {The 32nd AAAI Conference on Artitifical Intelligence},
       year = {2018}
    }
    
    @inproceedings{stgconvnet,
       author = {Xie, Jianwen and Zhu, Song-Chun and Wu, Ying Nian},
       title = {Synthesizing Dynamic Patterns by Spatial-Temporal Generative ConvNet},
       booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
       month = {July},
       year = {2017}
    } 

For any questions, please contact Jianwen Xie (jianwen@ucla.edu) and Zilong Zheng (zilongzheng0318@ucla.edu).
