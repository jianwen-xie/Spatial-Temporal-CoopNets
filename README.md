# Spatial-Temporal CoopNets

This repository contains a tensorflow implementation for spatial-temporal CoopNets

## Requirements
- Python 2.7 or Python 3.3+
- [Tensorflow r1.3+](https://www.tensorflow.org/install/)
- Install required Python libraries
    ```bash
    pip install numpy scipy
    ```

## How to run

- Clone this repo:
    ```bash
    git clone https://github.com/jianwen-xie/Spatial-Temporal-CoopNets
    cd Spatial-Temporal-CoopNets
    ```
- To train a model with ***fire_pot*** video:
    ```bash
    $ python train_coop_video.py --category fire_pot --data_path ./trainingVideo --output_dir ./output_coop_video --num_epochs 1000 --num_chains 2
    ```
- Synthesized results will be saved in
    ```bash
    ./output_coop_video/fire_pot
    ```
    
## References
    @inproceedings{coopnets,
       author = {Xie, Jianwen and Lu, Yang and Gao, Ruiqi and Wu, Ying Nian},
       title = {Cooperative Learning of Energy-Based Model and Latent Variable Model via MCMC Teaching},
       booktitle = {The 32nd AAAI Conference on Artitifical Intelligence},
       year = {2018}
    }

For any questions, please contact Jianwen Xie (jianwen@ucla.edu) and Zilong Zheng (zilongzheng0318@ucla.edu).
