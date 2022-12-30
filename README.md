# SuperCATs
For more information, check out the paper on [[paper link]](https://ieeexplore.ieee.org/document/9954872). Also check out project page here [[Project Page link]](https://ku-cvlab.github.io/SuperCATs/).<br>
*This paper is accepted in ICCE-Asia'22*

<img src="fig/result1.png" height="300" width="400"> <img src="fig/result2.png" height="298" width="400">

>**Cost Aggregation with Transformers for Sparse Correspondence** <br><br>
>Abstract : In this work, we introduce a novel network, namely SuperCATs, which aims to find a correspondence field between visually similar images. SuperCATs stands on the shoulder of the recently proposed matching networks, SuperGlue and CATs, taking the merits of both for constructing an integrative framework. Specifically, given keypoints and corresponding descriptors, we first apply attentional aggregation consisting of self- and cross- graph neural network to obtain feature descriptors. Subsequently, we construct a cost volume using the descriptors, which then undergoes a tranformer aggregator for cost aggregation. With this approach, we manage to replace the handcrafted module based on solving an optimal transport problem initially included in SuperGlue with a transformer well known for its global receptive fields, making our approach more robust to severe deformations. We conduct experiments to demonstrate the effectiveness of the proposed method, and show that the proposed model is on par with SuperGlue for both indoor and outdoor scenes.


# Network
Overview of our model is illustrated below:
![overview](fig/overview.png)
Structure of Transformer Aggregator is illustrated below:
![aggregator](fig/aggregator.png)

# Training
To train the SuperGlue with default parameters, run the following command:
```
python train.py
```
Additional useful command line parameters

* Use `--epoch` to set the number of epochs (default: `20`).
* Use `--train_path` to set the path to the directory of training images.
* Use `--eval_output_dir` to set the path to the directory in which the visualizations is written (default: `dump_match_pairs/`).
* Use `--show_keypoints` to visualize the detected keypoints (default: `False`).
* Use `--viz_extension` to set the visualization file extension (default: `png`). Use pdf for highest-quality.

# BibTex
If you find this research useful, please consider citing:
```BibTex
@inproceedings{lee2022cost,
  title={Cost Aggregation with Transformers for Sparse Correspondence},
  author={Lee, Seungjun and An, Seungjun and Hong, Sunghwan and Cho, Seokju and Nam, Jisu and Hong, Susung and Kim, Seungryong},
  booktitle={2022 IEEE International Conference on Consumer Electronics-Asia (ICCE-Asia)},
  pages={1--4},
  year={2022},
  organization={IEEE}
}
```
