## Segmenting Remote Sensing Anomalies at Instance Level via Anomaly Map-Guided Adaptation(TGRS 2024)

<p align="center">
  <img src=./figures/motivation.jpg width="600"> 
  <figcaption>Fig.1 Motivation of the proposed adaptation. (a) In existing models, each query is correlated with one specific category (e.g., the ship,
vehicle, and plane). (b) For anomaly detection task, the anomaly category is very special, where the anomaly objects have a larger intraclass variance even with
unseen objects. (c) To bridge this gap, our adaptations are designed to
adapt the queries and pixel embeddings to learn anomaly-aware content and features.</figcaption>
</p>

This is a PyTorch implementation of the [Instance Adaptation for RSAD](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10630639): 
```
@article{li2024segmenting,
  title={Segmenting Remote Sensing Anomalies at Instance-level via Anomaly Map Guided Adaptation},
  author={Li, Jingtao and Zhong, Yanfei and Zhao, Hengwei and Gao, Zhi and Wang, Xinyu},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2024},
  publisher={IEEE}
}
```

### Outline
1. This work extends the anomaly segmentation from the pixel-level to the instance level for object-centric, cleaner, and counting support detection results. The meta-architecture of query-based models is adopted to output the object masks for the end-to-end advantage.
2.  General adaptations are proposed, which extract the anomaly information from pixel-level anomaly map to guide the embedding refinement and query selection. The guidance injects the anomaly-aware representation to prevent the model from learning certain category anomalies.
3. A channel preprocessing strategy is designed to deal with the varying channels of input remote sensing images. The strategy extracts only three channels while keeping the anomaly information existing by computing the deviation distance explicitly.

<p align="center">
  <img src=./figures/framework.jpg width="600"> 
  <figcaption>Fig.1 Framework of proposed general instance adaptation.</figcaption>
</p>


### Preparation

1. Install required packages according to the requirements.txt.
2. Download the simulated training dataset (using simulation workflow from [UniADRS](https://github.com/Jingtao-Li-CVer/UniADRS)) and testing dataset from [here](). Update the dataset path in file `train_net.py (line 16-19)` and `tools/visualize_json_results.py (line 14-17)`.
3. Download the pre-trained [maskdino](https://github.com/IDEA-Research/detrex-storage/releases/download/maskdino-v0.1.0/maskdino_r50_50ep_300q_hid1024_3sd1_instance_maskenhanced_mask46.1ap_box51.5ap.pth) checkpoint and replace the 'WEIGHTS' key in config.yaml.

### Model Training and Testing

1. MMRAD is trained with simulated samples and can infer the real anomaly samples of different modalities directly.
2. Starting the training and quantitative testing process using the following one command.

```
python train_net.py
```
3. Visualizing the inferring results using script `tools/visualize_json_results.py`.

### Position of Implemented General Adaptation
1. **Anomaly generation**. Line 446-448 in `/maskdino_ada/modeling/pixel_decoder/maskdino_encoder.py`.
2. **Embedding refinement**. Line 386-395 in `/maskdino_ada/modeling/transformer_decoder/maskdino_decoder.py`.
3. **Query selection**. Line 429-432 in `/maskdino_ada/modeling/transformer_decoder/maskdino_decoder.py`.
4. **Varying channel preprocessing**. Line 401-409 in `/maskdino_ada/maskdino.py`


### Qualitative result  

 &emsp;The following are the exemplified localization results on three modalities. We found the instance-level supervision promoted the pixel-level results as well.

<p align="center">
  <img src=./figures/instance_res.jpg width="600"> 
</p>

<p align="center">
  <img src=./figures/pixel_res.jpg width="600"> 
</p>

