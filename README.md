# EC523_proj
ENGEC 523 Project: Age prediction models for AD classification

The goal of this project is to explore age prediction models based on brain MRI images. We used different levels of interpretability and multiple modalities. For future work, we aim to use these age prediction models to 

We are testing 3 different models. The implementation of each is in a different directory:
1. 3D CNN using weighted T1 MRI images: 3D_CNN<br>
   -file.py -> description<br>
   -file2.py -> description<br>

2. GAT implemented on features extracted from T1 MRI images: T1_GAT<br>
   -brain_graph.py -> Class with all graph information per MRI<br>
   -create_dataset.py -> Class to make dataset for GAT<br>
   -get_hcp_graphs.py -> Go from segmentation maps to graphs. Calls brain_graph.py<br>
   -preprocess_data.py -> Go from T1w MRIs to segmented maps.<br>
   -graph_attention_model.py -> Main class for GAT implementation.<br>
   -train_GAT.ipynb -> Jupyter notebook with training, validation, testing, and ROI identification.<br>

3. GNN implemented on structural connectivity matrix comming from diffusion MRI: SC_GNN<br>
   -00.tract.sh -> tractography structural connectivty matrix generator <br>
   -GNN.py -> GNN model <br>
   -train_GNN.py -> trains the GNN <br>
   -test.py -> test the GNN <br>
   -Heatmap.py -> creates the ROI heatnaos for important regions <br>

Training and testing was done on BU's SCC. Data files and system setup need to be set up separately to fully run all pipelines, as they are not available in thie repository. The three best models have been saved and are available under the "models" folder.
