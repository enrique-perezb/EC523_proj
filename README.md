# EC523_proj
ENGEC 523 Project: Age prediction models for AD classification

We are testing 3 different models. The implementation of each is in a different directory:
1. 3D CNN using weighted T1 MRI images.

2. GAT implemented on features extracted from T1 MRI images.<br>
  -preprocess_mri.ipynb -> create graphs from T1<br>
  -train_GAT.ipynb -> use graphs to train and predict age) 

3. GNN implemented on structural connectivity matrix comming from diffusion MRI.

It is important to note that the training and testing was done on SCC which means that the training and testing python files will compile in this repository as the data residents on the SCC. The data is too large to move to this repository. The 3 best models however have been saved and can be tested with outside MRIs that the user can provide.
