# DGL & Pytorch_Recursive Neural Networks_Sentiment Classification
## Introduction
This project uses the DGL (Deep Graph Library) package to improve the training speed of Recursive Neural Networks(RvNN), which takes only 4-6 seconds every training epoch on the RTX server.
* References<br>
    
    * [1] Socher R, Lin C C, Manning C, et al. Parsing natural scenes and natural language with recursive neural networks[C] . ICML-11. 2011: 129-136. https://ai.stanford.edu/~ang/papers/icml11ParsingWithRecursiveNeuralNetworks.pdf
    * [2] Socher R, Perelygin A, Wu J, et al. Recursive deep models for semantic compositionality over a sentiment treebank[C]. EMNLP. 2013: 1631-1642. https://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf 
    * The Pytorch codes for RvNN by clli from NUSTM. https://github.com/NUSTM/PyTorch_RvNN  
* Dependencies<br>
    * Python >= 3.5<br>
    * PyTorch >= 0.4.1<br>
    * DLG >= 0.3<br>
* DGL(Deep Graph Library)<br>
    * Document: https://docs.dgl.ai/tutorials/basics/1_first.html
    * Github: https://github.com/dmlc/dgl
## Usage
python train.py
## Results
The RvNN model is trained on SST and the accuracy of predicting
fine-grained sentiment labels at all phrase lengths(All) or
full sentences(Root) is as follows:<br>

|               |  Acc_All  |  Acc_Root |     
| :-----------: | :-------: | :-------: |
|   Paper [2]   |   79.0    |   43.0    |
| My Recurrence |   75.71   |   47.96   |
