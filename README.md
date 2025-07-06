# IMDB_BINARY_GNN_Classification

**Objective:** This dataset consists of graphs representing collaboration networks for movies. Nodes are actors, and edges represent co-occurrence in a movie. The task is to classify the movie genre (represented by each graph).

**Why IMDB-BINARY Dataset?** IMDB-BINARY is an unattributed graph dataset. Nodes (actors) have no inherent features, which means the GNN must learn representations solely from the graph structure (connectivity). This tests the model's ability to extract structural patterns.

**Dataset**: IMDB-BINARY from TUDataset (1000 graphs, 2 classes)  
**Model:** GCN with 3 convolution layers, global mean pooling, and a linear classification head  
**Implementation:** Based on PyTorch Geometric documentation  
**Result:** Achieved ~65–75% accuracy on the test set after 100 epochs  
