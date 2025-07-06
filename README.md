# IMDB_BINARY_GNN_Classification

**Objective:** This dataset consists of graphs representing collaboration networks for movies. Nodes are actors, and edges represent co-occurrence in a movie. The task is to classify the movie genre (represented by each graph).

**Why IMDB-BINARY Dataset?** IMDB-BINARY is an unattributed graph dataset. Nodes (actors) have no inherent features, which means the GNN must learn representations solely from the graph structure (connectivity). This tests the model's ability to extract structural patterns.

A common trick for unattributed graphs is to:

*  Use a constant feature: Every node gets a feature vector of all 1s (or some other constant value).
*  or use an identity matrix as features: Each node i gets a one-hot encoded vector where the i-th element is 1 and others are 0. This essentially uses the node's unique identity as its feature. This is often the default behavior if num_node_features is 0 and we pass x=None to GCNConv.
**Dataset**: IMDB-BINARY from TUDataset (1000 graphs, 2 classes)
**Model:** GCN with 3 convolution layers, global mean pooling, and a linear classification head
**Implementation:** Based on PyTorch Geometric documentation
**Result:** Achieved ~65–75% accuracy on the test set after 100 epochs
