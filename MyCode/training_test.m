features = semantic_feat;
num_classes = 9;

predicted_labels = clustering_with_labels(features, labels, num_classes);
compute_accuracy(predicted_labels,labels);