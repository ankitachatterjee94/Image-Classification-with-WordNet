import torch
import numpy as np
import scipy.linalg, scipy.spatial.distance
from collections import OrderedDict
from class_hierarchy import ClassHierarchy


def get_Softlabels(hierarchy_file, beta=20, class_list=None, is_a=False, str_ids=False):

	id_type = str if str_ids else int
	hierarchy = ClassHierarchy.from_file(hierarchy_file, is_a_relations = is_a, id_type = id_type)


	if class_list is not None:
		with open(class_list) as class_file:
			unique_labels = list(OrderedDict((id_type(l.strip().split()[0]), None) for l in class_file if l.strip() != '').keys())

	else:
		unique_labels = [lbl for lbl in hierarchy.nodes if (lbl not in hierarchy.children) or (len(hierarchy.children[lbl]) == 0)]
		if not str_ids:
			unique_labels.sort()
	linear_labels = { lbl : i for i, lbl in enumerate(unique_labels) }

	sem_class_dist = np.zeros((len(unique_labels), len(unique_labels)))
	for i in range(len(unique_labels)):
		for j in range(i + 1, len(unique_labels)):
			sem_class_dist[i,j] = sem_class_dist[j,i] = hierarchy.lcs_height(unique_labels[i], unique_labels[j])
	soft_labels = np.zeros_like(sem_class_dist)
	for i in range(len(unique_labels)):
		soft_labels[i] = np.exp(-beta*sem_class_dist[i])
	
	return soft_labels

def make_all_soft_labels(distances, classes, hardness):
    distance_matrix = torch.Tensor([[distances[c1, c2] for c1 in classes] for c2 in classes])
    max_distance = torch.max(distance_matrix)
    distance_matrix /= max_distance
    soft_labels = torch.exp(-hardness * distance_matrix) / torch.sum(torch.exp(-hardness * distance_matrix), dim=0)
    return soft_labels


def make_batch_onehot_labels(target, num_classes, batch_size, gpu):
    onehot_labels = torch.zeros((batch_size, num_classes), dtype=torch.float32).cuda(gpu)
    for i in range(batch_size):
        onehot_labels[i, target[i]] = 1.0
    return onehot_labels


def make_batch_soft_labels(all_soft_labels, target, num_classes, batch_size, gpu):
    soft_labels = torch.zeros((batch_size, num_classes), dtype=torch.float32).cuda(gpu)
    for i in range(batch_size):
        this_label = all_soft_labels[:, target[i]]
        soft_labels[i, :] = this_label
    return soft_labels
