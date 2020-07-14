import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import Counter
import torch as th

def knn_eval(query_vectors, query_labels, gallery_vectors, gallery_labels, key='subject_inner_ids', k1=0, k2=5):
    nbrs = NearestNeighbors(n_neighbors=k2, algorithm='ball_tree').fit(gallery_vectors)
    hit_vector = []
    recall_vector = []
    pre_vector = []
    distances, indices = nbrs.kneighbors(query_vectors)
    # print("gallery_labels.shape:", gallery_labels.shape)
    target_labels = query_labels[key]
    known_labels = gallery_labels[key]
    label_counts = Counter(known_labels)
    for i in range(len(query_vectors)):
        target_label = target_labels[i]
        found_labels = np.array(known_labels)[indices[i]]
        found_labels = found_labels[k1:k2]
        found_label_count = Counter(found_labels)
        pre = found_label_count[target_label] / len(found_labels)
        pre_vector.append(pre)
        recall = found_label_count[target_label] / min(label_counts[target_label], len(found_labels))
        recall_vector.append(recall)
        try:
            if target_label in found_labels:
                hit_vector.append(1)
            else:
                hit_vector.append(0)
        except Exception as e:
            print(e)
            print("found_labels:", found_labels)
            hit_vector.append(0)
    hit_vector = np.asarray(hit_vector)
    hit_rate = 1.0 * hit_vector.sum() / len(hit_vector)
    mean_pre = np.asarray(pre_vector).mean()
    mean_recall = np.asarray(recall_vector).mean()
    return hit_rate, mean_pre, mean_recall

def hit_k(test_label, gallery_labels, similarity, k=5):
    top_k = similarity.topk(k, largest=True)
    indices = top_k.indices
    hit = 0
    for index in indices:
        if test_label == gallery_labels[index]:
            hit += 1
    if hit > 0:
        return 1
    else:
        return 0


def get_knn(probe_feature, gallery_features, k=5):
    gallery_features = th.from_numpy(gallery_features).to('cuda', dtype=th.float)
    probe_feature = th.from_numpy(probe_feature).to('cuda', dtype=th.float)
    dist = th.norm(gallery_features - probe_feature, dim=1, p=None)
    knn = dist.topk(k, largest=False)
    dist = knn.values
    indices = knn.indices
    return dist, indices

def knn_element(probe_feature, gallery_features, probe_label, gallery_labels, k=5):
    dist, indices = get_knn(probe_feature, gallery_features, k)
    indices = indices.cpu().numpy()
    found_labels = [gallery_labels[idx] for idx in indices]
    found_labels = np.asarray(found_labels)
    found_label_counts = Counter(found_labels)
    n_found = found_label_counts[probe_label]
    return n_found

def knn_evaluate(probe_features, probe_labels, gallery_features, gallery_labels, k=5, remove_first=False):
    n_len = len(probe_features)
    n_hit = 0
    pre = 0
    recall = 0
    label_counts = Counter(gallery_labels)

    for i in range(n_len):
        n_found = knn_element(probe_features[i], gallery_features, probe_labels[i], gallery_labels, k)
        n_list = k
        if remove_first:
            n_found -= 1
            n_list -= 1
        pre += n_found * 1.0 / n_list
        if min(label_counts[probe_labels[i]], n_list) == 0:
            recall += 1
        else:
            recall += n_found * 1.0 / min(label_counts[probe_labels[i]], n_list)
        if n_found > 0:
            n_hit += 1
    hit_rate = n_hit * 1.0 / n_len
    pre = pre * 1.0 / n_len
    recall = recall * 1.0 / n_len

    return hit_rate, pre, recall