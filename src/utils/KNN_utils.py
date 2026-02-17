import numpy as np 
import torch
import faiss.contrib.torch_utils
import torch.nn.functional as F

def KNN_dis_search_decrease(target, index, K=50, select=1, shift = False):
    """
        Perform K-Nearest Neighbors distance search and return the indices and distances
        of the top selected neighbors with the smallest distances.
            target (torch.Tensor): The target tensor to search for neighbors.
            index (faiss.Index): The FAISS index to perform the search.
            K (int, optional): The number of nearest neighbors to search for. Defaults to 50.
            select (int, optional): The number of top neighbors to select based on distance. Defaults to 1.
            shift (bool, optional): If True, select the top neighbors with the smallest distances.
                                    If False, select the top neighbors with the largest distances. Defaults to False.
        Returns:
            tuple: A tuple containing:
                - minDidx (torch.Tensor): The indices of the selected top neighbors.
                - k_th_distance (torch.Tensor): The distances of the selected top neighbors.
    """

    target_norm = torch.norm(target, p=2, dim=1, keepdim=True)
    normed_target = target / target_norm

    distance, _  = index.search(normed_target, K)
    k_th_distance = distance[:, -1]
    if shift:
        k_th_distance, minDidx = torch.topk(-k_th_distance, select)
    else:
        k_th_distance, minDidx = torch.topk(k_th_distance, select)
    return minDidx, k_th_distance


def KNN_dis_search(target, index, K=50, num_points=10, shift = False, length=2000, depth=342):
    """
    Perform K-Nearest Neighbors (KNN) distance search on the given target tensor using a FAISS index.
    This function normalizes the target tensor and performs a KNN search to find the K nearest neighbors
    for each point in the target tensor. It then selects a specified number of points based on the k-th 
    distance and returns these points from the target tensor.
        Args:
            target (torch.Tensor): The target tensor to search for nearest neighbors. Each row represents a point.
            index (faiss.Index): The FAISS index to perform the search. It should be pre-trained and compatible with the target tensor.
            num_points (int, optional): The number of points to return based on the k-th distance. Default is 10.
            shift (bool, optional): Whether to shift the distances for top-k selection. If True, the distances are negated before selection. Default is False.
            length (int, optional): The length parameter for reshaping the k-th distance tensor. This affects the shape of the tensor used in top-k selection. Default is 2000.
            depth (int, optional): The depth parameter (currently unused). Default is 342.
        Returns:
            torch.Tensor: A tensor containing the selected points from the target tensor based on the k-th distance.
        """
    target_norm = torch.norm(target, p=2, dim=1, keepdim=True)
    normed_target = target / target_norm

    distance, _  = index.search(normed_target, K)
    k_th_distance = distance[:, -1]
    k_th = k_th_distance.view(length, -1)
    if shift:
        k_th_distance, minDidx = torch.topk(-k_th, num_points, dim=0)
    else:
        k_th_distance, minDidx = torch.topk(k_th, num_points, dim=0)
    minDidx = minDidx.squeeze()
    point_list = []
    if len(minDidx.shape) == 1:
        minDidx = minDidx.reshape(-1, 1)
    for i in range(minDidx.shape[1]):
        point_list.append(i * length + minDidx[:, i])

    print(f"KNN_dis_search return tensor: {target[torch.cat(point_list)].shape}")

    return target[torch.cat(point_list)]

def generate_outliers(ID, input_index, negative_samples, ID_points_num=2, K=20, select=1, shift=False, 
                      gaussian_variance=0.1, sampling_ratio=1.0, pic_nums=30, depth=342, device='cuda'):
    """
Summary
The generate_outliers function uses KNN search to identify boundary data points from the input data, 
adds Gaussian noise to negative samples, and performs another KNN search to generate outlier points. 
The function is designed to work efficiently on a specified device (CPU or GPU) 
and allows for various customizations through its parameters.
    Parameters:
    ID (torch.Tensor): The input data tensor.
    input_index (Index): The index object used for KNN search.
    negative_samples (torch.Tensor): The tensor containing negative samples.
    ID_poins_num (int, optional): Number of ID points. Default is 2.
    K (int, optional): Number of nearest neighbors to consider. Default is 20.
    select (int, optional): Number of selected points. Default is 1.
    shift (bool, optional): Whether to apply shift. Default is False.
    gaussian_variance (float, optional): Variance for Gaussian noise. Default is 0.1.
    sampling_ratio (float, optional): Ratio of samples to use. Default is 1.0.
    pic_nums (int, optional): Number of pictures. Default is 30.
    depth (int, optional): Depth parameter. Default is 342.
    device (str, optional): Device to use ('cuda' or 'cpu'). Default is 'cuda'.
    Returns:
    tuple: A tuple containing:
        - point (torch.Tensor): The generated outlier points.
        - boundary_data (torch.Tensor): The boundary data points.
    """
    length = negative_samples.shape[0]
    data_norm = torch.norm(ID, p=2, dim=1, keepdim=True)
    print(f"ID device: {ID.device}")
    normed_data = ID / data_norm
    rand_int = np.random.choice(normed_data.shape[0], 
                                int(normed_data.shape[0] * sampling_ratio),
                                replace=False)
    index = input_index
    index.add(normed_data[rand_int])
    minD_idx, _ = KNN_dis_search_decrease(ID, index, K, select, shift)
    boundary_data = ID[minD_idx]

    minD_idx = minD_idx[np.random.choice(select, int(pic_nums), replace=False)]
    data_point_list = torch.cat([ID[i:i+1].repeat(length, 1) for i in minD_idx])
    negative_samples_cov = gaussian_variance * negative_samples.repeat(pic_nums, 1).to(device)
    negative_samples_list = F.normalize(negative_samples_cov + data_point_list, p=2, dim=1)
    point = KNN_dis_search(negative_samples_list, index, K, ID_points_num, shift, length, depth)
    index.reset()

    return point, boundary_data
    