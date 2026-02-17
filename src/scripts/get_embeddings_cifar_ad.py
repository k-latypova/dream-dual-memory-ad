import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
from src.utils.KNN_utils import generate_outliers
import faiss
import os
from tqdm.auto import tqdm


FEATURES_NUM = 768

def get_embeddings(args):
    device = 'cuda' if args.gpu else 'cpu'
    if device == 'cuda':
        resources = faiss.StandardGpuResources()
        KNN_index = faiss.GpuIndexFlatL2(resources, FEATURES_NUM)
    else:
        KNN_index = faiss.IndexFlatL2(FEATURES_NUM)
    anchor = torch.from_numpy(np.load(args.anchor_file)).to(device)
    data_dict = torch.from_numpy(np.load(args.embed_file)).to(device)
    if not args.save_dir:
        args.save_dir = os.path.dirname(args.embed_file)        
    
    ID = F.normalize(data_dict, p=2, dim=1)

    sample_points = []
    for idx in range(100):
        distribution = MultivariateNormal(torch.zeros(FEATURES_NUM).to(device),
                                            torch.eye(FEATURES_NUM).to(device))
        negative_samples = distribution.rsample((1500,))
        sample_point, boundary_point = generate_outliers(ID,
                                                            input_index=KNN_index,
                                                            negative_samples=negative_samples,
                                                            ID_points_num=1,
                                                            K=args.K_in_knn,
                                                            select=args.select,
                                                            shift=args.inlier,
                                                            gaussian_variance=args.gaussian_mag,
                                                            sampling_ratio=1.0,
                                                            pic_nums=50,
                                                            depth=768,
                                                            device=device)
        sample_points.append(sample_point)
    sample_points = torch.cat(sample_points).view(-1, FEATURES_NUM)

    embeddings = [sample_points * anchor[args.class_idx].norm()]

    embeddings = torch.stack(embeddings).view(-1, FEATURES_NUM)
    
    filename = './inlier' if args.inlier else './outlier'
    filename += f'_npos_embed_noise_{args.gaussian_mag}_K_{args.K_in_knn}_select_{args.select}.npy'
    file_path = os.path.join(args.save_dir, filename)
    np.save(file_path, embeddings.cpu().numpy())
    print(f"Embeddings saved to {file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--inlier', type=int, default=0, help='Defines if we are generateing embeddings for inliers')
    parser.add_argument('--gaussian_mag', type=float, default=0.07, help='Magnitude of Gaussian noise for OOD detection')
    parser.add_argument('--K_in_knn', type=int, default=200, help='Number of nearest neighbors to consider for embeddings generation')
    parser.add_argument('--select', type=int, default=50, help='Number of selected points')
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use')
    parser.add_argument('--embed-file', type=str, required=True, help='File with the embeddings of the dataset')
    parser.add_argument('--anchor-file', type=str, required=True, help='File with the anchor that the latent space was conditioned on')
    parser.add_argument('--class_idx', type=int, default=100, help='Normal class idx')
    parser.add_argument('--save_dir', type=str, default=None)
    args = parser.parse_args()
    get_embeddings(args)
   
