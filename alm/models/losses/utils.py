import torch
import torch.nn as nn

class MaskedConsistency:
    def __init__(self) -> None:
        self.loss = nn.MSELoss(reduction="mean")

    def __call__(self, pred, gt, mask):
        return self.loss(mask * pred, mask * gt)
    
    def __repr__(self):
        return self.loss.__repr__()
    
class MaskedVelocityConsistency:
    def __init__(self) -> None:
        self.loss = nn.MSELoss(reduction="mean")

    def __call__(self, pred, gt, mask):
        term1 = self.velocity(mask * pred)
        temr2 = self.velocity(mask * gt)
        return self.loss(term1, temr2)

    def velocity(self, term):
        velocity = term[:, 1:] - term[:, :-1]
        return velocity

    def __repr__(self):
        return self.loss.__repr__()


# flame_lmk_faces = [[2210, 2212, 2213],
#          [3060, 3059, 1962],
#          [3485, 3060, 1961],
#          [3382, 3384, 3381],
#          [3385, 3388, 3386],
#          [3387, 3389, 3390],
#          [3392, 3418, 3419],
#          [3415, 3395, 3393],
#          [3414, 3399, 3397],
#          [3634, 3595, 3598],
#          [3643, 3637, 3594],
#          [3588, 3587, 3583],
#          [3584, 3581, 3582],
#          [3742, 3580, 3577],
#          [2012, 3756,  566],
#          [2009, 2012, 2011],
#          [ 728,  731,  730],
#          [1983, 1984, 1985],
#          [3157, 3708, 3158],
#          [ 338, 3153,  335],
#          [3154, 3712, 3705],
#          [2179, 2178, 3684],
#          [ 674, 3851,  673],
#          [3863, 3868, 2135],
#          [2134,   27,   16],
#          [2138, 2139, 3865],
#          [ 572,  571,  570],
#          [2194, 3553, 3542],
#          [3561,  739, 3518],
#          [1757, 3521, 3501],
#          [3526, 3564, 1819],
#          [2748, 2746, 2750],
#          [2792, 2794, 2795],
#          [1692, 3556, 3507],
#          [1678, 1677, 1675],
#          [1612, 1618, 1610],
#          [2440, 2428, 2437],
#          [2383, 2453, 2495],
#          [2493, 3689, 2494],
#          [2509, 3631, 3632],
#          [2293, 2298, 2299],
#          [2333, 2296, 2295],
#          [3833, 3832, 1358],
#          [1342, 1343, 3855],
#          [1344, 1218, 1034],
#          [1182, 1175, 1154],
#          [ 955,  883,  884],
#          [ 881,  897,  896],
#          [2845, 2715, 2714],
#          [2849, 2813, 2850],
#          [2811, 2866, 2774],
#          [1657, 3543, 3546],
#          [1694, 1657, 1751],
#          [1734, 1735, 1696],
#          [1730, 1578, 1579],
#          [1774, 1795, 1796],
#          [1802, 1865, 1866],
#          [1850, 3506, 3503],
#          [2905, 2949, 2948],
#          [2899, 2898, 2881],
#          [2719, 2718, 2845],
#          [3533, 2786, 2785],
#          [3533, 3531, 2786],
#          [1669, 3533, 1668],
#          [1730, 1578, 1579],
#          [1826, 1848, 1849],
#          [3504, 3509, 2937],
#          [2938, 2937, 2928]] # extracted from the mica-trakcer code

# def flame_vertice_2_lmk(vertice, lmk_faces = flame_lmk_faces):
        
#     if len(vertice.shape) == 3: # if the vertices is resizes to (B, N, V * 3), reshape it back to (B, N, V, 3)
#         resized = True
#         B, N, _ = vertice.shape
#         vertice = vertice.reshape(B, N, -1, 3)
#         V = vertice.shape[-2]
#     else:
#         resized = False
#         B, N, V, _ = vertice.shape

#     _vertice = vertice.view(-1, V, 3) # (B * N, V, 3) -> (_B, V, 3)
#     _B = _vertice.shape[0]

#     lmk_faces = torch.tensor(lmk_faces,).clone() # (68, 3)

#     lmk_faces = lmk_faces.unsqueeze(0).expand(_B, -1, -1).to(vertice.device) # (68, 3) -> (_B, 68, 3)
#     lmk_faces += torch.arange(_B, device=vertice.device).view(-1, 1, 1) * V # (_B, 68, 3)
#     lmk_vertices = _vertice.reshape(-1, 3)[lmk_faces].view(_B, -1, 3, 3) # (_B, 68, 3, 3), here we have to use .reshape(-1, 3) to make sure the index is correct, view reports error
#     landmarks = lmk_vertices.mean(dim=-2) # (_B, 68, 3), every vertice in the face contributes to the landmark

#     if resized:
#         landmarks = landmarks.view(B, N, -1) # (B, N, 68 * 3)
#     else:
#         landmarks = landmarks.view(B, N, -1, 3) # (B, N, 68, 3)

#     return landmarks

# def cusum(data, threshold, drift):
#     """Cumulative sum algorithm (CUSUM) to detect abrupt changes in data."""
#     # Initialize variables
#     thres = torch.zeros_like(data)
#     # Compute the cumulative sum using torch.cumsum
#     for i in range(1, data.shape[1]):
#         # Update the cumulative sum
#         prev_thres = thres[:, i-1].unsqueeze(1)
#         delta = (data[:, i, :] - data[:, i-1, :]).abs().unsqueeze(1) #torch.norm(data[:, i, :] - data[:, i-1, :], dim=1).unsqueeze(1)
#         thres[:, i] = torch.max(torch.zeros_like(prev_thres), prev_thres + delta - threshold).squeeze(1)
#         # Update the threshold
#         threshold += drift

#     return thres

# the following is required by BIWI
import os
import numpy as np
import pickle

with open(os.path.join("datasets/biwi/regions", "lve.txt")) as f:
    maps = f.read().split(", ")
    mouth_map = [int(i) for i in maps]

with open(os.path.join("datasets/biwi/regions", "fdd.txt")) as f:
    maps = f.read().split(", ")
    upper_map = [int(i) for i in maps]

# open /home/zhiyuan_ma/code/FaceDiffusion/datasets/vocaset/FLAME_masks.pkl
with open(os.path.join("datasets/vocaset", "FLAME_masks.pkl"), "rb") as f:
    masks = pickle.load(f, encoding='latin1')
    vocaset_mouth_map = masks["lips"].tolist()
    vocaset_upper_map = masks["forehead"].tolist() + masks["eye_region"].tolist()
    vocaset_upper_map = list(set(vocaset_upper_map))


def vocaset_upper_face_variance(motion, ):
    L2_dis_upper = np.array([np.square(motion[:,v, :]) for v in vocaset_upper_map])
    L2_dis_upper = np.transpose(L2_dis_upper, (1,0,2))
    L2_dis_upper = np.sum(L2_dis_upper,axis=2)
    L2_dis_upper = np.std(L2_dis_upper, axis=0)
    motion_std = np.mean(L2_dis_upper)
    return torch.tensor(motion_std).float() #torch.from_numpy(motion_std).float()

def vocaset_mouth_distance(vertices_gt, vertices_pred):
    L2_dis = np.array([np.square(vertices_gt[:,v, :] - vertices_pred[:,v, :]) for v in vocaset_mouth_map])
    L2_dis = np.transpose(L2_dis, (1,0,2)) # (V, N, 3) -> (N, V, 3)
    L2_dis = np.sum(L2_dis, axis=2) # (N, V, 3) -> (N, V)
    L2_dis = np.max(L2_dis, axis=1) # (N, V) -> (N)
    return torch.tensor(L2_dis).float()

def biwi_upper_face_variance(motion, ):
    L2_dis_upper = np.array([np.square(motion[:,v, :]) for v in upper_map])
    L2_dis_upper = np.transpose(L2_dis_upper, (1,0,2))
    L2_dis_upper = np.sum(L2_dis_upper,axis=2)
    L2_dis_upper = np.std(L2_dis_upper, axis=0)
    motion_std = np.mean(L2_dis_upper)
    return torch.tensor(motion_std).float() #torch.from_numpy(motion_std).float()

def biwi_mouth_distance(vertices_gt, vertices_pred):
    L2_dis = np.array([np.square(vertices_gt[:,v, :] - vertices_pred[:,v, :]) for v in mouth_map])
    L2_dis = np.transpose(L2_dis, (1,0,2)) # (V, N, 3) -> (N, V, 3)
    L2_dis = np.sum(L2_dis, axis=2) # (N, V, 3) -> (N, V)
    L2_dis = np.max(L2_dis, axis=1) # (N, V) -> (N)
    return torch.tensor(L2_dis).float()

