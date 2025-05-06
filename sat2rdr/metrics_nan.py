import numpy as np

def cal_csi(pred, target, threshold, epsilon=0): #1e-6):
    """
    Calculate CSI, POD, and FAR for a given prediction and target at a specific threshold.

    Args:
        pred (torch.Tensor): Predicted values.
        target (torch.Tensor): Ground truth values.
        threshold (float): Threshold to calculate binary classification metrics.
        epsilon (float): Small value to avoid division by zero.

    Returns:
        tuple: (csi, pod, far) for the given threshold.
    """
    # Convert predictions and targets to binary values based on the threshold
    pred_binary = (pred >= threshold).float()
    target_binary = (target >= threshold).float()

    # Calculate True Positives, False Positives, and False Negatives
    tp = ((pred_binary == 1) & (target_binary == 1)).sum().item()
    fp = ((pred_binary == 1) & (target_binary == 0)).sum().item()
    fn = ((pred_binary == 0) & (target_binary == 1)).sum().item()

    # Compute CSI, POD, and FAR, adding epsilon to denominators to prevent division by zero
    if ((tp+fp+fn) == 0) or ((tp+fn) == 0) or ((tp+fp) == 0):
        csi = np.nan
        pod = np.nan
        far = np.nan
    else:  
        csi = tp / (tp + fp + fn + epsilon)
        pod = tp / (tp + fn + epsilon)
        far = fp / (tp + fp + epsilon)

    return csi, pod, far

