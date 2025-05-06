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
    tp = ((pred_binary == 1) & (target_binary == 1)).sum().item() #hits
    fp = ((pred_binary == 1) & (target_binary == 0)).sum().item() #false_alarms
    fn = ((pred_binary == 0) & (target_binary == 1)).sum().item() #misses
    tn = ((pred_binary == 0) & (target_binary == 0)).sum().item() #correct_negatives

    # Compute CSI, POD, and FAR, adding epsilon to denominators to prevent division by zero
    csi = tp / (tp + fp + fn + epsilon)
    pod = tp / (tp + fn + epsilon)
    far = fp / (tp + fp + epsilon)
    hss = (2 * (tp*tn - fp*fn)) / ((tp + fn)*(fn+tn) + (tp +fp)*(fp+tn) + epsilon)
    bias =(tp + fp) / (tp + fn + epsilon)
    
    return csi, pod, far

