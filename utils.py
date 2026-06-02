import logging
import numpy as np
import scipy.sparse as sp
import torch

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def get_logger(log_filename):
    """
    创建一个新的 Logger，并绑定到指定的日志文件。
    """
    logger = logging.getLogger(log_filename)  # 使用文件名作为 Logger 名称
    logger.setLevel(logging.DEBUG)

    # 避免重复添加 Handler
    if logger.hasHandlers():
        logger.handlers.clear()

    # 设置 FileHandler
    file_handler = logging.FileHandler(log_filename, encoding="gbk")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # 设置 StreamHandler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # 添加 Handler 到 Logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
