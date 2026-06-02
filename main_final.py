import argparse
import warnings

from GNNs import *
from models import Encoder
from src_final import *
from data_loader import *
from utils import get_logger

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    # Initialize argument parser for all configurable hyperparameters
    parser = argparse.ArgumentParser('GeoGAD')

    # ---------------------- Basic experimental setup ----------------------
    parser.add_argument('--data', type=str, default='aids',
                        help='Dataset name')
    parser.add_argument('--device', type=str, default='cuda:0', help='Computation device')
    parser.add_argument('--anom_type', type=int, default=0, choices=[0, 1],
                        help='Anomaly construction setting, 0 or 1')
    parser.add_argument('--diff', type=float, default=0.1,
                        help='Training set contamination ratio')

    # ---------------------- GNN encoder hyperparameters ----------------------
    parser.add_argument('--dim', type=int, default=32, help='Hidden dimension size of GNN encoder')
    parser.add_argument('--n_layers', type=int, default=3, help='Number of convolution layers in encoder')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for GNN pre-training')
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='Weight coefficient for adjacency matrix positive weight')
    parser.add_argument('--tau', type=float, default=0.8, help='Hyperparameter for geometry gating loss')
    parser.add_argument('--dp', type=float, default=0.3, help='Dropout probability inside encoder layers')
    parser.add_argument('--w_decay', type=float, default=1e-6, help='Weight decay for pre-training optimizer')

    # ---------------------- Training epoch & checkpoint save settings ----------------------
    parser.add_argument('--pre_epoch', type=int, default=100, help='Total pre-training epochs for GNN encoders')
    parser.add_argument('--pre_save_interval', type=int, default=20,
                        help='Epoch interval to save pre-trained checkpoint')
    parser.add_argument('--cls_epoch', type=int, default=100,
                        help='Training epochs for one-class autoencoder classifier')
    parser.add_argument('--cls_save_interval', type=int, default=20, help='Save interval for classifier training')
    parser.add_argument('--cls_wd', type=float, default=1e-4, help='Weight decay for downstream classifier optimizer')

    # ---------------------- Downstream one-class classifier search space ----------------------
    parser.add_argument('--r4_dim_list', nargs='+', type=int, default=[32, 64, 128],
                        help='Search list of hidden dimension for MLP autoencoder')
    parser.add_argument('--r4_lr_list', nargs='+', type=float, default=[1e-2, 1e-3, 1e-4],
                        help='Search list of learning rate for MLP autoencoder')
    parser.add_argument('--n_feat', type=int, default=4, help='Input feature dimension of one-class autoencoder')

    # ---------------------- Experiment repetition setting ----------------------
    parser.add_argument('--num_sim', type=int, default=5,
                        help='Repeat times of independent experiment with different random seeds')

    args = parser.parse_args()

    # Unpack all arguments into local variables
    data = args.data
    device = args.device
    anom_type = args.anom_type
    diff = args.diff
    lr = args.lr
    dim = args.dim
    n_layer = args.n_layers
    gamma = args.gamma
    tau = args.tau
    dp = args.dp
    w_decay = args.w_decay
    pretrain_epochs = args.pre_epoch
    pretrain_save_interval = args.pre_save_interval
    cls_epochs = args.cls_epoch
    cls_save_interval = args.cls_save_interval
    cls_wd = args.cls_wd
    r4_dim_list = args.r4_dim_list
    r4_lr_list = args.r4_lr_list
    N_FEAT = args.n_feat
    num_simulation = args.num_sim

    # Create log folder and logger instance
    log_root = 'log'
    os.makedirs(log_root, exist_ok=True)
    log_file = f"{log_root}/{data}.log"
    logger = get_logger(log_file)
    logger.info(f"Running configuration args: {args}")

    # Create checkpoint save folder
    cpt_path = args.data
    os.makedirs(cpt_path, exist_ok=True)

    # Replace diff=0 with 'easy' for clean training set (no contamination)
    if diff == 0.0:
        diff = "easy"

    # Load TU graph dataset and get default batch size
    dataset, BSize = load_torch_dataset(data, device)
    # Split dataset into train/valid/test with specified anomaly setting & contamination ratio
    S = create_train_valid_test(dataset, diff, anom_type)
    # Preprocess adjacency matrix and positive sample weights for reconstruction loss
    adj, adj_pos_weights = prepare_data(dataset, device, gamma)
    # Get original node feature dimension from first graph sample
    n_feat = dataset[0].x.shape[1]

    # Only reserve AUROC & AUPRC metrics
    ValidAUROC = np.zeros((num_simulation, pretrain_save_interval + 1))
    ValidAUPRC = np.zeros((num_simulation, pretrain_save_interval + 1))
    TestAUROC = np.zeros((num_simulation, pretrain_save_interval + 1))
    TestAUPRC = np.zeros((num_simulation, pretrain_save_interval + 1))

    print(f"Total number of graphs in {data}: {len(dataset)}")

    # Loop over multiple independent random-seed simulations
    for K in range(num_simulation):
        # Fix random seed for reproducibility
        torch.manual_seed(K)
        torch.random.manual_seed(K)
        np.random.seed(K)

        # Unpack sample index and label from pre-defined split
        train_idxs = S[K][0][0]
        train_labels = S[K][0][1]
        # Extract indices of pure normal samples only for pre-training
        train_norm_idxs = [idx for idx, label in zip(S[K][0][0], S[K][0][1]) if label == 1]
        valid_idxs = S[K][1][0]
        valid_labels = S[K][1][1]
        test_idxs = S[K][2][0]
        test_labels = S[K][2][1]

        # Initialize Euclidean-space GNN Encoder (without hyperbolic curvature)
        encoder_euc = Encoder(
            num_features=n_feat,
            hidden_units=dim,
            num_layers=n_layer,
            dropout=dp,
            mlp_layers=2,
            train_eps=False,
            is_encoder=True,
            use_kappa=False,
            kappa_value=-1.0
        ).to(device)

        # Initialize Hyperbolic-space GNN Encoder (with fixed negative curvature)
        encoder_hyp = Encoder(
            num_features=n_feat,
            hidden_units=dim,
            num_layers=n_layer,
            dropout=dp,
            mlp_layers=2,
            train_eps=False,
            is_encoder=True,
            use_kappa=True,
            kappa_value=-1.0
        ).to(device)

        # Geometry gating network to fuse Euclidean & Hyperbolic representations
        gate_net = GeometryGate(in_dim=8, hidden_dim=32, out_mode="fuse").to(device)

        # Edge reconstruction decoder heads for two encoders
        edge_decoder_euc = MLP_Decoder(int(n_layer * dim), dim, dim).to(device)
        edge_decoder_hyp = MLP_Decoder(int(n_layer * dim), dim, dim).to(device)
        # Node feature reconstruction decoder heads for two encoders
        feature_decoder_euc = MLP_Decoder(int(n_layer * dim), dim, n_feat).to(device)
        feature_decoder_hyp = MLP_Decoder(int(n_layer * dim), dim, n_feat).to(device)

        # Pre-training manager for unsupervised graph representation learning
        gnn_trainer = MUSE_representation_learning(datasets=dataset, device=device, labels=adj,
                                                   labels_pos_weights=adj_pos_weights)

        # Pre-train Euclidean encoder with reconstruction objective
        _, trained_parameters_euc = gnn_trainer.train_euc(model=encoder_euc, feature_head=feature_decoder_euc,
                                                          edge_head=edge_decoder_euc,
                                                          saving_interval=pretrain_save_interval,
                                                          train_idxs=train_norm_idxs, lr=lr,
                                                          weight_decay=w_decay,
                                                          epochs=pretrain_epochs, batch_size=BSize,
                                                          return_loss=True, seed=K, pth_path=cpt_path)
        # Pre-train Hyperbolic encoder with reconstruction objective
        _, trained_parameters_hyp = gnn_trainer.train_hyp(model=encoder_hyp, feature_head=feature_decoder_hyp,
                                                          edge_head=edge_decoder_hyp,
                                                          saving_interval=pretrain_save_interval,
                                                          train_idxs=train_norm_idxs, lr=lr,
                                                          weight_decay=w_decay,
                                                          epochs=pretrain_epochs, batch_size=BSize,
                                                          return_loss=True, seed=K, pth_path=cpt_path)

        # Iterate over all saved pre-trained checkpoint epochs
        for p_iter in range(len(trained_parameters_hyp)):
            # Load saved checkpoint weights for Euclidean branch
            curP = [
                torch.load(f'{cpt_path}/euc_model_state_gating_{K}.pth'),
                torch.load(f'{cpt_path}/euc_feature_head_state_gating_{K}.pth'),
                torch.load(f'{cpt_path}/euc_edge_head_state_gating_{K}.pth')
            ]
            # Load saved checkpoint weights for Hyperbolic branch
            curP_hyp = [
                torch.load(f'{cpt_path}/hyp_model_state_gating_{K}.pth'),
                torch.load(f'{cpt_path}/hyp_feature_head_state_gating_{K}.pth'),
                torch.load(f'{cpt_path}/hyp_edge_head_state_gating_{K}.pth')
            ]

            # Initialize one-class classification trainer with dual-branch encoders + gating module
            ae_trainer = MUSE_oneclass_classification(
                model1=encoder_euc, feature_encoder1=feature_decoder_euc, edge_encoder1=edge_decoder_euc,
                model2=encoder_hyp, feature_encoder2=feature_decoder_hyp, edge_encoder2=edge_decoder_hyp,
                gate_net=gate_net, datasets=dataset, device=device, labels=adj, pos_weights=adj_pos_weights,
                B_size=BSize
            )

            # Buffer to store best validation AUROC & AUPRC only
            valid_best_auroc = test_best_auroc = 0
            valid_best_auprc = test_best_auprc = 0

            # Grid search over downstream autoencoder classifier hyperparameters
            for r4_dim in r4_dim_list:
                for r4_lr in r4_lr_list:
                    torch.manual_seed(K)
                    torch.random.manual_seed(K)
                    np.random.seed(K)

                    # Build one-class MLP autoencoder classifier
                    MLP_autoencoder = AutoEncoder(in_dim=N_FEAT, hid_dim=r4_dim,
                                                  n_layers=3, drop_p=0.0).to(device)

                    # Train classifier, only fetch AUROC & AUPRC, discard unused metrics
                    (p_valid_auroc, p_valid_ap, p_test_auroc, p_test_ap) = ae_trainer.train_MLP_CL(
                        encoder_param=curP, encoder_param_hyp=curP_hyp, classifier=MLP_autoencoder,
                        train_idxs=train_norm_idxs,
                        train_all_idxs=train_idxs, valid_idxs=valid_idxs,
                        valid_labels=valid_labels,
                        test_idxs=test_idxs, test_labels=test_labels,
                        lr=r4_lr, epochs=cls_epochs, saving_interval=cls_save_interval, w_decay=cls_wd, seed=K, tau=tau)

                    # Update best metric if current validation performance is superior
                    if p_valid_auroc > valid_best_auroc:
                        valid_best_auroc = p_valid_auroc
                        test_best_auroc = p_test_auroc
                    if p_valid_ap > valid_best_auprc:
                        valid_best_auprc = p_valid_ap
                        test_best_auprc = p_test_ap

            # Save best searched result of current checkpoint iteration into metric array
            ValidAUROC[K, p_iter] = valid_best_auroc
            ValidAUPRC[K, p_iter] = valid_best_auprc
            TestAUROC[K, p_iter] = test_best_auroc
            TestAUPRC[K, p_iter] = test_best_auprc

            # Log intermediate results (only AUROC & AUPRC)
            logger.info(
                "Simulation:{0} | P_iter: {1} | Valid AUROC: {2:.4f} | Valid AUPRC: {3:.4f} "
                "| Test AUROC: {4:.4f} | Test AUPRC: {5:.4f}".format(
                    K, p_iter, valid_best_auroc, valid_best_auprc, test_best_auroc, test_best_auprc))

    # Select optimal checkpoint index via maximum average validation performance
    idx_best_auroc = np.argmax(np.mean(ValidAUROC, axis=0))
    idx_best_auprc = np.argmax(np.mean(ValidAUPRC, axis=0))

    # Output final averaged test performance with standard deviation
    logger.info("=" * 60)
    logger.info(f"Hyperparams | LR:{lr:.6f} | HiddenDim:{dim} | Layer:{n_layer} | Gamma:{gamma} | Tau:{tau}")
    logger.info(f"Dataset: {data} | AnomType:{anom_type} | ContamDiff:{diff}")
    logger.info("Final Test Results (selected by best validation):")
    logger.info("AUROC: Mean={0:.4f}, STD={1:.4f}".format(np.mean(TestAUROC[:, idx_best_auroc]),
                                                          np.std(TestAUROC[:, idx_best_auroc])))
    logger.info("AUPRC: Mean={0:.4f}, STD={1:.4f}\n\n".format(np.mean(TestAUPRC[:, idx_best_auprc]),
                                                              np.std(TestAUPRC[:, idx_best_auprc])))
