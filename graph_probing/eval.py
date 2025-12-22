from absl import app, flags
import os
import sys
from pathlib import Path
import logging

# Configure logging (suppress timestamps/levels as run_logged already adds them)
logging.basicConfig(level=logging.INFO, format='[EVAL] %(message)s')

# Get main directory from environment or use current directory
main_dir = Path(os.environ.get('MAIN_DIR', '.'))

# Add project root to path to ensure utils package can be found
project_root = Path(__file__).parent.parent
# Ensure project root is first in path, even before current directory
project_root_str = str(project_root)
if project_root_str in sys.path:
    sys.path.remove(project_root_str)
sys.path.insert(0, project_root_str)

from setproctitle import setproctitle

import numpy as np
import torch
torch.set_default_dtype(torch.float32)

from graph_probing.dataset import get_brain_network_dataloader, get_brain_network_linear_dataloader
from graph_probing.utils import eval_model, eval_model_space

from utils.constants import hf_model_name_map
from utils.model_utils import get_num_nodes
from utils.probing_model import GCNProbe as GCNRegressor, MLPProbe as MLPRegressor

flags.DEFINE_string("dataset", "openwebtext", "The name of the dataset.")
flags.DEFINE_string("probe_input", "activation", "The input type for linear probing: activation, activation_avg or corr.")
flags.DEFINE_float("density", 1.0, "The density of the input.")
flags.DEFINE_boolean("from_sparse_data", False, "Whether to use sparse data.")
flags.DEFINE_string("llm_model_name", "gpt2", "The name of the LLM model.")
flags.DEFINE_integer("ckpt_step", -1, "The checkpoint step.")
flags.DEFINE_integer("llm_layer", 0, "The layer of the LLM model.")
flags.DEFINE_integer("batch_size", 16, "The batch size.")
flags.DEFINE_integer("eval_batch_size", 16, "The evaluation batch size.")
flags.DEFINE_integer("num_workers", 4, "Number of workers.")
flags.DEFINE_integer("prefetch_factor", 4, "Prefetch factor.")
flags.DEFINE_boolean("nonlinear_activation", True, "Whether to use nonlinear activation.")
flags.DEFINE_integer("num_channels", 32, "The number of channels in GNN probes.")
flags.DEFINE_integer("num_layers", 1, "The number of GNN layers.")
flags.DEFINE_float("dropout", 0.0, "The dropout rate.")
flags.DEFINE_float("test_set_ratio", 0.2, "The size of the test set.")
flags.DEFINE_boolean("in_memory", False, "In-memory dataset.")
flags.DEFINE_integer("gpu_id", 0, "The GPU ID.")
flags.DEFINE_integer("seed", 42, "The random seed.")
FLAGS = flags.FLAGS


def main(_):
    logging.info("="*60)
    logging.info("STEP 4: Probe Evaluation Pipeline")
    logging.info("="*60)
    logging.info(f"Dataset: {FLAGS.dataset}")
    logging.info(f"Model: {FLAGS.llm_model_name} (checkpoint: {FLAGS.ckpt_step})")
    logging.info(f"Layer: {FLAGS.llm_layer}")
    logging.info(f"Probe input: {FLAGS.probe_input}")
    logging.info(f"Network density: {FLAGS.density}")
    logging.info(f"Batch size: {FLAGS.batch_size}")
    logging.info(f"GPU ID: {FLAGS.gpu_id}")

    if FLAGS.gpu_id == -1:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{FLAGS.gpu_id}")

    logging.info(f"\n[1/4] Loading dataset...")
    hf_model_name = hf_model_name_map[FLAGS.llm_model_name]
    if FLAGS.dataset == "art":
        dataset_filename = str(main_dir / "st_data/art.csv")
        target = "decade"
        num_output = 1
        normalize_targets = True
    elif FLAGS.dataset == "world_place":
        dataset_filename = str(main_dir / "st_data/world_place.csv")
        target = ["lat", "lon"]
        num_output = 2
        normalize_targets = True
    else:
        revision = "main" if FLAGS.ckpt_step == -1 else f"step{FLAGS.ckpt_step}"
        if hf_model_name.startswith("EleutherAI") and revision != "main":
            dataset_filename = str(main_dir / f"data/graph_probing/{FLAGS.dataset}-10k-{FLAGS.llm_model_name}-{revision}.csv")
        else:
            dataset_filename = str(main_dir / f"data/graph_probing/{FLAGS.dataset}-10k-{FLAGS.llm_model_name}.csv")
        target = "perplexities"
        num_output = 1
        normalize_targets = True
    logging.info(f"✓ Dataset loaded from: {dataset_filename}")

    if FLAGS.num_layers > 0:
        logging.info(f"\n[2/4] Loading GCN model with {FLAGS.num_layers} layers...")
        _, test_data_loader = get_brain_network_dataloader(
            dataset_filename,
            network_density=FLAGS.density,
            from_sparse_data=FLAGS.from_sparse_data,
            llm_model_name=FLAGS.llm_model_name,
            ckpt_step=FLAGS.ckpt_step,
            llm_layer=FLAGS.llm_layer,
            batch_size=FLAGS.batch_size,
            eval_batch_size=FLAGS.eval_batch_size,
            num_workers=FLAGS.num_workers,
            prefetch_factor=FLAGS.prefetch_factor,
            test_set_ratio=FLAGS.test_set_ratio,
            in_memory=FLAGS.in_memory,
            seed=FLAGS.seed,
            target=target,
            dataset_name=FLAGS.dataset,
            normalize_targets=normalize_targets,
        )

        model = GCNRegressor(
            num_nodes=get_num_nodes(FLAGS.llm_model_name, FLAGS.llm_layer, FLAGS.probe_input),
            hidden_channels=FLAGS.num_channels,
            num_layers=FLAGS.num_layers,
            nonlinear_activation=FLAGS.nonlinear_activation,
            dropout=FLAGS.dropout,
            num_output=num_output,
        ).to(device)
        model_type = "GCN"
    else:
        logging.info(f"\n[2/4] Loading MLP model (linear)...")
        _, test_data_loader = get_brain_network_linear_dataloader(
            dataset_filename,
            FLAGS.probe_input,
            feature_density=FLAGS.density,
            llm_model_name=FLAGS.llm_model_name,
            ckpt_step=FLAGS.ckpt_step,
            llm_layer=FLAGS.llm_layer,
            batch_size=FLAGS.batch_size,
            eval_batch_size=FLAGS.eval_batch_size,
            num_workers=FLAGS.num_workers,
            prefetch_factor=FLAGS.prefetch_factor,
            test_set_ratio=FLAGS.test_set_ratio,
            seed=FLAGS.seed,
            target=target,
            dataset_name=FLAGS.dataset,
            normalize_targets=normalize_targets,
        )
        model = MLPRegressor(
            num_nodes=get_num_nodes(FLAGS.llm_model_name, FLAGS.llm_layer, FLAGS.probe_input),
            hidden_channels=FLAGS.num_channels,
            num_layers=-FLAGS.num_layers,
            num_output=num_output,
        ).to(device)
        model_type = "MLP"

    logging.info(f"✓ Model created ({model_type})")
    logging.info(f"\n[3/4] Loading trained weights...")
    if FLAGS.ckpt_step == -1:
        save_model_name = f"{FLAGS.llm_model_name}/{FLAGS.dataset}"
    else:
        save_model_name = f"{FLAGS.llm_model_name}_step{FLAGS.ckpt_step}/{FLAGS.dataset}"
    model_save_path = main_dir / f"saves/graph_probing/{save_model_name}/layer_{FLAGS.llm_layer}/best_model_density-{FLAGS.density}_dim-{FLAGS.num_channels}_hop-{FLAGS.num_layers}_input-{FLAGS.probe_input}.pth"
    model.load_state_dict(torch.load(str(model_save_path), map_location=device, weights_only=True))
    logging.info(f"✓ Weights loaded from: {model_save_path}")

    logging.info(f"\n[4/4] Running evaluation on test set...")
    if FLAGS.dataset == "world_place":
        all_y, all_pred = eval_model_space(model, test_data_loader, device, num_layers=FLAGS.num_layers)
    else:
        all_y, all_pred = eval_model(model, test_data_loader, device, num_layers=FLAGS.num_layers)
    results = np.vstack((all_y.T, all_pred.T)) if FLAGS.dataset == "world_place" else np.vstack((all_y, all_pred))
    logging.info(f"✓ Evaluation completed")
    logging.info("="*60)
    logging.info("✓ STEP 4 Complete: Probe Evaluation Finished")
    logging.info("="*60)
    np.save(str(main_dir / f"saves/graph_probing/{save_model_name}/layer_{FLAGS.llm_layer}/results_density-{FLAGS.density}_dim-{FLAGS.num_channels}_hop-{FLAGS.num_layers}_input-{FLAGS.probe_input}.npy"), results)


if __name__ == "__main__":
    setproctitle("llm graph probing")
    app.run(main)
