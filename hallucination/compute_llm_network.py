from absl import app, flags, logging
from multiprocessing import Process, Queue, set_start_method
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set multiprocessing start method to 'spawn' for CUDA compatibility
# Must be done before any CUDA operations
try:
    set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

from tqdm import tqdm
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gensim.downloader as gensim_downloader
from gensim.utils import tokenize

from hallucination.utils import format_prompt
from utils.constants import hf_model_name_map
from utils.model_utils import load_tokenizer_and_model

main_dir = Path(os.environ.get('MAIN_DIR', '.'))

flags.DEFINE_enum(
    "dataset_name",
    "truthfulqa",
    ["truthfulqa", "halueval", "medhallu", "helm"],
    "The dataset to process."
)
flags.DEFINE_string("llm_model_name", "qwen2.5-0.5b", "The name of the LLM model.")
flags.DEFINE_integer("ckpt_step", -1, "The checkpoint step.")
flags.DEFINE_multi_integer("llm_layer", [12], "The layer list.")
flags.DEFINE_integer("batch_size", 16, "Batch size.")
flags.DEFINE_multi_integer("gpu_id", [0, 1], "The GPU ID.")
flags.DEFINE_integer("num_workers", 40, "Number of processes for computing networks.")
flags.DEFINE_boolean("resume", False, "Resume from the last generation.")
flags.DEFINE_boolean("sparse", False, "Whether to generate sparse networks.")
flags.DEFINE_float("network_density", 1.0, "The density of the network.")
flags.DEFINE_boolean("random", False, "Whether to generate random data.")
flags.DEFINE_boolean("aggregate_layers", False, "If multiple layers are provided, concatenate them before computing a combined FC matrix in addition to per-layer FCs.")
flags.DEFINE_float("dataset_fraction", 1.0, "Fraction of dataset to use (0.1-1.0 where 1.0 = all data)")
FLAGS = flags.FLAGS

_WORD2VEC_MODEL = None


def get_word2vec_model():
    global _WORD2VEC_MODEL
    if _WORD2VEC_MODEL is None:
        _WORD2VEC_MODEL = gensim_downloader.load("word2vec-google-news-300")
    return _WORD2VEC_MODEL


def compute_word2vec_embedding(text, model):
    # Tokenize text while keeping case and removing punctuation accents
    tokens = list(tokenize(text, lowercase=False, deacc=True))
    tokens_in_vocab = [token for token in tokens if token in model.key_to_index]
    if not tokens_in_vocab:
        embedding = np.zeros(model.vector_size, dtype=np.float32)
    else:
        embedding = model.get_mean_vector(tokens_in_vocab).astype(np.float32)
    return embedding


def run_llm(
    rank,
    num_producers,
    queue,
    dataset_filename,
    model_name,
    ckpt_step,
    gpu_id,
    batch_size,
    layer_list,
    resume,
    p_save_path,
    random
):
    # Initialize CUDA/ROCm in worker process
    import os
    if gpu_id >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        os.environ['HIP_VISIBLE_DEVICES'] = str(gpu_id)
        os.environ['ROCR_VISIBLE_DEVICES'] = str(gpu_id)
    
    logging.info(f"[Producer {rank}] Initializing on GPU {gpu_id}...")
    
    tokenizer, model = load_tokenizer_and_model(model_name, ckpt_step, gpu_id)
    
    # Report device configuration AFTER model is loaded (CUDA is initialized by model loading)
    import torch
    if torch.cuda.is_available():
        try:
            actual_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(actual_device)
            is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
            device_type = "ROCm/HIP" if is_rocm else "CUDA"
            logging.info(f"[Producer {rank}] Using {device_type} Device {actual_device}: {device_name}")
        except:
            pass
    else:
        logging.info(f"[Producer {rank}] Using CPU (no GPU available)")

    df = pd.read_csv(dataset_filename)
    num_questions = len(df)
    original_input_questions = df["question"].tolist()[rank::num_producers]
    original_input_answers = df["answer"].tolist()[rank::num_producers]
    original_labels = df["label"].tolist()[rank::num_producers]
        
    if not resume:
        input_texts = format_prompt(original_input_questions, original_input_answers, model_name, tokenizer)
        labels = original_labels
        question_indices = list(range(rank, num_questions, num_producers))
    else:
        input_texts = []
        labels = []
        question_indices = []
        for i, question_idx in enumerate(range(rank, num_questions, num_producers)):
            if not os.path.exists(f"{p_save_path}/{question_idx}"):
                input_texts.extend(format_prompt([original_input_questions[i]], [original_input_answers[i]], model_name, tokenizer))
                labels.append(original_labels[i])
                question_indices.append(question_idx)

    if len(input_texts) > 0:
        tokenizer.pad_token = tokenizer.eos_token
        logging.info(f"[Producer {rank}] Loading word2vec model for semantic embeddings...")
        word2vec_model = get_word2vec_model()
        logging.info(f"[Producer {rank}] Extracting word2vec embeddings for {len(input_texts)} texts (this may take a moment)...")
        sample_word2vec_embeddings = np.stack([compute_word2vec_embedding(text, word2vec_model) for text in input_texts], axis=0)
        answer_tokens = [list(tokenize(str(answer))) for answer in original_input_answers]
        sample_word_token_counts = [[len(tokens)] for tokens in answer_tokens]
        logging.info(f"[Producer {rank}] Starting LLM forward passes on {len(input_texts)} texts with batch_size={batch_size}...")

        with torch.no_grad():
            for i in range(0, len(input_texts), batch_size):
                inputs = tokenizer(input_texts[i:i+batch_size], padding=True, truncation=False, return_tensors="pt")
                batch_input_ids = inputs["input_ids"]
                batch_attention_mask = inputs["attention_mask"]

                model_output = model(
                    input_ids=batch_input_ids.to(model.device),
                    attention_mask=batch_attention_mask.to(model.device),
                    output_hidden_states=True,
                )
                
                # Get correctness label
                batch_labels = labels[i:i+batch_input_ids.shape[0]]

                batch_hidden_states_all_layers = torch.stack(model_output.hidden_states[1:]).cpu().float().numpy() # (num_layers, batch_size, seq_length, hidden_size)
                batch_hidden_states_layer_average = batch_hidden_states_all_layers.mean(axis=-1) # (num_layers, batch_size, seq_length)
                # Treat provided layer_list as 0-based indices
                num_layers_avail = batch_hidden_states_all_layers.shape[0]
                zero_based = [int(l) for l in layer_list if 0 <= int(l) < num_layers_avail]
                if not zero_based:
                    # Fallback: use last layer
                    zero_based = [num_layers_avail - 1]
                batch_hidden_states = batch_hidden_states_all_layers[zero_based]
                if random:
                    batch_hidden_states = np.random.rand(*batch_hidden_states.shape)
                
                actual_batch_size = batch_hidden_states.shape[1]
                batch_question_indices = question_indices[i:i+actual_batch_size]

                batch_word2vec_embeddings = sample_word2vec_embeddings[i:i+actual_batch_size]
                batch_word_token_counts = sample_word_token_counts[i:i+actual_batch_size]
                queue.put((batch_hidden_states_layer_average, batch_hidden_states, batch_attention_mask.numpy(), batch_question_indices, batch_labels, batch_word2vec_embeddings, batch_word_token_counts))
        logging.info(f"[Producer {rank}] Finished queueing all {len(input_texts)} texts. Waiting for consumers to drain the queue...")


def run_corr(queue, layer_list, p_save_path, worker_idx, sparse=False, network_density=1.0, aggregate_layers=False):
    from torch_geometric.utils import dense_to_sparse
    import json
    
    def check_question_complete(p_dir_name, layer_list, sparse=False, network_density=1.0, aggregate_layers=False):
        """Check if all required files for a question already exist."""
        required_files = [
            f"{p_dir_name}/label.npy",
            f"{p_dir_name}/layer_average_corr.npy",
            f"{p_dir_name}/layer_average_activation.npy",
            f"{p_dir_name}/layer_average_degree.npy",
            f"{p_dir_name}/word2vec_average.npy",
            f"{p_dir_name}/word2vec_token_count.npy",
        ]
        
        # Check per-layer files
        density_tag = f"{int(round(network_density * 100)):02d}"
        for layer_idx in layer_list:
            required_files.extend([
                f"{p_dir_name}/layer_{layer_idx}_degree.npy",
                f"{p_dir_name}/layer_{layer_idx}_corr.npy",
            ])
            if sparse:
                required_files.extend([
                    f"{p_dir_name}/layer_{layer_idx}_sparse_{density_tag}_edge_index.npy",
                    f"{p_dir_name}/layer_{layer_idx}_sparse_{density_tag}_edge_attr.npy",
                ])
            required_files.extend([
                f"{p_dir_name}/layer_{layer_idx}_activation.npy",
                f"{p_dir_name}/layer_{layer_idx}_activation_avg.npy",
            ])
        
        # Check combined layers if enabled
        if aggregate_layers and len(layer_list) > 1:
            required_files.extend([
                f"{p_dir_name}/layers_combined_degree.npy",
                f"{p_dir_name}/layers_combined_corr.npy",
            ])
            if sparse:
                required_files.extend([
                    f"{p_dir_name}/layers_combined_sparse_{density_tag}_edge_index.npy",
                    f"{p_dir_name}/layers_combined_sparse_{density_tag}_edge_attr.npy",
                ])
            required_files.extend([
                f"{p_dir_name}/layers_combined_activation.npy",
                f"{p_dir_name}/layers_combined_activation_avg.npy",
            ])
        
        # Check if ALL required files exist
        for fpath in required_files:
            if not os.path.exists(fpath):
                return False
        return True
    
    processed_count = 0
    skipped_count = 0
    excluded_count = 0
    excluded_indices = []
    class_stats = {
        0: {"sum": 0.0, "sum_sq": 0.0, "count": 0, "sum_means": 0.0, "sum_stds": 0.0, "q_count": 0},
        1: {"sum": 0.0, "sum_sq": 0.0, "count": 0, "sum_means": 0.0, "sum_stds": 0.0, "q_count": 0},
    }
    
    with torch.no_grad():
        while True:
            batch = queue.get(block=True)
            if batch == "STOP":
                break
            hidden_states_layer_average, hidden_states, attention_mask, question_indices, labels, word2vec_embeddings, word_token_counts = batch
            
            for i, question_idx in enumerate(question_indices):
                p_dir_name = f"{p_save_path}/{question_idx}"

                # Already complete?
                if os.path.exists(p_dir_name) and check_question_complete(p_dir_name, layer_list, sparse, network_density, aggregate_layers):
                    skipped_count += 1
                    if skipped_count % 50 == 0 or skipped_count <= 5:
                        logging.info(f"[Worker {worker_idx}] Skipped question {question_idx} (already processed). Total skipped: {skipped_count}")
                    continue

                sentence_attention_mask = attention_mask[i]

                # Compute layer-average corr and validate
                layer_average_hidden_states = hidden_states_layer_average[:, i, sentence_attention_mask == 1]
                if layer_average_hidden_states.size == 0 or layer_average_hidden_states.shape[1] < 2:
                    excluded_count += 1
                    excluded_indices.append(int(question_idx))
                    continue
                layer_average_corr = np.corrcoef(layer_average_hidden_states)
                if not np.isfinite(layer_average_corr).all():
                    excluded_count += 1
                    excluded_indices.append(int(question_idx))
                    continue

                # Accumulate class stats from off-diagonal of layer-average corr
                lbl = int(labels[i])
                n = layer_average_corr.shape[0]
                if n > 1:
                    mask = ~np.eye(n, dtype=bool)
                    vals = layer_average_corr[mask]
                    vals = vals[np.isfinite(vals)]
                    if vals.size > 0:
                        class_stats[lbl]["sum"] += float(vals.sum())
                        class_stats[lbl]["sum_sq"] += float((vals ** 2).sum())
                        class_stats[lbl]["count"] += int(vals.size)
                        class_stats[lbl]["sum_means"] += float(vals.mean())
                        class_stats[lbl]["sum_stds"] += float(vals.std())
                        class_stats[lbl]["q_count"] += 1

                # Validate per-layer corrs before writing any files
                per_layer_states = []
                per_layer_corrs = []
                invalid = False
                for j, layer_idx in enumerate(layer_list):
                    layer_hidden_states = hidden_states[j, i]
                    sentence_hidden_states = layer_hidden_states[sentence_attention_mask == 1].T
                    if sentence_hidden_states.shape[1] < 2:
                        invalid = True
                        break
                    per_layer_states.append(sentence_hidden_states)
                    corr = np.corrcoef(sentence_hidden_states)
                    if not np.isfinite(corr).all():
                        invalid = True
                        break
                    per_layer_corrs.append((layer_idx, corr))
                if invalid:
                    excluded_count += 1
                    excluded_indices.append(int(question_idx))
                    continue

                # Combined layers validation if requested
                combined_corr = None
                combined_states = None
                if aggregate_layers and len(per_layer_states) > 1:
                    combined_states = np.concatenate(per_layer_states, axis=0)
                    if combined_states.shape[1] < 2:
                        excluded_count += 1
                        excluded_indices.append(int(question_idx))
                        continue
                    combined_corr = np.corrcoef(combined_states)
                    if not np.isfinite(combined_corr).all():
                        excluded_count += 1
                        excluded_indices.append(int(question_idx))
                        continue

                # Passed validation: now write outputs
                os.makedirs(p_dir_name, exist_ok=True)
                np.save(f"{p_dir_name}/label.npy", labels[i])

                np.save(f"{p_dir_name}/layer_average_corr.npy", layer_average_corr)
                layer_average_activation = layer_average_hidden_states[:, -1]
                np.save(f"{p_dir_name}/layer_average_activation.npy", layer_average_activation)
                layer_average_degree = np.abs(layer_average_corr).sum(axis=1)
                np.save(f"{p_dir_name}/layer_average_degree.npy", layer_average_degree)
                np.save(f"{p_dir_name}/word2vec_average.npy", word2vec_embeddings[i])
                np.save(f"{p_dir_name}/word2vec_token_count.npy", word_token_counts[i])

                percentile_threshold = network_density * 100
                density_tag = f"{int(round(network_density * 100)):02d}"
                for (layer_idx, corr), sentence_hidden_states in zip(per_layer_corrs, per_layer_states):
                    degree = np.abs(corr).sum(axis=1)
                    np.save(f"{p_dir_name}/layer_{layer_idx}_degree.npy", degree)
                    np.save(f"{p_dir_name}/layer_{layer_idx}_corr.npy", corr)
                    try:
                        vmax = np.percentile(np.abs(corr), 99) if corr.size else 1.0
                        plt.figure(figsize=(6, 5))
                        plt.imshow(corr, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
                        plt.colorbar()
                        plt.title(f"Layer {layer_idx} Corr (pre-threshold)")
                        plt.tight_layout()
                        plt.savefig(f"{p_dir_name}/layer_{layer_idx}_corr.png")
                        plt.close()
                    except Exception:
                        pass
                    threshold = np.percentile(np.abs(corr), 100 - percentile_threshold)
                    corr_thresh = corr.copy()
                    corr_thresh[np.abs(corr_thresh) < threshold] = 0
                    np.fill_diagonal(corr_thresh, 1.0)
                    np.save(f"{p_dir_name}/layer_{layer_idx}_corr_thresh_{density_tag}.npy", corr_thresh)
                    try:
                        vmax_t = np.percentile(np.abs(corr_thresh), 99) if corr_thresh.size else 1.0
                        plt.figure(figsize=(6, 5))
                        plt.imshow(corr_thresh, cmap="RdBu_r", vmin=-vmax_t, vmax=vmax_t)
                        plt.colorbar()
                        plt.title(f"Layer {layer_idx} Corr (threshold={network_density})")
                        plt.tight_layout()
                        plt.savefig(f"{p_dir_name}/layer_{layer_idx}_corr_thresh_{density_tag}.png")
                        plt.close()
                    except Exception:
                        pass
                    if sparse:
                        corr_t = torch.from_numpy(corr_thresh)
                        edge_index, edge_attr = dense_to_sparse(corr_t)
                        np.save(f"{p_dir_name}/layer_{layer_idx}_sparse_{density_tag}_edge_index.npy", edge_index.numpy())
                        np.save(f"{p_dir_name}/layer_{layer_idx}_sparse_{density_tag}_edge_attr.npy", edge_attr.numpy())
                    activation = sentence_hidden_states[:, -1]
                    np.save(f"{p_dir_name}/layer_{layer_idx}_activation.npy", activation)
                    activation_avg = sentence_hidden_states.mean(-1)
                    np.save(f"{p_dir_name}/layer_{layer_idx}_activation_avg.npy", activation_avg)

                if aggregate_layers and combined_states is not None and combined_corr is not None:
                    combined_degree = np.abs(combined_corr).sum(axis=1)
                    np.save(f"{p_dir_name}/layers_combined_degree.npy", combined_degree)
                    np.save(f"{p_dir_name}/layers_combined_corr.npy", combined_corr)
                    try:
                        vmax_c = np.percentile(np.abs(combined_corr), 99) if combined_corr.size else 1.0
                        plt.figure(figsize=(6, 5))
                        plt.imshow(combined_corr, cmap="RdBu_r", vmin=-vmax_c, vmax=vmax_c)
                        plt.colorbar()
                        plt.title("Layers Combined Corr (pre-threshold)")
                        plt.tight_layout()
                        plt.savefig(f"{p_dir_name}/layers_combined_corr.png")
                        plt.close()
                    except Exception:
                        pass
                    threshold = np.percentile(np.abs(combined_corr), 100 - percentile_threshold)
                    combined_corr_thresh = combined_corr.copy()
                    combined_corr_thresh[np.abs(combined_corr_thresh) < threshold] = 0
                    np.fill_diagonal(combined_corr_thresh, 1.0)
                    np.save(f"{p_dir_name}/layers_combined_corr_thresh_{density_tag}.npy", combined_corr_thresh)
                    try:
                        vmax_ct = np.percentile(np.abs(combined_corr_thresh), 99) if combined_corr_thresh.size else 1.0
                        plt.figure(figsize=(6, 5))
                        plt.imshow(combined_corr_thresh, cmap="RdBu_r", vmin=-vmax_ct, vmax=vmax_ct)
                        plt.colorbar()
                        plt.title(f"Layers Combined Corr (threshold={network_density})")
                        plt.tight_layout()
                        plt.savefig(f"{p_dir_name}/layers_combined_corr_thresh_{density_tag}.png")
                        plt.close()
                    except Exception:
                        pass
                    if sparse:
                        combined_corr_t = torch.from_numpy(combined_corr_thresh)
                        edge_index, edge_attr = dense_to_sparse(combined_corr_t)
                        np.save(f"{p_dir_name}/layers_combined_sparse_{density_tag}_edge_index.npy", edge_index.numpy())
                        np.save(f"{p_dir_name}/layers_combined_sparse_{density_tag}_edge_attr.npy", edge_attr.numpy())
                    combined_activation = combined_states[:, -1]
                    np.save(f"{p_dir_name}/layers_combined_activation.npy", combined_activation)
                    combined_activation_avg = combined_states.mean(-1)
                    np.save(f"{p_dir_name}/layers_combined_activation_avg.npy", combined_activation_avg)

                processed_count += 1
                if processed_count % 100 == 0 or processed_count <= 10:
                    logging.info(f"[Worker {worker_idx}] Processed question {question_idx}. Total processed: {processed_count}, Skipped: {skipped_count}, Excluded(NaN): {excluded_count}")
    
    # Write per-worker summaries for later merge
    try:
        with open(os.path.join(p_save_path, f"exclusions_worker_{worker_idx}.txt"), "w") as f:
            for idx in excluded_indices:
                f.write(str(idx) + "\n")
        with open(os.path.join(p_save_path, f"summary_worker_{worker_idx}.json"), "w") as f:
            json.dump({
                "worker": worker_idx,
                "counts": {"processed": processed_count, "skipped": skipped_count, "excluded_nan": excluded_count},
                "class_stats": class_stats,
            }, f)
    except Exception as e:
        logging.warning(f"[Worker {worker_idx}] Failed to write per-worker summary: {e}")

    logging.info(f"[Worker {worker_idx}] Finished processing. Total processed: {processed_count}, Total skipped: {skipped_count}, Total excluded(NaN): {excluded_count}")
    print(f"Worker {worker_idx} finished processing.")


def main(_):
    # Suppress PyTorch/PyG warnings
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    # Configure absl logging format with datetime stamps
    logging.use_absl_handler()
    import logging as stdlib_logging
    absl_handler = logging.get_absl_handler()
    absl_handler.setFormatter(stdlib_logging.Formatter('%(asctime)s %(filename)s:%(lineno)d - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    
    logging.info("\n\n" + "="*10)
    logging.info("STEP 2: NEURAL NETWORK COMPUTATION (FC MATRICES)")
    logging.info("="*10)
    
    # Report main process device configuration
    import torch
    logging.info(f"Main Process Device Configuration:")
    logging.info(f"  PyTorch CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
        device_type = "ROCm/HIP" if is_rocm else "CUDA"
        logging.info(f"  Device Type: {device_type}")
        logging.info(f"  Available GPUs: {torch.cuda.device_count()}")
        for i in range(min(torch.cuda.device_count(), len(FLAGS.gpu_id))):
            logging.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        logging.info(f"  Device Type: CPU (no GPU detected)")
    
    logging.info(f"Dataset: {FLAGS.dataset_name}")
    logging.info(f"Model: {FLAGS.llm_model_name}")
    logging.info("="*10 + "\n")
    
    # ===== LOAD CONFIGURATION =====
    dataset_filename = main_dir / "data/hallucination" / f"{FLAGS.dataset_name}.csv"
    dataset_dir = main_dir / "data/hallucination" / FLAGS.dataset_name
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Load dataset and apply fraction early for network computation
    import pandas as pd
    full_data = pd.read_csv(dataset_filename)
    original_data_size = len(full_data)
    sampled_question_ids = None
    if FLAGS.dataset_fraction < 1.0:
        # Random sampling without fixed random_state for true randomness
        sampled_data = full_data.sample(frac=FLAGS.dataset_fraction).reset_index(drop=True)
        sampled_question_ids = set(sampled_data['question_id'].unique())
        logging.info(f"Using {len(sampled_question_ids)} unique questions (fraction={FLAGS.dataset_fraction} of {original_data_size})")
        
        # Sanity check
        expected_size = int(original_data_size * FLAGS.dataset_fraction)
        actual_size = len(sampled_data)
        tolerance = max(1, int(0.01 * original_data_size))  # 1% tolerance
        if abs(actual_size - expected_size) <= tolerance:
            logging.info(f"✓ Sanity check: Dataset size ({actual_size}) is consistent with fraction ({FLAGS.dataset_fraction}) of original ({original_data_size})")
        else:
            logging.warning(f"⚠ Sanity check: Dataset size ({actual_size}) deviates from expected ({expected_size})")
    else:
        logging.info(f"Using all questions in dataset (fraction={FLAGS.dataset_fraction})")

    model_name = FLAGS.llm_model_name
    # Sanitize model name by replacing '/', '-', and '.' with '_' for filesystem paths
    sanitized_model_name = model_name.replace('/', '_').replace('-', '_').replace('.', '_')
    if FLAGS.ckpt_step == -1:
        model_dir = sanitized_model_name
    else:
        model_dir = f"{sanitized_model_name}_step{FLAGS.ckpt_step}"
    dir_name = os.path.join(dataset_dir, model_dir)
    os.makedirs(dir_name, exist_ok=True)

    logging.info(f"Dataset: {FLAGS.dataset_name}")
    logging.info(f"Model: {model_name}")
    logging.info(f"Checkpoint step: {FLAGS.ckpt_step}")
    logging.info(f"Layers: {FLAGS.llm_layer}")
    logging.info(f"Network density: {FLAGS.network_density:.1%}")
    logging.info(f"Sparse mode: {FLAGS.sparse}")
    logging.info(f"Batch size: {FLAGS.batch_size}")
    logging.info(f"Dataset fraction: {FLAGS.dataset_fraction:.1%}")
    logging.info(f"Output directory: {dir_name}")
    logging.info(f"Number of GPUs: {len(FLAGS.gpu_id)}")
    logging.info(f"Number of workers: {FLAGS.num_workers}")
    logging.info(f"Aggregate layers: {FLAGS.aggregate_layers}")
    
    layer_list = FLAGS.llm_layer

    queue = Queue()

    logging.info("="*10)
    logging.info("GPU ALLOCATION & MULTIPROCESSING CONFIGURATION")
    logging.info("="*10)
    
    # ===== START PRODUCER PROCESSES =====
    producers = []
    hf_model_name = hf_model_name_map.get(model_name, model_name)
    
    logging.info(f"{'─'*10}")
    logging.info(f"Producer Processes (LLM Forward Pass):")
    logging.info(f"  Number of producers: {len(FLAGS.gpu_id)}")
    for i, gpu_id in enumerate(FLAGS.gpu_id):
        logging.info(f"  Producer {i} → GPU {gpu_id}")
    logging.info(f"{'─'*10}\n")
    
    for i, gpu_id in enumerate(FLAGS.gpu_id):
        p = Process(
            target=run_llm,
            args=(
                i,
                len(FLAGS.gpu_id),
                queue,
                dataset_filename,
                hf_model_name,
                FLAGS.ckpt_step,
                gpu_id,
                FLAGS.batch_size,
                layer_list,
                FLAGS.resume,
                dir_name,
                FLAGS.random,
            )
        )
        p.start()
        producers.append(p)

    num_workers = FLAGS.num_workers
    consumers = []
    
    logging.info(f"{'─'*10}")
    logging.info(f"Consumer Processes (FC Matrix Computation):")
    logging.info(f"  Number of consumers: {num_workers}")
    logging.info(f"  Task: Compute and save functional connectivity matrices")
    logging.info(f"{'─'*10}\n")
    for worker_idx in range(num_workers):
        p = Process(
            target=run_corr,
            args=(queue, layer_list, dir_name, worker_idx, FLAGS.sparse, FLAGS.network_density, FLAGS.aggregate_layers))
        p.start()
        consumers.append(p)


    logging.info("="*10)
    logging.info("PROCESSING PHASE: PRODUCERS & CONSUMERS")
    logging.info("="*10)
    logging.info(f"Waiting for {len(producers)} producer(s) to complete forward passes...")
    
    producer_errors = []
    for i, producer in enumerate(producers):
        producer.join()
        if producer.exitcode != 0:
            error_msg = f"Producer {i} failed with exit code {producer.exitcode}"
            logging.error(f"  ✗ {error_msg}")
            producer_errors.append(error_msg)
        else:
            logging.info(f"  ✓ Producer {i} completed (GPU)")
    
    if producer_errors:
        logging.error("✗ One or more producers failed. Stopping pipeline.")
        logging.error("\n".join(producer_errors))
        # Send stop signals to consumers before exiting
        for _ in range(num_workers):
            queue.put("STOP")
        for consumer in consumers:
            consumer.join()
        import sys
        sys.exit(1)
    
    logging.info("✓ All producers completed successfully")
    
    logging.info(f"\nSending completion signal to {num_workers} consumer(s)...")
    for _ in range(num_workers):
        queue.put("STOP")
    
    logging.info(f"Waiting for {num_workers} consumer(s) to complete FC computation...")
    consumer_errors = []
    for i, consumer in enumerate(consumers):
        consumer.join()
        if consumer.exitcode != 0:
            error_msg = f"Consumer {i} failed with exit code {consumer.exitcode}"
            logging.error(f"  ✗ {error_msg}")
            consumer_errors.append(error_msg)
        else:
            logging.info(f"  ✓ Consumer {i} completed (CPU)")
    
    if consumer_errors:
        logging.error("✗ One or more consumers failed. Stopping pipeline.")
        logging.error("\n".join(consumer_errors))
        import sys
        sys.exit(1)
    
    logging.info("✓ All consumers completed successfully")
    
    logging.info("="*10)
    logging.info("STEP 2 COMPLETE: Neural Topology Computation")
    logging.info("="*10)
    logging.info("✓ FC matrices computed and saved successfully")
    logging.info("="*10 + "\n\n")


if __name__ == "__main__":
    app.run(main)
