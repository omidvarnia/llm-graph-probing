from absl import app, flags, logging
from multiprocessing import Process, Queue
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
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
    
    tokenizer, model = load_tokenizer_and_model(model_name, ckpt_step, gpu_id)

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
            for i in tqdm(range(0, len(input_texts), batch_size), position=rank, desc=f"Producer {rank}: Processing {len(input_texts)} texts"):
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
    with torch.no_grad():
        while True:
            batch = queue.get(block=True)
            if batch == "STOP":
                break
            hidden_states_layer_average, hidden_states, attention_mask, question_indices, labels, word2vec_embeddings, word_token_counts = batch
            for i, question_idx in enumerate(question_indices):
                sentence_attention_mask = attention_mask[i]

                p_dir_name = f"{p_save_path}/{question_idx}"
                os.makedirs(p_dir_name, exist_ok=True)
                
                # Save correctness label
                np.save(f"{p_dir_name}/label.npy", labels[i])

                layer_average_hidden_states = hidden_states_layer_average[:, i, sentence_attention_mask == 1]
                layer_average_corr = np.corrcoef(layer_average_hidden_states)
                # Handle NaN/Inf from zero-variance features
                layer_average_corr = np.nan_to_num(layer_average_corr, nan=0.0, posinf=0.0, neginf=0.0)
                np.save(f"{p_dir_name}/layer_average_corr.npy", layer_average_corr)

                layer_average_activation = layer_average_hidden_states[:, -1]
                np.save(f"{p_dir_name}/layer_average_activation.npy", layer_average_activation)

                layer_average_degree = np.abs(layer_average_corr).sum(axis=1)
                np.save(f"{p_dir_name}/layer_average_degree.npy", layer_average_degree)

                np.save(f"{p_dir_name}/word2vec_average.npy", word2vec_embeddings[i])
                np.save(f"{p_dir_name}/word2vec_token_count.npy", word_token_counts[i])

                per_layer_states = []
                for j, layer_idx in enumerate(layer_list):
                    layer_hidden_states = hidden_states[j, i]
                    sentence_hidden_states = layer_hidden_states[sentence_attention_mask == 1].T
                    per_layer_states.append(sentence_hidden_states)
                    corr = np.corrcoef(sentence_hidden_states)
                    # Handle NaN/Inf from zero-variance features or numerical instability
                    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
                    # Ensure correlations stay in [-1, 1] range
                    corr = np.clip(corr, -1.0, 1.0)
                    
                    degree = np.abs(corr).sum(axis=1)
                    np.save(f"{p_dir_name}/layer_{layer_idx}_degree.npy", degree)

                    # Save pre-threshold dense correlation (NPY + image)
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

                    # Compute thresholded dense correlation regardless of sparse flag
                    percentile_threshold = network_density * 100
                    density_tag = f"{int(round(network_density * 100)):02d}"
                    threshold = np.percentile(np.abs(corr), 100 - percentile_threshold)
                    corr_thresh = corr.copy()
                    corr_thresh[np.abs(corr_thresh) < threshold] = 0
                    np.fill_diagonal(corr_thresh, 1.0)

                    # Save post-threshold dense correlation (NPY + image)
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

                    # If sparse mode, also save sparse representation from thresholded matrix
                    if sparse:
                        corr_t = torch.from_numpy(corr_thresh)
                        edge_index, edge_attr = dense_to_sparse(corr_t)
                        edge_index = edge_index.numpy()
                        edge_attr = edge_attr.numpy()
                        np.save(f"{p_dir_name}/layer_{layer_idx}_sparse_{density_tag}_edge_index.npy", edge_index)
                        np.save(f"{p_dir_name}/layer_{layer_idx}_sparse_{density_tag}_edge_attr.npy", edge_attr)

                    activation = sentence_hidden_states[:, -1]
                    np.save(f"{p_dir_name}/layer_{layer_idx}_activation.npy", activation)

                    activation_avg = sentence_hidden_states.mean(-1)
                    np.save(f"{p_dir_name}/layer_{layer_idx}_activation_avg.npy", activation_avg)

                if aggregate_layers and len(per_layer_states) > 1:
                    combined_states = np.concatenate(per_layer_states, axis=0)
                    combined_corr = np.corrcoef(combined_states)

                    combined_degree = np.abs(combined_corr).sum(axis=1)
                    np.save(f"{p_dir_name}/layers_combined_degree.npy", combined_degree)

                    # Save combined pre-threshold dense correlation (NPY + image)
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

                    # Compute and save combined thresholded dense correlation (NPY + image)
                    percentile_threshold = network_density * 100
                    density_tag = f"{int(round(network_density * 100)):02d}"
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
                        edge_index = edge_index.numpy()
                        edge_attr = edge_attr.numpy()
                        np.save(f"{p_dir_name}/layers_combined_sparse_{density_tag}_edge_index.npy", edge_index)
                        np.save(f"{p_dir_name}/layers_combined_sparse_{density_tag}_edge_attr.npy", edge_attr)

                    combined_activation = combined_states[:, -1]
                    np.save(f"{p_dir_name}/layers_combined_activation.npy", combined_activation)

                    combined_activation_avg = combined_states.mean(-1)
                    np.save(f"{p_dir_name}/layers_combined_activation_avg.npy", combined_activation_avg)

    print(f"Worker {worker_idx} finished processing.")


def main(_):
    logging.info("="*60)
    logging.info("Computing LLM Neural Network Topology")
    logging.info("="*60)
    
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
    logging.info(f"Network density: {FLAGS.network_density}")
    logging.info(f"Sparse mode: {FLAGS.sparse}")
    logging.info(f"Batch size: {FLAGS.batch_size}")
    logging.info(f"Dataset fraction: {FLAGS.dataset_fraction}")
    logging.info(f"Output directory: {dir_name}")
    logging.info(f"Number of GPUs: {len(FLAGS.gpu_id)}")
    logging.info(f"Number of workers: {FLAGS.num_workers}")
    logging.info(f"Aggregate layers: {FLAGS.aggregate_layers}")
    
    layer_list = FLAGS.llm_layer

    queue = Queue()

    logging.info("\n" + "="*60)
    logging.info("Starting multiprocessing pipeline...")
    logging.info("="*60)
    
    # ===== START PRODUCER PROCESSES =====
    producers = []
    hf_model_name = hf_model_name_map.get(model_name, model_name)
    logging.info(f"Starting {len(FLAGS.gpu_id)} producer process(es)...")
    for i, gpu_id in enumerate(FLAGS.gpu_id):
        logging.info(f"  Producer {i} using GPU {gpu_id}")
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
    logging.info(f"\n" + "="*60)
    logging.info(f"Starting {num_workers} consumer worker(s)...")
    logging.info("="*60)
    for worker_idx in range(num_workers):
        p = Process(
            target=run_corr,
            args=(queue, layer_list, dir_name, worker_idx, FLAGS.sparse, FLAGS.network_density, FLAGS.aggregate_layers))
        p.start()
        consumers.append(p)

    logging.info("\n" + "="*60)
    logging.info("Processing dataset...")
    logging.info("="*60)
    logging.info(f"Waiting for {len(producers)} producer(s) to complete...")
    for i, producer in enumerate(producers):
        producer.join()
        logging.info(f"  ✓ Producer {i} terminated")
    logging.info("✓ All producers completed")
    
    logging.info(f"Sending STOP signal to {num_workers} consumer(s) and waiting for queue drain...")
    for _ in range(num_workers):
        queue.put("STOP")
    logging.info(f"Waiting for {num_workers} consumer(s) to complete processing the queue...")
    for i, consumer in enumerate(consumers):
        consumer.join()
        logging.info(f"  ✓ Consumer {i} terminated")
    logging.info("✓ All consumers completed")
    
    logging.info("\n" + "="*60)
    logging.info("✓ Neural topology computation completed successfully")
    logging.info("="*60)


if __name__ == "__main__":
    app.run(main)
