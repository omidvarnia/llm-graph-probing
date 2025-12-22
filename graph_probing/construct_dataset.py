from absl import app, flags
import itertools
import os
import sys
from pathlib import Path
import logging

# Configure logging (suppress timestamps/levels as run_logged already adds them)
logging.basicConfig(level=logging.INFO, format='[CONSTRUCT_DATASET] %(message)s')

# Get main directory from environment or use current directory
main_dir = Path(os.environ.get('MAIN_DIR', '.'))

# Add project root to path to ensure utils package can be found
project_root = Path(__file__).parent.parent
# Ensure project root is first in path, even before current directory
project_root_str = str(project_root)
if project_root_str in sys.path:
    sys.path.remove(project_root_str)
sys.path.insert(0, project_root_str)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from tqdm import tqdm

from datasets import load_dataset
from evaluate import load
perplexity_revised = load("graph_probing/perplexity_revised.py", module_type="metric")
import numpy as np
import pandas as pd
import torch.multiprocessing as mp
from transformers import AutoTokenizer

from utils.constants import hf_model_name_map

flags.DEFINE_string("dataset", "openwebtext", "The name of the dataset.")
flags.DEFINE_string("llm_model_name", "gpt2", "The name of the LLM model.")
flags.DEFINE_integer("ckpt_step", -1, "The checkpoint step.")
flags.DEFINE_integer("batch_size", 32, "Batch size.")
flags.DEFINE_multi_integer("gpu_id", [0, 1], "The GPU ID.")
flags.DEFINE_string("output_dir", str(main_dir / "data/hallucination"), "Directory to save the constructed dataset.")
FLAGS = flags.FLAGS


def split_examples(examples):
    text = examples["text"]
    sentences = text.split("\n\n")

    segments = []
    current_segment = []
    current_word_count = 0

    for sentence in sentences:
        sentence_word_count = len(sentence.split())
        # If current_segment is not empty and adding the sentence would exceed the limit,
        # finish the current segment and start a new one.
        if current_segment and (current_word_count + sentence_word_count > 800):
            segments.append("\n\n".join(current_segment))
            current_segment = [sentence]
            current_word_count = sentence_word_count
        else:
            current_segment.append(sentence)
            current_word_count += sentence_word_count

    # Append any remaining sentences as the last segment.
    if current_segment:
        segments.append("\n\n".join(current_segment))

    return {"sentences": segments}


def run_ppl(rank, queue, model_name, revision, gpu_id, batch_size, all_sentences, p_sentence_indices):
    logging.info(f"[GPU {gpu_id[rank]}] Starting perplexity computation for rank {rank}")
    sentences = [all_sentences[i] for i in p_sentence_indices[rank]]
    logging.info(f"[GPU {gpu_id[rank]}] Loading tokenizer for {model_name}...")
    if model_name.startswith("EleutherAI"):
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    logging.info(f"[GPU {gpu_id[rank]}] Tokenizing {len(sentences)} sentences...")
    tokenized_sentences = tokenizer(sentences, padding=False, truncation=False)
    filtered_sentences = []
    logging.info(f"[GPU {gpu_id[rank]}] Filtering sentences by token length (256-1024)...")
    for idx, sentence in enumerate(tqdm(sentences, desc=f"GPU {gpu_id[rank]}: Filtering", position=rank, leave=True)):
        if 256 <= len(tokenized_sentences["input_ids"][idx]) < 1024:
            filtered_sentences.append(sentence)
    logging.info(f"[GPU {gpu_id[rank]}] Filtered {len(filtered_sentences)}/{len(sentences)} sentences. Computing perplexities...")
    results = perplexity_revised.compute(
        model_id=model_name,
        revision=revision,
        add_start_token=False,
        batch_size=batch_size,
        predictions=filtered_sentences,
        device=f"cuda:{gpu_id[rank]}",
        position=rank,
    )
    perplexities = results["perplexities"]
    logging.info(f"[GPU {gpu_id[rank]}] Computed perplexities for {len(filtered_sentences)} sentences. Putting results in queue...")
    queue.put((rank, filtered_sentences, perplexities))


def main(_):
    logging.info("="*60)
    logging.info("STEP 1: Dataset Construction Pipeline")
    logging.info("="*60)
    logging.info(f"Dataset: {FLAGS.dataset}")
    logging.info(f"Model: {FLAGS.llm_model_name} (checkpoint: {FLAGS.ckpt_step})")
    logging.info(f"Batch size: {FLAGS.batch_size}")
    logging.info(f"GPU IDs: {FLAGS.gpu_id}")
    
    # Load dataset
    logging.info("\n[1/5] Loading dataset...")
    if FLAGS.dataset == "openwebtext":
        dataset = load_dataset("sam2ai/openwebtext-10k", split="train", streaming=False)
    else:
        raise ValueError(f"Unknown dataset: {FLAGS.dataset}")
    logging.info(f"Loaded {len(dataset)} samples")

    # Filter and split examples
    logging.info("\n[2/5] Filtering and splitting examples...")
    dataset = dataset.filter(lambda x: x["text"] != "")
    dataset = dataset.map(split_examples, remove_columns=dataset.column_names)
    all_sentences = list(itertools.chain(*dataset["sentences"]))
    logging.info(f"Total sentences: {len(all_sentences)}")
    p_sentence_indices = np.array_split(np.arange(len(all_sentences)), len(FLAGS.gpu_id))

    hf_model_name = hf_model_name_map[FLAGS.llm_model_name]
    revision = "main" if FLAGS.ckpt_step == -1 else f"step{FLAGS.ckpt_step}"
    logging.info(f"Model revision: {revision}")

    # Compute perplexities
    logging.info("\n[3/5] Computing perplexities in parallel...")
    with mp.Manager() as manager:
        queue = manager.Queue()
        logging.info(f"Spawning {len(FLAGS.gpu_id)} processes for GPU(s): {FLAGS.gpu_id}")
        mp.spawn(
            run_ppl,
            args=(queue, hf_model_name, revision, FLAGS.gpu_id, FLAGS.batch_size, all_sentences, p_sentence_indices),
            nprocs=len(FLAGS.gpu_id),
            join=True,
        )
        logging.info(f"Receiving results from {len(FLAGS.gpu_id)} processes...")
        mp_results = [queue.get() for _ in range(len(FLAGS.gpu_id))]
        logging.info(f"✓ Received {len(mp_results)} results")

    # Aggregate results
    logging.info("\n[4/5] Aggregating results and filtering by perplexity...")
    mp_results.sort(key=lambda x: x[0])
    filtered_sentences = []
    perplexities = []
    for _, sentences, ppls in mp_results:
        filtered_sentences.extend(sentences)
        perplexities.extend(ppls)
    logging.info(f"Total filtered sentences: {len(filtered_sentences)}")

    min_ppl = np.percentile(perplexities, 1)
    max_ppl = np.percentile(perplexities, 99)
    logging.info(f"Perplexity range (1st-99th percentile): [{min_ppl:.2f}, {max_ppl:.2f}]")

    saved_sentences = []
    saved_perplexities = []
    logging.info("Filtering sentences within perplexity range...")
    for sentence, ppl in tqdm(zip(filtered_sentences, perplexities), desc="Filtering perplexity", total=len(filtered_sentences), position=0, leave=True):
        if min_ppl <= ppl <= max_ppl:
            saved_sentences.append(sentence)
            saved_perplexities.append(ppl)
    logging.info(f"✓ Final dataset size: {len(saved_sentences)} sentences")

    # Save dataset
    logging.info("\n[5/5] Saving dataset...")

    saved_dataset = {
        "sentences": saved_sentences,
        "perplexities": saved_perplexities
    }

    if hf_model_name.startswith("EleutherAI") and revision != "main":
        save_path = main_dir / f"data/graph_probing/{FLAGS.dataset}-10k-{FLAGS.llm_model_name}-{revision}.csv"
    else:
        save_path = main_dir / f"data/graph_probing/{FLAGS.dataset}-10k-{FLAGS.llm_model_name}.csv"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df = pd.DataFrame(saved_dataset)
    df.to_csv(save_path, index=False)
    logging.info(f"✓ Dataset saved to: {save_path}")
    logging.info("="*60)
    logging.info("✓ STEP 1 Complete: Dataset Construction Finished")
    logging.info("="*60)


if __name__ == "__main__":
    app.run(main)
