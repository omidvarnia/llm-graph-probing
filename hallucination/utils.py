import torch
from absl import logging
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from utils.constants import BASE_MODELS, QWEN_CHAT_MODELS


def select_device(gpu_id: int = 0):
    """
    Intelligently select device: ROCm → CUDA → CPU
    
    Priority:
    1. ROCm (AMD GPUs) - check HIP_VISIBLE_DEVICES
    2. CUDA (NVIDIA GPUs) - check CUDA_VISIBLE_DEVICES  
    3. CPU fallback (no GPU available)
    
    Args:
        gpu_id: GPU index to use (default: 0)
    
    Returns:
        torch.device: Selected device
    """
    if torch.cuda.is_available():
        # Detect device type
        device_name = torch.cuda.get_device_name(gpu_id)
        is_amd = "AMD" in device_name or "MI" in device_name or "Radeon" in device_name
        is_rocm = "HIP" in torch.version.hip if hasattr(torch.version, 'hip') else False
        
        try:
            device = torch.device(f"cuda:{gpu_id}")
            
            # Sanity check: test basic GPU operations
            try:
                idx = torch.tensor([0, 0, 1, 1], device=device)
                x = torch.randn(4, 8, device=device)
                out = torch.zeros(2, 8, device=device)
                out.index_add_(0, idx, x)
                _ = (out @ out.T).sum()  # matmul sanity check
            except RuntimeError as e:
                logging.warning(f"GPU sanity check failed: {e}. Falling back to CPU.")
                device = torch.device("cpu")
                logging.info(f"Using device: CPU (fallback after GPU sanity check failed)")
                return device
            
            device_type = "ROCm/HIP" if is_rocm else "CUDA"
            logging.info(f"✓ Using device: {device} (GPU: {device_name}, Type: {device_type})")
            return device
            
        except RuntimeError as e:
            logging.warning(f"Failed to use CUDA device {gpu_id}: {e}. Falling back to CPU.")
            device = torch.device("cpu")
            logging.info(f"Using device: CPU (fallback)")
            return device
    else:
        logging.warning("CUDA/ROCm not available. Using CPU (this will be slow).")
        device = torch.device("cpu")
        logging.info(f"Using device: CPU")
        return device


def format_prompt(questions, answers, model_name, tokenizer):
    prompts = []
    for question, answer in zip(questions, answers):
        if model_name in QWEN_CHAT_MODELS:
            messages = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": str(answer)}
            ]
            prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=False,
                return_tensors=None, return_dict=False,
                tokenize=False, continue_final_message=False, enable_thinking=False)
        elif model_name in BASE_MODELS:
            prompt = f"Question: {question} Answer: {str(answer)}"
        else:
            raise NotImplementedError(f"Model {model_name} is not supported.")
        prompts.append(prompt)
    return prompts


def format_prompt_ccs(questions, answers, model_name, tokenizer, suffix):
    prompts = []
    for question, answer in zip(questions, answers):
        if model_name in QWEN_CHAT_MODELS:
            messages = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": str(answer)},
                {"role": "user", "content": f"Is the answer above correct?"},
                {"role": "assistant", "content": suffix}
            ]
            prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=False,
                return_tensors=None, return_dict=False,
                tokenize=False, continue_final_message=False, enable_thinking=False)
        elif model_name in BASE_MODELS:
            prompt = f"Question: {question} Answer: {str(answer)} Is the answer above correct? {suffix}."
        else:
            raise NotImplementedError(f"Model {model_name} is not supported.")
        prompts.append(prompt)
    return prompts


def test_fn(model, data_loader, device, num_layers):
    model.eval()
    all_preds = []
    all_labels = []
    all_outputs = []
    with torch.no_grad():
        for data in data_loader:
            if num_layers > 0:
                data = data.to(device)
                out = model(data.x, data.edge_index, data.edge_attr, data.batch)
                pred = out.argmax(dim=-1)
                all_preds.append(pred.cpu())
                all_labels.append(data.y.cpu())
                all_outputs.append(out.cpu())
            else:
                x, y = data
                x, y = x.to(device), y.to(device)
                out = model(x)
                pred = out.argmax(dim=-1)
                all_preds.append(pred.cpu())
                all_labels.append(y.cpu())
                all_outputs.append(out.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_outputs = torch.cat(all_outputs).numpy()
    
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)  # Shape: (2, 2) with [[TN, FP], [FN, TP]]
    
    return acc, precision, recall, f1, cm

def test_fn_ccs(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in data_loader:
            x_yes, x_no, y = data
            x_yes, x_no, y = x_yes.to(device), x_no.to(device), y.to(device)
            out_yes = model.forward(x_yes)
            out_no = model.forward(x_no)
            pred = 0.5 * (out_yes + (1 - out_no))
            all_preds.append((pred.cpu().flatten() >= 0.5).long())
            all_labels.append(y.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    acc_rev = accuracy_score(1 - all_labels, all_preds)
    precision_rev, recall_rev, f1_rev, _ = precision_recall_fscore_support(1 - all_labels, all_preds, average='binary', zero_division=0)
    cm_rev = confusion_matrix(1 - all_labels, all_preds)

    if acc_rev > acc:
        return acc_rev, precision_rev, recall_rev, f1_rev, cm_rev
    
    return acc, precision, recall, f1, cm
