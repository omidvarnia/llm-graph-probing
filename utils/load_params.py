#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path

# Minimal YAML reader for simple key: value sections
# Supports two-level mapping: top-level section (no indent) then indented key: value
# Booleans: true/false; null -> empty string; numbers -> int/float; strings otherwise

def parse_simple_yaml(text: str):
    cfg = {}
    current = None
    for raw in text.splitlines():
        line = raw.split('#', 1)[0].rstrip()  # strip comments and right spaces
        if not line.strip():
            continue
        if not line.startswith(' ') and line.endswith(':'):
            # section header
            key = line[:-1].strip()
            cfg[key] = {}
            current = key
            continue
        if current is None:
            continue
        # expect indented key: value
        if ':' in line:
            k, v = line.strip().split(':', 1)
            k = k.strip()
            v = v.strip()
            # type inference
            if v.lower() in ('null', 'none', ''):
                val = ''
            elif v.lower() in ('true', 'false'):
                val = 'true' if v.lower() == 'true' else 'false'
            else:
                try:
                    if '.' in v:
                        val = str(float(v))
                    else:
                        val = str(int(v))
                except ValueError:
                    # strip quotes if present
                    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                        val = v[1:-1]
                    else:
                        val = v
            cfg[current][k] = val
    return cfg

def export_shell_vars(section: dict, pipeline: str):
    # Map config keys to shell variable names
    if pipeline == 'hallucination':
        mapping = {
            'dataset_name': 'DATASET_NAME',
            'model_name': 'MODEL_NAME',
            'ckpt_step': 'CKPT_STEP',
            'batch_size': 'BATCH_SIZE',
            'layer_list': 'LAYER_LIST',
            'probe_input': 'PROBE_INPUT',
            'density': 'DENSITY',
            'eval_batch_size': 'EVAL_BATCH_SIZE',
            'hidden_channels': 'HIDDEN_CHANNELS',
            'num_layers': 'NUM_LAYERS',
            'learning_rate': 'LEARNING_RATE',
            'from_sparse_data': 'FROM_SPARSE_DATA',
            'aggregate_layers': 'AGGREGATE_LAYERS',
            'num_epochs': 'NUM_EPOCHS',
            'gpu_id': 'GPU_ID',
            'early_stop_patience': 'EARLY_STOP_PATIENCE',
            'label_smoothing': 'LABEL_SMOOTHING',
            'gradient_clip': 'GRADIENT_CLIP',
            'warmup_epochs': 'WARMUP_EPOCHS',
            'dataset_fraction': 'DATASET_FRACTION',
        }
    else:  # neuropathology
        mapping = {
            'dataset_name': 'DATASET_NAME',
            'model_name': 'MODEL_NAME',
            'ckpt_step': 'CKPT_STEP',
            'layer_list': 'LAYER_LIST',
            'density': 'DENSITY',
            'num_layers': 'NUM_LAYERS',
            'hidden_channels': 'HIDDEN_CHANNELS',
            'gpu_id': 'GPU_ID',
            'from_sparse_data': 'FROM_SPARSE_DATA',
            'early_stop_patience': 'EARLY_STOP_PATIENCE',
            'disease_pattern': 'DISEASE_PATTERN',
            'num_clusters': 'NUM_CLUSTERS',
            'within_scale': 'WITHIN_SCALE',
            'between_scale': 'BETWEEN_SCALE',
            'rewiring_prob': 'REWIRING_PROB',
            'distance_threshold': 'DISTANCE_THRESHOLD',
            'aggregate_layers': 'AGGREGATE_LAYERS',
        }
    out = []
    for k, var in mapping.items():
        if k in section:
            val = section[k]
            # ensure booleans are lower-case strings 'true'/'false'
            out.append(f"{var}={val}")
    return "\n".join(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--pipeline', choices=['hallucination', 'neuropathology'], required=True)
    args = ap.parse_args()

    text = Path(args.config).read_text(encoding='utf-8')
    cfg = parse_simple_yaml(text)
    common = cfg.get('common', {})
    if args.pipeline not in cfg:
        print(f"echo 'ERROR: pipeline {args.pipeline} not found in {args.config}' >&2; exit 1")
        return
    merged = common.copy()
    merged.update(cfg[args.pipeline])
    print(export_shell_vars(merged, args.pipeline))

if __name__ == '__main__':
    main()
