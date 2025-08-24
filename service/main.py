from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict
import os, re, json
import numpy as np
import torch
import torch.nn as nn
from transformers import XLMRobertaModel, XLMRobertaConfig, AutoTokenizer
import gdown

# -------------------- Config --------------------
CKPT_FOLDER_ID_OR_URL = os.getenv("CKPT_FOLDER_ID", "").strip() 
CKPT_DIR              = os.getenv("CKPT_DIR", "/models/emo_ckpt").strip()
DEVICE = torch.device("cpu")
torch.set_num_threads(max(1, int(os.getenv("TORCH_NUM_THREADS", "1"))))

# -------------------- Utils --------------------
def _extract_drive_id(s: str) -> str:
    if not s:
        return ""
    m = re.search(r'/folders/([A-Za-z0-9_-]+)', s) or re.search(r'[?&]id=([A-Za-z0-9_-]+)', s)
    return m.group(1) if m else s

def ensure_ckpt() -> str:
    os.makedirs(CKPT_DIR, exist_ok=True)
    required = [
        os.path.join(CKPT_DIR, "best.pt"),
        os.path.join(CKPT_DIR, "tokenizer.json"),
        os.path.join(CKPT_DIR, "tokenizer_config.json"),
        os.path.join(CKPT_DIR, "special_tokens_map.json"),
        os.path.join(CKPT_DIR, "sentencepiece.bpe.model"),
    ]
    if all(os.path.exists(p) for p in required):
        print(f"[init] Using existing checkpoint at {CKPT_DIR}")
        return CKPT_DIR

    if not CKPT_FOLDER_ID_OR_URL:
        print("⚠️  CKPT_FOLDER_ID not set; service will start but /classify will return 503.")
        return CKPT_DIR

    folder_id = _extract_drive_id(CKPT_FOLDER_ID_OR_URL)
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    print(f"[init] Downloading checkpoint folder via gdown: {url}")
    try:
        gdown.download_folder(url=url, output=CKPT_DIR, quiet=False, use_cookies=False)
    except Exception as e:
        print(f"❌ gdown error: {e}")
    return CKPT_DIR

def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)

# -------------------- Heads --------------------
class TaskHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout_rate=0.1, depth=2):
        super().__init__()
        layers = []
        cur = input_dim
        for _ in range(depth):
            layers += [nn.Linear(cur, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate)]
            cur = hidden_dim
        layers += [nn.Linear(cur, num_classes)]
        self.layers = nn.Sequential(*layers)
    def forward(self, x): return self.layers(x)

class MultiTaskXLMR(nn.Module):
    def __init__(self, task_configs: Dict, encoder_config: XLMRobertaConfig, pooling_strategy: str = 'weighted'):
        super().__init__()
        # Build encoder EXACTLY with the config (no hub calls)
        self.encoder = XLMRobertaModel(encoder_config)
        print(f"[init] Encoder shape => vocab:{self.encoder.config.vocab_size}, "
              f"pos:{self.encoder.config.max_position_embeddings}, "
              f"type_vocab:{self.encoder.config.type_vocab_size}")

        self.pooling_strategy = pooling_strategy
        hidden = self.encoder.config.hidden_size
        self.attn_proj = nn.Linear(hidden, 1) if pooling_strategy == 'weighted' else None

        self.emotion_head  = TaskHead(hidden, task_configs['emotion']['dim'],  task_configs['emotion']['num_classes'],  task_configs['emotion']['dropout'],  task_configs['emotion']['depth'])
        self.severity_head = TaskHead(hidden, task_configs['severity']['dim'], task_configs['severity']['num_classes'], task_configs['severity']['dropout'], task_configs['severity']['depth'])
        self.topic_head    = TaskHead(hidden, task_configs['topic']['dim'],    task_configs['topic']['num_classes'],    task_configs['topic']['dropout'],    task_configs['topic']['depth'])
        self.intent_head   = TaskHead(hidden, task_configs['intent']['dim'],   task_configs['intent']['num_classes'],   task_configs['intent']['dropout'],   task_configs['intent']['depth'])

    def pool_features(self, hs: torch.Tensor, am: torch.Tensor) -> torch.Tensor:
        if self.pooling_strategy == 'cls':
            return hs[:, 0]
        if self.pooling_strategy == 'mean':
            mask = am.unsqueeze(-1).expand_as(hs).float()
            denom = torch.clamp(mask.sum(1), min=1e-9)
            return (hs * mask).sum(1) / denom
        if self.pooling_strategy == 'max':
            mask = am.unsqueeze(-1).expand_as(hs).bool()
            return hs.masked_fill(~mask, -1e9).max(1)[0]
        scores = self.attn_proj(hs).squeeze(-1)
        scores = scores.masked_fill(am == 0, -1e9)
        weights = torch.softmax(scores, dim=1)
        return torch.sum(hs * weights.unsqueeze(-1), dim=1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.pool_features(out.last_hidden_state, attention_mask)
        return {
            "logits_emotion":  self.emotion_head(pooled),
            "logits_severity": self.severity_head(pooled),
            "logits_topic":    self.topic_head(pooled),
            "logits_intent":   self.intent_head(pooled),
        }

# -------------------- Globals --------------------
tokenizer = None
model = None
label_maps = None
emotion_thresholds = None

def _build_xlmr_config_from_state(state_dict: Dict[str, torch.Tensor]) -> XLMRobertaConfig:
    """Derive the exact XLM-R config sizes from checkpoint tensors."""
    we = state_dict["encoder.embeddings.word_embeddings.weight"].shape  # (vocab, hidden)
    pe = state_dict["encoder.embeddings.position_embeddings.weight"].shape  # (pos, hidden)
    try:
        te = state_dict["encoder.embeddings.token_type_embeddings.weight"].shape  # (type_vocab, hidden)
        type_vocab = te[0]
    except KeyError:
        type_vocab = 1  # XLM-R default

    vocab_size = we[0]
    hidden_size = we[1]
    max_pos = pe[0]

    # Heuristic defaults for base; if your model is large, adjust here or infer heads/layers.
    num_hidden_layers = 12
    num_attention_heads = 12
    intermediate_size = hidden_size * 4

    cfg = XLMRobertaConfig(
        vocab_size=vocab_size,
        max_position_embeddings=max_pos,
        type_vocab_size=type_vocab,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers,
    )
    print(f"[init] Derived cfg from state -> vocab:{vocab_size}, pos:{max_pos}, type_vocab:{type_vocab}, hidden:{hidden_size}")
    return cfg

def load_model():
    """Download artifacts (if needed) and load model + tokenizer + label maps with exact config."""
    global tokenizer, model, label_maps, emotion_thresholds

    ensure_ckpt()
    ckpt_path = os.path.join(CKPT_DIR, "best.pt")
    if not os.path.exists(ckpt_path):
        print("❌ No checkpoint found; classifier will be unavailable.")
        return

    print(f"[init] Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = state["model_state_dict"]

    # Build EXACT config from checkpoint tensors
    encoder_cfg = _build_xlmr_config_from_state(sd)

    pooling = state.get('pooling', 'weighted')
    task_configs = state['task_configs']
    emotion_thresholds = state.get('best_emotion_thresholds', None)

    # Labels
    labels_path = os.path.join(CKPT_DIR, "label_maps.json")
    if os.path.exists(labels_path):
        with open(labels_path, "r") as f:
            label_maps_local = json.load(f)
    else:
        C_em = task_configs['emotion']['num_classes']
        C_se = task_configs['severity']['num_classes']
        C_to = task_configs['topic']['num_classes']
        C_in = task_configs['intent']['num_classes']
        emo_names = ["anger","anticipation","disgust","fear","joy","sadness","surprise","trust","neutral"] if C_em == 9 else [f"emotion_{i}" for i in range(C_em)]
        label_maps_local = {
            "emotion": emo_names,
            "severity": [f"severity_{i}" for i in range(C_se)],
            "topic":    [f"topic_{i}" for i in range(C_to)],
            "intent":   [f"intent_{i}" for i in range(C_in)],
        }
    label_maps = label_maps_local

    # Tokenizer (local only)
    try:
        tokenizer_local = AutoTokenizer.from_pretrained(CKPT_DIR, local_files_only=True)
        print("[init] Loaded tokenizer from CKPT_DIR")
    except Exception as e:
        print(f"❌ Failed to load tokenizer locally: {e}")
        tokenizer_local = None
    globals()['tokenizer'] = tokenizer_local

    # Model with exact config
    m = MultiTaskXLMR(task_configs, encoder_config=encoder_cfg, pooling_strategy=pooling)

    # Load weights
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[init] load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")
        if missing:
            print("  - missing:", missing[:10], ("..." if len(missing) > 10 else ""))
        if unexpected:
            print("  - unexpected:", unexpected[:10], ("..." if len(unexpected) > 10 else ""))
    m.eval()
    m.to(DEVICE)
    globals()['model'] = m
    print("✅ Classifier loaded and ready.")

# -------------------- FastAPI --------------------
app = FastAPI(title="VentPal Classifier Service", version="1.0.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

class ClassifyRequest(BaseModel):
    text: str

@app.on_event("startup")
def _startup():
    load_model()

@app.get("/health")
def health():
    cfg = getattr(model.encoder, "config", None) if model else None
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "vocab_size": getattr(cfg, "vocab_size", None) if cfg else None,
        "max_position_embeddings": getattr(cfg, "max_position_embeddings", None) if cfg else None,
        "type_vocab_size": getattr(cfg, "type_vocab_size", None) if cfg else None,
    }

@app.post("/classify")
def classify(req: ClassifyRequest):
    if model is None or tokenizer is None or label_maps is None:
        return {"error": "model_not_loaded"}, 503

    text = (req.text or "").strip()
    if not text:
        return {"error": "empty_text"}, 400

    tok = tokenizer(text, truncation=True, max_length=256, padding="max_length", return_tensors="pt")
    input_ids = tok["input_ids"].to(DEVICE)
    attn = tok["attention_mask"].to(DEVICE)

    with torch.no_grad():
        out = model(input_ids, attn)

        # Emotion (multi-label, sigmoid)
        em_logits = out["logits_emotion"][0].float().cpu().numpy()
        em_probs = 1.0 / (1.0 + np.exp(-em_logits))
        thresholds = (np.array(emotion_thresholds, dtype=np.float32)
                      if isinstance(emotion_thresholds, (list, np.ndarray)) and len(emotion_thresholds) == len(label_maps["emotion"])
                      else np.full_like(em_probs, 0.5, dtype=np.float32))
        mask = em_probs >= thresholds
        em_pred_labels = [label_maps["emotion"][i] for i, m in enumerate(mask) if m]
        top_em_idx = int(np.argmax(em_probs))
        top_em = label_maps["emotion"][top_em_idx]
        top_em_conf = float(em_probs[top_em_idx])

        # Single-label heads (softmax)
        def top1(logits, names):
            lg = logits[0].float().cpu().numpy()
            sm = softmax_np(lg)
            ix = int(np.argmax(sm))
            return names[ix], float(sm[ix]), [float(x) for x in sm]

        sev_name, sev_conf, sev_probs = top1(out["logits_severity"], label_maps["severity"])
        top_name, top_conf, top_probs = top1(out["logits_topic"],    label_maps["topic"])
        int_name, int_conf, int_probs = top1(out["logits_intent"],   label_maps["intent"])

    return {
        "emotion": {
            "probs": dict(zip(label_maps["emotion"], [float(x) for x in em_probs])),
            "thresholds": dict(zip(label_maps["emotion"], [float(x) for x in thresholds])),
            "pred_labels": em_pred_labels,
            "top": {"label": top_em, "conf": top_em_conf}
        },
        "severity": {"top": {"label": sev_name, "conf": sev_conf}, "probs": dict(zip(label_maps["severity"], sev_probs))},
        "topic":    {"top": {"label": top_name, "conf": top_conf}, "probs": dict(zip(label_maps["topic"], top_probs))},
        "intent":   {"top": {"label": int_name, "conf": int_conf}, "probs": dict(zip(label_maps["intent"], int_probs))},
        "label_maps": label_maps
    }
