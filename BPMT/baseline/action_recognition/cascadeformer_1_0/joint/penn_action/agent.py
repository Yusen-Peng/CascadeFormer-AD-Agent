import torch
from torch.utils.data import DataLoader
from typing import List, Dict, Any
from typing import Optional
import numpy as np
import time
import csv
from sklearn.neighbors import NearestNeighbors
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# local import
from finetuning import load_T1, load_T2, load_cross_attn, GaitRecognitionHead
from base_dataset import ActionRecognitionDataset
from penn_utils import build_penn_action_lists, split_train_val


########## Model #########################################################################
##########################################################################################

model_config = {
    "t1_ckpt": "action_checkpoints/Penn_finetuned_T1.pt",
    "t2_ckpt": "action_checkpoints/Penn_finetuned_T2.pt",
    "cross_attn_ckpt": "action_checkpoints/Penn_finetuned_cross_attn.pt",
    "gait_head_ckpt": "action_checkpoints/Penn_finetuned_head.pt",
    "hidden_size": 256,
    "n_heads": 8,
    "num_layers": 12,
}

dataset_config = {
    "num_classes": 12
}

# we have 15 action classes
classify_labels = [
    "baseball_pitch", "clean_and_jerk", "pull_ups", "strumming_guitar",
    "baseball_swing", "golf_swing", "push_ups", "tennis_forehand",
    "bench_press", "jumping_jacks", "sit_ups", "tennis_serve",
    "bowling", "jump_rope", "squats"
]

class CascadeFormerWrapper:
    def __init__(self, device="cuda"):
        self.device = device

        self.t1 = load_T1(model_config["t1_ckpt"], d_model=model_config["hidden_size"], nhead=model_config["n_heads"], num_layers=model_config["num_layers"], device=device)

        self.t2 = load_T2(model_config["t2_ckpt"], d_model=model_config["hidden_size"], nhead=model_config["n_heads"], num_layers=model_config["num_layers"], device=device)
        self.cross_attn = load_cross_attn(model_config["cross_attn_ckpt"], d_model=model_config["hidden_size"], device=device)

        # load the gait recognition head
        self.gait_head = GaitRecognitionHead(input_dim=model_config["hidden_size"], num_classes=dataset_config["num_classes"])
        self.gait_head.load_state_dict(torch.load(model_config["gait_head_ckpt"], map_location="cpu"))
        self.gait_head = self.gait_head.to(device)

        print("=" * 100)
        print("Aha! All models loaded successfully!")

        # set models to evaluation mode
        self.t1.eval()
        self.t2.eval()
        self.cross_attn.eval()
        self.gait_head.eval()

    @torch.inference_mode()
    def infer(self, skel_batch: torch.Tensor) -> Dict[str, Any]:
        """
        skel_batch: (B, T, J, C) float32
        returns dict with logits, probs, embedding
        """
        x1 = self.t1.encode(skel_batch.to(self.device))        
        x2 = self.t2.encode(x1)
        fused = self.cross_attn(x1, x2, x2)
        pooled = fused.mean(dim=1)
        logits = self.gait_head(pooled)
        probs = torch.softmax(logits, dim=-1).float()
        embedding = torch.nn.functional.normalize(pooled, dim=-1)
        return {
            "logits": logits.cpu().numpy(),
            "probs": probs.cpu().numpy(),
            "embedding": embedding.detach().cpu().numpy(),
        }

# global model wrapper
MODEL = CascadeFormerWrapper(device="cuda")


@tool("perceive_window", return_direct=False)
def perceive_window(skel_window: List[List[List[float]]]) -> Dict[str, Any]:
    """
        run CascadeFormer on a single window of skeletons and return structured event with probs, entropy, and embedding.
    """
    x = torch.tensor(skel_window, dtype=torch.float32).unsqueeze(0) # shape: (1,T,J,C)
    out = MODEL.infer(x)
    probs = out["probs"][0]
    event = {
        "top_label": classify_labels[int(np.argmax(probs))],
        "top_prob": float(np.max(probs)),
        "entropy": entropy(probs),
        "embedding": out["embedding"][0].tolist(),
    }
    return event

########## anomaly detection tool ########################################################
##########################################################################################


def build_normal_bank(
    model : CascadeFormerWrapper, 
    dataset, 
    batch_size: int = 32, 
    device: str = "cuda",
    per_class: bool = False,
    num_classes: Optional[int] = None
) -> np.ndarray:
    """
        build a normal bank of embeddings from the dataset using the provided model.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    
    if per_class:
        assert num_classes is not None, "num_classes required if per_class=True"
        banks = {c: [] for c in range(num_classes)}
    else:
        embeddings = []
    
    with torch.inference_mode():
        for skel_batch, labels in loader:   # (B,T,J,C), (B,)
            skel_batch = skel_batch.to(device)
            out = model.infer(skel_batch)   # dict with "embedding"
            emb = out["embedding"]          # (B, D), numpy
            
            if per_class:
                for e, lbl in zip(emb, labels):
                    banks[int(lbl)].append(e)
            else:
                embeddings.append(emb)
    
    if per_class:
        # Convert lists to arrays
        banks = {c: np.stack(b, axis=0).astype(np.float32) 
                 for c, b in banks.items() if len(b) > 0}
    else:
        banks = np.concatenate(embeddings, axis=0).astype(np.float32)
    return banks


class DistanceScorer:
    """
        How far is this embedding z from normal training data?
    """
    def __init__(self, k=5):

        root_dir = "Penn_Action/"
        train_seq, train_lbl, _, _ = build_penn_action_lists(root_dir)
        val_ratio = 0.05
        train_seq, train_lbl, _, _ = split_train_val(train_seq, train_lbl, val_ratio=val_ratio)
        dataset = ActionRecognitionDataset(train_seq, train_lbl)
        normal_bank = build_normal_bank(MODEL, dataset, per_class=False)
        self.nn = NearestNeighbors(n_neighbors=k).fit(normal_bank)
    def score(self, z: np.ndarray) -> float:
        dists, _ = self.nn.kneighbors(z.reshape(1, -1))
        return float(dists.mean())

KNNS = type("KNN", (), {"score": DistanceScorer().score})()

def entropy(p: np.ndarray) -> float:
    """
        How uncertain is the model's prediction?
    """
    p = np.clip(p, 1e-8, 1.0)
    return float(-(p * np.log(p)).sum())

@tool("score_anomaly", return_direct=False)
def score_anomaly(event: Dict[str, Any]) -> Dict[str, Any]:
    """
        compute anomaly scores from embedding and additional signals.
    """
    z = np.array(event["embedding"], dtype=np.float32)
    scores = {
        "knn_dist": KNNS.score(z),
        "low_conf": 1.0 - event["top_prob"],
        "ent": event["entropy"],
    }
    # simple fused score (tune/learn later)
    scores["anom_score"] = 0.5 * scores["knn_dist"] + 0.3 * scores["low_conf"] + 0.2 * scores["ent"]
    return scores

@tool("log_event", return_direct=False)
def log_event(event: Dict[str, Any], scores: Dict[str, Any]) -> str:
    """
        persist the event + scores into a CSV file.
    """
    with open("events.csv", "a", newline="") as f:
        fieldnames = list(event.keys()) + list(scores.keys()) + ["ts"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        # Write header only if file is empty
        if f.tell() == 0:
            writer.writeheader()

        row = {**event, **scores, "ts": time.time()}
        writer.writerow(row)
    return "logged"

@tool("raise_alert", return_direct=False)
def raise_alert(event: Dict[str, Any], scores: Dict[str, Any]) -> str:
    """
        send an alert with a short rationale.
    """
    return f"ALERT: {event['top_label']} (p={event['top_prob']:.2f}), anom={scores['anom_score']:.2f}"


########## RAG (policy + incident) #######################################################
##########################################################################################

emb = OpenAIEmbeddings(model="text-embedding-3-small")

policies_store = FAISS.from_texts(
    texts=[
        "If fighting is detected after 22:00 in Zone C, escalate to security level 2.",
        "If confidence < 0.35 AND kNN distance > 1.2, flag as anomaly.",
        "If elderly fall-like motion detected, notify medical response.",
    ],
    embedding=emb
)

incidents_store = FAISS.from_texts(
    texts=[
        "2025-08-01 Zone B crowd surge pattern; high entropy, high recon loss.",
        "2025-07-14 unusual loitering with low motion variance; medium kNN distance."
    ],
    embedding=emb
)

def retrieve_context(event: Dict[str, Any]) -> str:
    """
        retrieve context from the knowledge base
    """
    q = f"label={event['top_label']} prob={event['top_prob']:.2f} entropy={event['entropy']:.2f}"
    pol = "\n".join([d.page_content for d in policies_store.similarity_search(q, k=3)])
    inc = "\n".join([d.page_content for d in incidents_store.similarity_search(q, k=3)])
    return f"[POLICIES]\n{pol}\n\n[SIMILAR INCIDENTS]\n{inc}"


########## Anomaly detection agent #######################################################
##########################################################################################

POLICY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a surveillance policy agent. You receive a perception event JSON, anomaly scores, "
     "and retrieved policies/incidents. Decide ONE action: LOG or ALERT. "
     "Return JSON with keys: action, rationale."),
    ("human", "Event:\n{event}\n\nScores:\n{scores}\n\nContext:\n{context}")
])

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
policy_chain = POLICY_PROMPT | llm | StrOutputParser()

def decide(event: Dict[str, Any], scores: Dict[str, Any]) -> Dict[str, Any]:
    """
        make a decision based on the event, scores, and context.
    """
    ctx = retrieve_context(event)
    out = policy_chain.invoke({"event": event, "scores": scores, "context": ctx})
    try:
        import json
        decision = json.loads(out)
    except Exception:
        decision = {"action": "LOG", "rationale": "fallback"}
    return decision


def process_window(skel_window: List[List[List[float]]]) -> Dict[str, Any]:
    """
        Process a single window of skeleton data, run the model, score anomalies, and decide on action.
    """
    event = perceive_window(skel_window)            # CascadeFormer: perceive
    scores = score_anomaly(event)                   # anomaly scores
    decision = decide(event, scores)                     # LLM policy with RAG
    if decision["action"] == "ALERT":
        msg = raise_alert(event, scores)            # escalate
    else:
        msg = log_event(event, scores)              # just log
    return {"event": event, "scores": scores, "decision": decision, "result": msg}

