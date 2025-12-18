# -*- coding: utf-8 -*-
"""
BUTD Captioning Web App (boxes + labels + TSV) - Gradio older-compat (fixed headers/column_count)
"""

import os, io, json, base64, collections, collections.abc, numpy as np, cv2, torch, math, gradio as gr
from typing import Tuple, List, Dict, Any

# ---- Pillow>=10 legacy names shims ----
for _n in ("Mapping","MutableMapping","Sequence"):
    if not hasattr(collections, _n):
        setattr(collections, _n, getattr(collections.abc, _n))
try:
    from PIL import Image as _PIL_Image
    try:
        from PIL.Image import Resampling as _R
    except Exception:
        _R = None
    def _fb(name, val):
        if not hasattr(_PIL_Image, name):
            setattr(_PIL_Image, name, val)
    _fb("NEAREST",  _R.NEAREST  if _R else 0)
    _fb("BILINEAR", _R.BILINEAR if _R else 2)
    _fb("BICUBIC",  _R.BICUBIC  if _R else 3)
    _fb("LANCZOS",  _R.LANCZOS if _R else 1)
    _fb("ANTIALIAS",getattr(_PIL_Image, "LANCZOS", _R.LANCZOS if _R else 1))
    _fb("LINEAR",   getattr(_PIL_Image, "BILINEAR", _R.BILINEAR if _R else 2))
    _fb("CUBIC",    getattr(_PIL_Image, "BICUBIC",  _R.BICUBIC  if _R else 3))
except Exception:
    pass
try:
    cv2.setNumThreads(0)
except Exception:
    pass

# ====== CONFIG ======
YAML        = os.environ.get("BUTD_YAML",   r"D:\DACN\project\SCAN\py-bottom-up-attention\configs\VG-Detection\faster_rcnn_R_101_C4_attr_caffemaxpool.yaml")
WEIGHT      = os.environ.get("BUTD_WEIGHT", r"D:\DACN\project\SCAN\py-bottom-up-attention\checkpoints\faster_rcnn_from_caffe_attr.pkl")
CAPTION_CKPT= os.environ.get("BUTD_CAP_CKPT", r"D:\DACN\project\SCAN\py-bottom-up-attention\checkpoints\best_xe.pt")
VOCAB_JSON  = os.environ.get("BUTD_VOCAB",  r"D:\DACN\project\SCAN\py-bottom-up-attention\checkpoints\f30k_vocab.json")
OBJ_VOCAB   = os.environ.get("BUTD_OBJ_VOCAB",  r"D:\DACN\project\SCAN\py-bottom-up-attention\demo\data\genome\1600-400-20\objects_vocab.txt")  # optional path to objects_vocab.txt
ATTR_VOCAB  = os.environ.get("BUTD_ATTR_VOCAB", r"D:\DACN\project\SCAN\py-bottom-up-attention\demo\data\genome\1600-400-20\attributes_vocab.txt")  # optional path to attributes_vocab.txt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_OBJECTS = int(os.environ.get("BUTD_NUM_OBJECTS", "36"))
MIN_SIZE_TEST = int(os.environ.get("BUTD_MIN_TEST", "600"))
MAX_SIZE_TEST = int(os.environ.get("BUTD_MAX_TEST", "1000"))
RPN_POST_NMS_TOPK_TEST = int(os.environ.get("BUTD_RPN_TOPK", "150"))
MAX_LEN = int(os.environ.get("BUTD_MAXLEN", "20"))
TITLE = "BUTD Captioning (boxes + labels + TSV)"
DESC  = "BUTD 36×2048 + caption ckpt. Draws boxes w/ object+attr labels, shows a detections table, and exports TSV."

# ====== BUTD (Detectron2) ======
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data.transforms import ResizeShortestEdge
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputs, fast_rcnn_inference_single_image
from detectron2.data import MetadataCatalog

def build_butd_predictor(yaml_path, weight_path, device=DEVICE):
    assert os.path.isfile(yaml_path), f"Missing YAML: {yaml_path}"
    assert os.path.isfile(weight_path), f"Missing WEIGHT: {weight_path}"
    torch.backends.cudnn.benchmark = True
    try: torch.set_float32_matmul_precision("high")
    except Exception: pass
    cfg = get_cfg()
    cfg.set_new_allowed(True); cfg.merge_from_file(yaml_path); cfg.set_new_allowed(False)
    cfg.MODEL.WEIGHTS = weight_path
    cfg.MODEL.DEVICE = device
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST   = 0.6
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST      = RPN_POST_NMS_TOPK_TEST
    cfg.INPUT.MIN_SIZE_TEST = MIN_SIZE_TEST
    cfg.INPUT.MAX_SIZE_TEST = MAX_SIZE_TEST
    predictor = DefaultPredictor(cfg)
    predictor.aug = ResizeShortestEdge(int(MIN_SIZE_TEST), int(MAX_SIZE_TEST))

    # metadata for labels
    meta = MetadataCatalog.get("vg")
    def _read_txt(p):
        if p and os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as f:
                return [ln.strip() for ln in f if ln.strip()]
        return None
    objs = _read_txt(OBJ_VOCAB)
    attrs= _read_txt(ATTR_VOCAB)
    if objs:  meta.set(thing_classes=objs)
    else:     meta.set(thing_classes=[f"cls_{i}" for i in range(1600)])
    if attrs: meta.set(attr_classes=attrs)
    else:     meta.set(attr_classes=[f"attr_{i}" for i in range(400)])
    predictor.meta = meta
    return predictor

@torch.no_grad()
def butd_extract_36x2048(predictor, im_bgr: np.ndarray, num_objects: int = NUM_OBJECTS):
    model = predictor.model; model.eval()
    H, W = im_bgr.shape[:2]
    img_rgb = im_bgr[:, :, ::-1].copy()
    tfm = predictor.aug.get_transform(img_rgb)
    img_tfm = tfm.apply_image(img_rgb)
    tensor = torch.as_tensor(img_tfm.transpose(2, 0, 1).copy()).float().to(model.device)
    inputs = [{"image": tensor, "height": H, "width": W}]
    use_amp = (model.device.type == "cuda")
    with torch.cuda.amp.autocast(enabled=use_amp):
        images = model.preprocess_image(inputs)
        feats_all = model.backbone(images.tensor)
        proposals, _ = model.proposal_generator(images, feats_all, None)
        prop = proposals[0]
        in_feats = [feats_all[f] for f in model.roi_heads.in_features if f in feats_all]
        box_feats = model.roi_heads._shared_roi_transform(in_feats, [prop.proposal_boxes])
        if hasattr(model.roi_heads, "box_head"):
            pooled = model.roi_heads.box_head(box_feats)
        else:
            pooled = torch.nn.functional.adaptive_avg_pool2d(box_feats, (1,1)).flatten(1)
        outputs_tuple = model.roi_heads.box_predictor(pooled)
        if len(outputs_tuple) == 3:
            pred_cls_logits, pred_attr_logits, pred_deltas = outputs_tuple
        else:
            pred_cls_logits, pred_deltas = outputs_tuple
            pred_attr_logits = None
        fr_outputs = FastRCNNOutputs(model.roi_heads.box2box_transform,
                                     pred_cls_logits, pred_deltas, proposals, model.roi_heads.smooth_l1_beta)
        probs = fr_outputs.predict_probs()[0]
        boxes = fr_outputs.predict_boxes()[0]
    keep_ids = None; inst = None
    for th in np.arange(0.5, 1.01, 0.1):
        inst_try, ids = fast_rcnn_inference_single_image(
            boxes, probs, img_tfm.shape[:2], score_thresh=0.6, nms_thresh=th, topk_per_image=num_objects
        )
        if len(ids) == num_objects:
            inst, keep_ids = inst_try, ids; break
    if keep_ids is None:
        inst, keep_ids = fast_rcnn_inference_single_image(
            boxes, probs, img_tfm.shape[:2], score_thresh=0.0, nms_thresh=0.6, topk_per_image=num_objects
        )
    keep_ids = keep_ids.to(pooled.device)
    roi_feats = pooled.index_select(0, keep_ids).detach().cpu().numpy().astype("float32")
    # attach attributes if available
    if pred_attr_logits is not None:
        attr_prob = pred_attr_logits[..., :-1].softmax(-1)
        max_attr_prob, max_attr_label = attr_prob.max(-1)
        inst.attr_scores  = max_attr_prob.index_select(0, keep_ids).detach().cpu()
        inst.attr_classes = max_attr_label.index_select(0, keep_ids).detach().cpu()
    inst = detector_postprocess(inst.to("cpu"), H, W)
    return inst, roi_feats

# ====== Caption models ======
import torch.nn as nn
PAD, BOS, EOS, UNK = 0,1,2,3

class TopDown2LSTM(nn.Module):
    def __init__(self, feat_dim=2048, att_dim=512, hidden_dim=512, emb_dim=300, vocab_size=10000, drop=0.1):
        super().__init__()
        self.feat_proj = nn.Linear(feat_dim, att_dim)
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.att_lstm = nn.LSTMCell(emb_dim + att_dim, hidden_dim)
        self.lang_lstm= nn.LSTMCell(hidden_dim + att_dim, hidden_dim)
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim + att_dim, att_dim),
            nn.Tanh(),
            nn.Linear(att_dim, 1)
        )
        self.out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(drop)
    @torch.no_grad()
    def greedy(self, feats, bos_id=BOS, eos_id=EOS, max_len=20):
        self.eval()
        B, K, D = feats.size()
        f_att = self.feat_proj(feats)
        f_mean = f_att.mean(1)
        h1=c1 = feats.new_zeros(B, 512)
        h2=c2 = feats.new_zeros(B, 512)
        x = self.emb(torch.full((B,), bos_id, device=feats.device, dtype=torch.long))
        out_ids = []
        for t in range(max_len-1):
            inp1 = torch.cat([x, f_mean], dim=-1)
            h1, c1 = self.att_lstm(inp1, (h1,c1))
            h1_t = h1.unsqueeze(1).expand(-1,K,-1)
            e = self.attn(torch.cat([h1_t, f_att], dim=-1)).squeeze(-1)
            alpha = torch.softmax(e, dim=-1)
            ctx = (alpha.unsqueeze(-1) * f_att).sum(1)
            h2, c2 = self.lang_lstm(torch.cat([h1, ctx], dim=-1), (h2,c2))
            logit = self.out(self.dropout(h2))
            nxt = torch.argmax(logit, dim=-1)
            out_ids.append(nxt)
            x = self.emb(nxt)
        return torch.stack(out_ids, 1)

class QKVLSTM(nn.Module):
    def __init__(self, feat_dim, att_dim, hidden_dim, emb_dim, vocab_size, drop=0.1):
        super().__init__()
        self.feat_proj = nn.Linear(feat_dim, att_dim)
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.lstm = nn.LSTMCell(emb_dim + att_dim, hidden_dim)
        self.att_q = nn.Linear(hidden_dim, att_dim)
        self.att_k = nn.Linear(att_dim,  att_dim)
        self.att_v = nn.Linear(att_dim,  att_dim)
        self.out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(drop)
        self.scale = math.sqrt(att_dim)
    @torch.no_grad()
    def greedy(self, feats, bos_id=BOS, eos_id=EOS, max_len=20):
        self.eval()
        B, K, D = feats.size()
        f = self.feat_proj(feats)
        h=c = feats.new_zeros(B, self.lstm.hidden_size)
        x = self.emb(torch.full((B,), bos_id, device=feats.device, dtype=torch.long))
        out_ids = []
        for t in range(max_len-1):
            Q = self.att_q(h); K = self.att_k(f); V = self.att_v(f)
            scores = torch.einsum("bd,bkd->bk", Q, K) / self.scale
            alpha = torch.softmax(scores, dim=-1)
            ctx = torch.einsum("bk,bkd->bd", alpha, V)
            h, c = self.lstm(torch.cat([x, ctx], dim=-1), (h,c))
            logit = self.out(self.dropout(h))
            nxt = torch.argmax(logit, dim=-1)
            out_ids.append(nxt)
            x = self.emb(nxt)
        return torch.stack(out_ids, 1)

class UpDownDecoder(nn.Module):
    def __init__(self, vocab_size, emb=256, hid=512, feat_dim=2048, num_regions=36, pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=pad_idx)
        self.feat_proj = nn.Linear(feat_dim, hid)
        self.att_q = nn.Linear(hid, hid)
        self.att_k = nn.Linear(hid, hid)
        self.att_v = nn.Linear(hid, 1)
        self.lstm = nn.LSTMCell(emb + hid, hid)
        self.out  = nn.Linear(hid, vocab_size)
    def attend(self, feats, h):
        f = self.feat_proj(feats)
        q = self.att_q(h).unsqueeze(1).expand_as(f)
        e = self.att_v(torch.tanh(f + self.att_k(q))).squeeze(-1)
        alpha = torch.softmax(e, dim=-1)
        ctx = (alpha.unsqueeze(-1) * f).sum(1)
        return ctx
    @torch.no_grad()
    def greedy(self, feats, max_len=30, bos=1, eos=2):
        B,R,D = feats.shape
        device = feats.device
        h = torch.zeros(B, self.lstm.hidden_size, device=device)
        c = torch.zeros(B, self.lstm.hidden_size, device=device)
        cur = torch.full((B,), bos, dtype=torch.long, device=device)
        outs=[]
        for _ in range(max_len-1):
            emb_t = self.emb(cur)
            ctx   = self.attend(feats, h)
            inp   = torch.cat([emb_t, ctx], dim=-1)
            h,c   = self.lstm(inp, (h,c))
            cur   = self.out(h).argmax(-1)
            outs.append(cur)
        return torch.stack(outs,1)

# ====== vocab utils ======
def normalize_itos(itos_raw, vocab_size_hint=None) -> List[str]:
    if isinstance(itos_raw, list):
        return itos_raw
    if isinstance(itos_raw, dict):
        mapping = {}
        for k,v in itos_raw.items():
            try: ki = int(k)
            except Exception: continue
            mapping[ki] = v
        if mapping:
            max_idx = max(mapping.keys())
            N = max_idx+1
            if vocab_size_hint is not None: N = max(N, vocab_size_hint)
            out = ["<unk>"]*N
            for i,t in mapping.items():
                if 0 <= i < N: out[i]=t
            return out
    N = vocab_size_hint or 10000
    return ["<unk>"]*N

def find_token_id(candidates: List[str], itos_list: List[str], stoi: Dict[str,int], default: int):
    for c in candidates:
        if c in stoi: return stoi[c]
    lower = [t.lower() for t in itos_list]
    for c in candidates:
        if c.lower() in lower:
            return lower.index(c.lower())
    return default

def load_vocab(json_path: str):
    assert os.path.isfile(json_path), f"Missing vocab JSON: {json_path}"
    with open(json_path, "r", encoding="utf-8") as f:
        v = json.load(f)
    itos_raw = v.get("itos", [])
    stoi_raw = v.get("stoi", {})
    itos_list = normalize_itos(itos_raw, vocab_size_hint=len(itos_raw) if isinstance(itos_raw, list) else None)
    if not stoi_raw: stoi_raw = {w:i for i,w in enumerate(itos_list)}
    stoi = {str(k): int(v) for k,v in stoi_raw.items()} if isinstance(stoi_raw, dict) else stoi_raw
    BOS_id = find_token_id(["<bos>","<start>","<sos>"], itos_list, stoi, default=1)
    EOS_id = find_token_id(["<eos>","<end>","</s>"], itos_list, stoi, default=2)
    PAD_id = find_token_id(["<pad>","<blank>"], itos_list, stoi, default=0)
    return itos_list, {"BOS": BOS_id, "EOS": EOS_id, "PAD": PAD_id}

def ids_to_text(ids, itos_list, BOS_id=1, EOS_id=2, PAD_id=0):
    words = []
    N = len(itos_list)
    for i in ids:
        if i == EOS_id: break
        if i in (PAD_id, BOS_id): continue
        token = itos_list[i] if 0 <= i < N else "<unk>"
        words.append(token)
    return " ".join(words)

# ====== load caption model (arch autodetect) ======
def load_caption_model_from_ckpt(ckpt_path, itos_list, tok, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt.get("model", ckpt)
    # infer dims
    if "emb.weight" in sd:
        emb_dim = sd["emb.weight"].shape[1]
        vocab_size = len(itos_list)
    else:
        emb_dim = 300; vocab_size = len(itos_list)
    hid_dim = sd["out.weight"].shape[1] if "out.weight" in sd else 512

    # detect architecture
    if any(k.startswith("att_q") for k in sd.keys()) and "lstm.weight_ih" in sd:
        if "att_v.weight" in sd and sd["att_v.weight"].shape[0] == 1:
            model = UpDownDecoder(vocab_size, emb=emb_dim, hid=hid_dim, feat_dim=2048, pad_idx=tok["PAD"]).to(device)
        else:
            att_dim = sd["att_q.weight"].shape[0]
            model = QKVLSTM(2048, att_dim, hid_dim, emb_dim, vocab_size).to(device)
    else:
        model = TopDown2LSTM(2048, 512, hid_dim, emb_dim, vocab_size).to(device)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print("[WARN] Non-strict load. Missing:", missing, "| Unexpected:", unexpected)
    model.eval()
    return model

# ====== Visualization & Export helpers ======
def _build_labels(inst, meta):
    thing = getattr(meta, "thing_classes", None) or []
    attrc = getattr(meta, "attr_classes", None) or []
    labels = []
    has_attr = hasattr(inst, "attr_classes") and hasattr(inst, "attr_scores")
    classes = inst.pred_classes.tolist() if hasattr(inst, "pred_classes") else [None]*len(inst)
    scores  = inst.scores.tolist() if hasattr(inst, "scores") else [None]*len(inst)
    for i in range(len(inst)):
        cid = classes[i]
        obj = thing[cid] if (cid is not None and cid < len(thing)) else (f"cls_{cid}" if cid is not None else "?")
        sc  = scores[i] if scores[i] is not None else None
        if has_attr:
            aid = int(inst.attr_classes[i].item())
            asc = float(inst.attr_scores[i].item())
            att = attrc[aid] if aid < len(attrc) else f"attr_{aid}"
            if sc is not None:
                lab = f"{obj} ({sc:.2f}) | {att} ({asc:.2f})"
            else:
                lab = f"{obj} | {att} ({asc:.2f})"
        else:
            lab = f"{obj} ({sc:.2f})" if sc is not None else f"{obj}"
        labels.append(lab)
    return labels

def draw_boxes(img_bgr, boxes_xyxy, labels, topk=None, lw=2):
    img = img_bgr.copy()
    H,W = img.shape[:2]
    n = len(boxes_xyxy)
    if topk is not None: n = min(n, topk)
    for i in range(n):
        x1,y1,x2,y2 = boxes_xyxy[i]
        x1 = int(max(0,min(W-1,x1))); y1=int(max(0,min(H-1,y1)))
        x2 = int(max(0,min(W-1,x2))); y2=int(max(0,min(H-1,y2)))
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), lw)
        if labels and i < len(labels):
            txt = labels[i]
            (tw,th), baseline = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            bg_y1 = max(0, y1 - th - 6)
            bg_y2 = bg_y1 + th + 6
            cv2.rectangle(img, (x1, bg_y1), (x1+tw+6, bg_y2), (0,0,0), -1)
            cv2.putText(img, txt, (x1+3, bg_y2-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb

def encode_b64(arr: np.ndarray) -> str:
    arr = np.ascontiguousarray(arr.astype(np.float32))
    return base64.b64encode(arr.tobytes()).decode("utf-8")

# ====== Boot & Inference ======
from functools import lru_cache
@lru_cache()
def boot_all():
    assert VOCAB_JSON and os.path.isfile(VOCAB_JSON), f"Please set BUTD_VOCAB to a valid vocab.json. Got: {VOCAB_JSON}"
    itos_list, tok = load_vocab(VOCAB_JSON)
    predictor = build_butd_predictor(YAML, WEIGHT, device=DEVICE)
    captioner  = load_caption_model_from_ckpt(CAPTION_CKPT, itos_list, tok, DEVICE)
    return predictor, captioner, itos_list, tok

@torch.no_grad()
def run_pipeline(pil_image, k_draw=10, draw_labels=True):
    predictor, captioner, itos_list, tok = boot_all()
    if pil_image is None:
        return None, "Please upload an image.", None, None
    img_rgb = np.array(pil_image)
    img_bgr = img_rgb[:, :, ::-1].copy()
    H,W = img_bgr.shape[:2]
    inst, feats = butd_extract_36x2048(predictor, img_bgr, num_objects=NUM_OBJECTS)

    # sort by score for drawing & table
    inst_cpu = inst.to("cpu")
    if hasattr(inst_cpu, "scores") and len(inst_cpu) > 0:
        order = torch.argsort(inst_cpu.scores, descending=True)
        inst_cpu = inst_cpu[order]

    labels = _build_labels(inst_cpu, getattr(predictor, "meta", None)) if draw_labels else None
    boxes = inst_cpu.pred_boxes.tensor.numpy().astype(np.int32) if len(inst_cpu)>0 else np.zeros((0,4),dtype=np.int32)
    vis = draw_boxes(img_bgr, boxes, labels, topk=int(k_draw) if k_draw else None, lw=2)

    # caption
    feats_t = torch.from_numpy(feats).unsqueeze(0).to(DEVICE)
    try:
        seq = captioner.greedy(feats_t, bos_id=tok["BOS"], eos_id=tok["EOS"], max_len=MAX_LEN)[0].tolist()
    except TypeError:
        seq = captioner.greedy(feats_t, bos=tok["BOS"], eos=tok["EOS"], max_len=MAX_LEN)[0].tolist()
    caption = ids_to_text(seq, itos_list, BOS_id=tok["BOS"], EOS_id=tok["EOS"], PAD_id=tok["PAD"])

    # detections table rows
    has_attr = hasattr(inst_cpu, "attr_classes") and hasattr(inst_cpu, "attr_scores")
    rows = []
    for i in range(len(inst_cpu)):
        x1,y1,x2,y2 = [int(v) for v in inst_cpu.pred_boxes.tensor[i].tolist()]
        obj_id = int(inst_cpu.pred_classes[i].item()) if hasattr(inst_cpu,"pred_classes") else -1
        obj_sc = float(inst_cpu.scores[i].item()) if hasattr(inst_cpu,"scores") else float("nan")
        obj_lb = predictor.meta.thing_classes[obj_id] if 0<=obj_id<len(predictor.meta.thing_classes) else f"cls_{obj_id}"
        if has_attr:
            a_id = int(inst_cpu.attr_classes[i].item())
            a_sc = float(inst_cpu.attr_scores[i].item())
            a_lb = predictor.meta.attr_classes[a_id] if 0<=a_id<len(predictor.meta.attr_classes) else f"attr_{a_id}"
        else:
            a_id = -1; a_sc = float("nan"); a_lb = ""
        rows.append([i, x1,y1,x2,y2, obj_id,obj_lb,obj_sc, a_id,a_lb,a_sc])

    # Export TSV (BUTD style) for this image
    img_id = "uploaded"
    tsv_path = os.path.abspath("butd_last.tsv")
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("image_id\timage_w\timage_h\tnum_boxes\tboxes\tfeatures\n")
        f.write("\t".join([
            img_id, str(W), str(H), str(boxes.shape[0]),
            encode_b64(inst.pred_boxes.tensor.numpy().astype(np.float32)),
            encode_b64(feats.astype(np.float32))
        ]))
        f.write("\n")

    return vis, caption, rows, tsv_path

HEADERS = ["idx","x1","y1","x2","y2","obj_id","obj_label","obj_score","attr_id","attr_label","attr_score"]

with gr.Blocks(title=TITLE) as demo:
    gr.Markdown(f"### {TITLE}\n{DESC}")
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(type="pil", label="Upload image")
            k_draw = gr.Slider(1, NUM_OBJECTS, value=10, step=1, label="Top-K boxes to draw")
            draw_labels = gr.Checkbox(value=True, label="Draw labels/scores")
            btn = gr.Button("Run")
        with gr.Column(scale=2):
            out_img = gr.Image(type="numpy", label="Detections")
            out_txt = gr.Textbox(lines=2, label="Caption")
            # For older Gradio: set headers and column_count to same length
            out_tbl = gr.Dataframe(headers=HEADERS, label="Detections (boxes + labels)",
                                   interactive=False, column_count=len(HEADERS))
            out_tsv = gr.File(label="Download TSV (BUTD style)")
    btn.click(fn=run_pipeline, inputs=[inp, k_draw, draw_labels], outputs=[out_img, out_txt, out_tbl, out_tsv])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
