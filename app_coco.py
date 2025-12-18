
try:
    from PIL import Image as _PIL_Image
    _alias = {"ANTIALIAS": "LANCZOS", "LINEAR": "BILINEAR", "CUBIC": "BICUBIC"}
    try:
        Resampling = _PIL_Image.Resampling  # Pillow >= 10
        if not hasattr(_PIL_Image, "LANCZOS"):  setattr(_PIL_Image, "LANCZOS",  Resampling.LANCZOS)
        if not hasattr(_PIL_Image, "BILINEAR"): setattr(_PIL_Image, "BILINEAR", Resampling.BILINEAR)
        if not hasattr(_PIL_Image, "BICUBIC"):  setattr(_PIL_Image, "BICUBIC",  Resampling.BICUBIC)
    except Exception:
        pass
    for old_name, new_name in _alias.items():
        if not hasattr(_PIL_Image, old_name) and hasattr(_PIL_Image, new_name):
            setattr(_PIL_Image, old_name, getattr(_PIL_Image, new_name))
except Exception:
    pass
# -----------------------------------------------

import os, io, json, base64, collections, collections.abc, numpy as np, cv2, torch, gradio as gr, time, hashlib, warnings
from typing import List, Dict

# Optional: disable Gradio analytics notice
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

# ---- Patch gradio_client JSON schema path (bool schemas) ----
try:
    import gradio_client.utils as _gcu
    _orig_get_type = getattr(_gcu, "get_type", None)
    _orig__json_schema_to_python_type = getattr(_gcu, "_json_schema_to_python_type", None)

    def _safe_get_type(schema):
        if isinstance(schema, bool):
            return "Any"
        return _orig_get_type(schema) if _orig_get_type else "Any"

    def _safe__json_schema_to_python_type(schema, defs=None):
        if isinstance(schema, bool):
            return "Any"
        return _orig__json_schema_to_python_type(schema, defs) if _orig__json_schema_to_python_type else "Any"

    if _orig_get_type:
        _gcu.get_type = _safe_get_type
    if _orig__json_schema_to_python_type:
        _gcu._json_schema_to_python_type = _safe__json_schema_to_python_type
except Exception:
    pass
# ------------------------------------------------------------

# ========== General settings ==========
warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release")
try:
    cv2.setNumThreads(0)
except Exception:
    pass

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True
try: torch.set_float32_matmul_precision("high")
except Exception: pass

# ========== Paths via env ==========
YAML        = os.environ.get("BUTD_YAML",   r"/home/aurora/Image_Captioning_BUTD/.venv310/checkpoints/faster_rcnn_R_101_C4_attr_caffemaxpool.yaml")
WEIGHT      = os.environ.get("BUTD_WEIGHT", r"/home/aurora/Image_Captioning_BUTD/.venv310/checkpoints/faster_rcnn_from_caffe_attr.pkl")
VOCAB_JSON  = os.environ.get("BUTD_VOCAB",  r"/home/aurora/Image_Captioning_BUTD/.venv310/checkpoints/vocab_coco.json")
CE_CKPT     = os.environ.get("BUTD_CE_CKPT",   r"/home/aurora/Image_Captioning_BUTD/.venv310/checkpoints/xe_best.pt")
SCST_CKPT   = os.environ.get("BUTD_SCST_CKPT", r"/home/aurora/Image_Captioning_BUTD/.venv310/checkpoints/scst_best.pt")
OBJ_VOCAB   = os.environ.get("BUTD_OBJ_VOCAB",  r"/home/aurora/Image_Captioning_BUTD/.venv310/checkpoints/objects_vocab.txt")
ATTR_VOCAB  = os.environ.get("BUTD_ATTR_VOCAB", r"/home/aurora/Image_Captioning_BUTD/.venv310/checkpoints/attributes_vocab.txt")

NUM_OBJECTS = int(os.environ.get("BUTD_NUM_OBJECTS", "36"))
MIN_SIZE_TEST = int(os.environ.get("BUTD_MIN_TEST", "600"))
MAX_SIZE_TEST = int(os.environ.get("BUTD_MAX_TEST", "1000"))

# ========== Detectron2 imports ==========
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data.transforms import ResizeShortestEdge
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.structures import Boxes, Instances, pairwise_iou  # dùng hàm chính thức
from detectron2.layers import nms                                  # NMS chính thức
from detectron2.data import MetadataCatalog
import torch.nn.functional as F

# ========== Build predictor (BUTD C4 attr) ==========
def build_butd_predictor(yaml_path, weight_path, device=DEVICE):
    assert os.path.isfile(yaml_path), f"Missing YAML: {yaml_path}"
    assert os.path.isfile(weight_path), f"Missing WEIGHT: {weight_path}"

    cfg = get_cfg()
    cfg.set_new_allowed(True); cfg.merge_from_file(yaml_path); cfg.set_new_allowed(False)

    cfg.MODEL.WEIGHTS = weight_path
    cfg.MODEL.DEVICE  = device

    # C4: RPN dùng res4, conv_dim=512 để khớp checkpoint Caffe
    cfg.MODEL.RPN.HEAD_NAME   = "StandardRPNHead"
    cfg.MODEL.RPN.IN_FEATURES = ["res4"]
    cfg.MODEL.RPN.CONV_DIMS   = [512]

    # ROI heads Res5 trên đặc trưng res4
    cfg.MODEL.ROI_HEADS.IN_FEATURES = ["res4"]
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST   = 0.5

    # Cho phép nhiều candidate, phần sau tự NMS gộp lớp
    cfg.TEST.DETECTIONS_PER_IMAGE = max(int(cfg.TEST.DETECTIONS_PER_IMAGE), NUM_OBJECTS * 8, 400)

    # Resize như BUTD
    cfg.INPUT.MIN_SIZE_TEST = MIN_SIZE_TEST
    cfg.INPUT.MAX_SIZE_TEST = MAX_SIZE_TEST

    predictor = DefaultPredictor(cfg)
    predictor.aug = ResizeShortestEdge(int(MIN_SIZE_TEST), int(MAX_SIZE_TEST))

    # vocab (để hiển thị label đúng)
    meta = MetadataCatalog.get("vg")

    def _read_txt(p):
        if p and os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as f:
                return [ln.strip() for ln in f if ln.strip()]
        return None

    objs  = _read_txt(OBJ_VOCAB)
    attrs = _read_txt(ATTR_VOCAB)
    meta.set(thing_classes = objs  if objs  else [f"cls_{i}"  for i in range(1600)])
    meta.set(attr_classes  = attrs if attrs else [f"attr_{i}" for i in range(400)])
    predictor.meta = meta
    return predictor

# ========== Class-agnostic selection + NMS ==========
def _class_agnostic_select(pred_cls_logits, pred_deltas, proposals, box2box, topk, iou_thr=0.5):
    """
    Chọn 1 lớp tốt nhất cho mỗi RoI rồi làm NMS gộp lớp.
    Trả về: keep_idx (R_kept,), boxes_xyxy (N,4), scores (N,), classes (N,)
    """
    probs = F.softmax(pred_cls_logits, dim=-1)          # (R, K+1)
    scores = probs[:, :-1]                              # (R, K)

    max_scores, max_classes = scores.max(dim=-1)        # (R,), (R,)

    R, Kp1 = probs.shape
    K = Kp1 - 1
    all_boxes = box2box.apply_deltas(pred_deltas, proposals.proposal_boxes.tensor)  # (R, K*4)
    all_boxes = all_boxes.view(R, K, 4)
    arng = torch.arange(R, device=all_boxes.device)
    chosen_boxes = all_boxes[arng, max_classes]         # (R, 4)

    keep = nms(chosen_boxes, max_scores, iou_thr)       # class-agnostic NMS

    # sort by score and clip to topk
    keep_scores = max_scores[keep]
    order = torch.argsort(keep_scores, descending=True)[:topk]
    keep = keep.index_select(0, order)

    return keep, chosen_boxes[keep], max_scores[keep], max_classes[keep]

# ========== BUTD feature extraction (36 x 2048) ==========
@torch.no_grad()
def butd_extract_36x2048(predictor, im_bgr: np.ndarray, num_objects: int = NUM_OBJECTS):
    """
    Trả về (inst, roi_feats[ K x 2048 ])
    Pipeline: preprocess → backbone → RPN → ROIAlign → box_head → box_predictor
              → class-agnostic NMS → map boxes về size gốc → cắt đúng K features.
    """
    model = predictor.model
    model.eval()

    H, W = im_bgr.shape[:2]
    img_rgb = im_bgr[:, :, ::-1].copy()
    tfm = predictor.aug.get_transform(img_rgb)
    img_tfm = tfm.apply_image(img_rgb)
    chw  = np.ascontiguousarray(img_tfm.transpose(2, 0, 1))
    tensor = torch.from_numpy(chw).to(model.device, non_blocking=True).float()
    inputs = [{"image": tensor, "height": H, "width": W}]

    use_amp = (model.device.type == "cuda")
    with torch.inference_mode(), torch.amp.autocast(device_type="cuda", enabled=use_amp):
        # 1) preprocess + backbone
        images   = model.preprocess_image(inputs)
        feats_all= model.backbone(images.tensor)

        # 2) proposals
        proposals, _ = model.proposal_generator(images, feats_all, None)
        prop0 = proposals[0]

        # 3) ROIAlign + box head
        in_feats = [feats_all[f] for f in model.roi_heads.in_features if f in feats_all]
        box_feats = model.roi_heads._shared_roi_transform(in_feats, [prop0.proposal_boxes])

        if hasattr(model.roi_heads, "box_head"):
            pooled = model.roi_heads.box_head(box_feats)   # (R, 2048)
        else:
            pooled = torch.nn.functional.adaptive_avg_pool2d(box_feats, (1, 1)).flatten(1)

        # 4) predictor
        outputs = model.roi_heads.box_predictor(pooled)
        if isinstance(outputs, (list, tuple)) and len(outputs) == 3:
            pred_cls_logits, pred_attr_logits, pred_deltas = outputs
        else:
            pred_cls_logits, pred_deltas = outputs
            pred_attr_logits = None

        # 5) Class-agnostic NMS
        topk_for_nms = max(num_objects * 6, 300)
        box2box = model.roi_heads.box_predictor.box2box_transform
        keep_idx, kept_boxes, kept_scores, kept_classes = _class_agnostic_select(
            pred_cls_logits, pred_deltas, prop0, box2box, topk=topk_for_nms, iou_thr=0.5
        )

        # 6) Giới hạn đúng num_objects
        if keep_idx.numel() > num_objects:
            keep_idx = keep_idx[:num_objects]
            kept_boxes   = kept_boxes[:num_objects]
            kept_scores  = kept_scores[:num_objects]
            kept_classes = kept_classes[:num_objects]

        # 7) Instances ở size đã resize rồi map về (H,W) gốc
        inst_rsz = Instances(images.image_sizes[0])
        inst_rsz.pred_boxes   = Boxes(kept_boxes)
        inst_rsz.scores       = kept_scores
        inst_rsz.pred_classes = kept_classes

        inst = detector_postprocess(inst_rsz, H, W)

        # 8) Lấy ROI features theo keep_idx
        roi_feats = pooled.index_select(0, keep_idx.to(pooled.device)).detach().cpu().numpy().astype("float32")

        # 9) Attributes (nếu có)
        if pred_attr_logits is not None:
            attr_prob = pred_attr_logits[..., :-1].softmax(-1)
            max_attr_prob, max_attr_label = attr_prob.max(-1)
            inst.attr_scores  = max_attr_prob.index_select(0, keep_idx).detach().cpu()
            inst.attr_classes = max_attr_label.index_select(0, keep_idx).detach().cpu()

    return inst.to("cpu"), roi_feats

# ========== UpDown 2-LSTM decoder ==========
import torch.nn as nn
PAD, BOS, EOS, UNK = 0,1,2,3

class UpDownDecoder(nn.Module):
    def __init__(self, vocab_size, emb=256, hid=512, feat_dim=2048, pad_idx=0, dropout=0.3):
        super().__init__()
        self.pad_idx = pad_idx
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=pad_idx)
        self.feat_proj = nn.Linear(feat_dim, hid)
        self.feat_ln = nn.LayerNorm(hid)
        self.att_lstm = nn.LSTMCell(emb + hid + hid, hid)
        self.lang_lstm = nn.LSTMCell(hid + hid, hid)
        self.att_v = nn.Linear(hid, hid)
        self.att_h = nn.Linear(hid, hid)
        self.att_u = nn.Linear(hid, 1)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hid, vocab_size)
        self.lstm = type('D', (), {'hidden_size': hid})()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
        for lstm in [self.att_lstm, self.lang_lstm]:
            for name,p in lstm.named_parameters():
                if 'bias' in name:
                    H = p.shape[0]//4
                    p.data[H:2*H] = 1.0

    def _attend(self, f, h_att):
        B,R,H = f.size()
        h = self.att_h(h_att).unsqueeze(1).expand(B,R,H)
        e = self.att_u(torch.tanh(self.att_v(f) + h)).squeeze(-1)
        alpha = torch.softmax(e, dim=-1)
        v_hat = torch.bmm(alpha.unsqueeze(1), f).squeeze(1)
        return v_hat, alpha

    def forward(self, feats, caps):
        B,R,_ = feats.shape; T = caps.size(1)
        f = self.feat_ln(self.feat_proj(feats))
        f_mean = f.mean(1)
        H = self.att_lstm.hidden_size
        h_att = feats.new_zeros(B,H); c_att = feats.new_zeros(B,H)
        h_lang = feats.new_zeros(B,H); c_lang = feats.new_zeros(B,H)
        logits = []
        for t in range(T-1):
            w_t = self.emb(caps[:,t])
            h_att, c_att = self.att_lstm(torch.cat([w_t, f_mean, h_lang], -1), (h_att, c_att))
            v_hat, _ = self._attend(f, h_att)
            h_lang, c_lang = self.lang_lstm(torch.cat([v_hat, h_att], -1), (h_lang, c_lang))
            logits.append(self.out(self.dropout(h_lang)))
        return torch.stack(logits, 1)

    @torch.no_grad()
    def greedy(self, feats, bos=1, eos=2, max_len=30):
        device = feats.device
        B,R,_ = feats.shape; H = self.att_lstm.hidden_size
        f = self.feat_ln(self.feat_proj(feats)); f_mean = f.mean(1)
        h_att = feats.new_zeros(B,H); c_att = feats.new_zeros(B,H)
        h_lang = feats.new_zeros(B,H); c_lang = feats.new_zeros(B,H)
        cur = torch.full((B,), bos, dtype=torch.long, device=device)
        outs=[]
        for _ in range(max_len-1):
            w_t = self.emb(cur)
            h_att, c_att = self.att_lstm(torch.cat([w_t, f_mean, h_lang], -1), (h_att, c_att))
            v_hat, _ = self._attend(f, h_att)
            h_lang, c_lang = self.lang_lstm(torch.cat([v_hat, h_att], -1), (h_lang, c_lang))
            cur = self.out(h_lang).argmax(-1)
            outs.append(cur)
        return torch.stack(outs, 1)

    @torch.no_grad()
    def beam_search(self, feats, bos=1, eos=2, pad=0, max_len=30, beam_size=5, length_penalty=0.7, no_repeat_ngram=3):
        device = feats.device
        B,R,_ = feats.shape; H = self.att_lstm.hidden_size
        f = self.feat_ln(self.feat_proj(feats)); f_mean = f.mean(1)
        beams = torch.full((B,1,1), bos, dtype=torch.long, device=device)
        scores = torch.zeros(B,1, device=device)
        hA = feats.new_zeros(B,1,H); cA = feats.new_zeros(B,1,H)
        hL = feats.new_zeros(B,1,H); cL = feats.new_zeros(B,1,H)
        finished = [[] for _ in range(B)]
        import torch.nn.functional as F
        for _ in range(1, max_len):
            K = beams.size(1)
            f_rep = f.unsqueeze(1).expand(B,K,R,H).reshape(B*K,R,H)
            fmean_rep = f_mean.unsqueeze(1).expand(B,K,H).reshape(B*K,H)
            prev = beams[:,:,-1].reshape(B*K)
            w_t = self.emb(prev)
            hA_ = hA.reshape(B*K,H); cA_ = cA.reshape(B*K,H)
            hL_ = hL.reshape(B*K,H); cL_ = cL.reshape(B*K,H)
            hA_, cA_ = self.att_lstm(torch.cat([w_t, fmean_rep, hL_], -1), (hA_, cA_))
            h = self.att_h(hA_).unsqueeze(1).expand(B*K,R,H)
            e = self.att_u(torch.tanh(self.att_v(f_rep) + h)).squeeze(-1)
            alpha = torch.softmax(e, -1)
            v_hat = torch.bmm(alpha.unsqueeze(1), f_rep).squeeze(1)
            hL_, cL_ = self.lang_lstm(torch.cat([v_hat, hA_], -1), (hL_, cL_))
            logp = F.log_softmax(self.out(hL_), -1)
            if no_repeat_ngram and no_repeat_ngram>0:
                toks = beams.detach().cpu().numpy()
                V = logp.size(1)
                from collections import defaultdict
                for bi in range(B*K):
                    b,k = divmod(bi, K)
                    seq = toks[b,k,:].tolist()
                    if len(seq) >= no_repeat_ngram-1:
                        grams=defaultdict(set)
                        for i in range(len(seq)-no_repeat_ngram+1):
                            grams[tuple(seq[i:i+no_repeat_ngram-1])].add(seq[i+no_repeat_ngram-1])
                        blocked = grams.get(tuple(seq[-(no_repeat_ngram-1):]), set())
                        if blocked:
                            logp[bi, list(blocked)] = -1e9
            cand = scores.unsqueeze(-1) + logp.view(B,K,-1)
            V = logp.size(1)
            cand = cand.view(B, -1)
            topk = torch.topk(cand, k=min(beam_size, cand.size(1)), dim=-1)
            next_scores, next_ids = topk.values, topk.indices
            next_beam = next_ids // V; next_tok = next_ids % V
            beams = torch.gather(beams, 1, next_beam.unsqueeze(-1).expand(-1,-1,beams.size(-1)))
            beams = torch.cat([beams, next_tok.unsqueeze(-1)], -1)
            hA = torch.gather(hA_.view(B,K,H), 1, next_beam.unsqueeze(-1).expand(-1,-1,H))
            cA = torch.gather(cA_.view(B,K,H), 1, next_beam.unsqueeze(-1).expand(-1,-1,H))
            hL = torch.gather(hL_.view(B,K,H), 1, next_beam.unsqueeze(-1).expand(-1,-1,H))
            cL = torch.gather(cL_.view(B,K,H), 1, next_beam.unsqueeze(-1).expand(-1,-1,H))
            scores = next_scores
            for b in range(B):
                for k in range(beams.size(1)):
                    if beams[b,k,-1].item()==eos:
                        L = beams[b,k].size(0)
                        lp = ((5+L)**length_penalty)/(6**length_penalty)
                        finished[b].append((scores[b,k].item()/lp, beams[b,k].clone()))
        outs=[]
        for b in range(B):
            if finished[b]:
                best = max(finished[b], key=lambda x:x[0])[1]
                outs.append(best[1:])
            else:
                j = int(scores[b].argmax().item())
                outs.append(beams[b,j,1:])
        maxL = max(s.size(0) for s in outs)
        res = torch.full((B,maxL), pad, dtype=torch.long, device=device)
        for i,s in enumerate(outs): res[i,:s.size(0)]=s
        return res

# ========== vocab & loader ==========
def load_vocab(json_path: str):
    assert os.path.isfile(json_path), f"Missing vocab JSON: {json_path}"
    with open(json_path, "r", encoding="utf-8") as f:
        v = json.load(f)
    itos = v.get("itos", v)
    if isinstance(itos, dict):
        N = max(int(k) for k in itos.keys())+1
        tmp = ["<unk>"]*N
        for k,s in itos.items():
            ki = int(k); 
            if 0<=ki<N: tmp[ki]=s
        itos = tmp
    stoi = v.get("stoi", {w:i for i,w in enumerate(itos)})
    def _find(cands, default):
        for c in cands:
            if c in stoi: return stoi[c]
        low = [t.lower() for t in itos]
        for c in cands:
            if c.lower() in low: return low.index(c.lower())
        return default
    BOS_id = _find(["<bos>","<start>","<sos>"], 1)
    EOS_id = _find(["<eos>","<end>","</s>"], 2)
    PAD_id = _find(["<pad>","<blank>"], 0)
    return itos, {"BOS":BOS_id, "EOS":EOS_id, "PAD":PAD_id}

def ids_to_text(ids, itos, BOS_id=1, EOS_id=2, PAD_id=0):
    out=[]
    N=len(itos)
    for t in ids:
        if t==EOS_id: break
        if t in (BOS_id, PAD_id): continue
        out.append(itos[t] if 0<=t<N else "<unk>")
    return " ".join(out)

def load_updown_from_ckpt(ckpt_path, itos, tok, device):
    sd = torch.load(ckpt_path, map_location=device)
    sd = sd.get("model", sd)
    vocab_size = len(itos)
    emb = sd["emb.weight"].shape[1] if "emb.weight" in sd else 256
    if "att_lstm.weight_ih" in sd:
        H = sd["att_lstm.weight_ih"].shape[0] // 4
    elif "lang_lstm.weight_hh" in sd:
        H = sd["lang_lstm.weight_hh"].shape[0] // 4
    else:
        H = 512
    feat_dim = sd["feat_proj.weight"].shape[1] if "feat_proj.weight" in sd else 2048
    model = UpDownDecoder(vocab_size, emb=emb, hid=H, feat_dim=feat_dim, pad_idx=tok["PAD"]).to(device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print("[WARN] Missing:", missing, "| Unexpected:", unexpected)
    model.eval(); return model

from functools import lru_cache
@lru_cache()
def boot_backbone_and_vocab():
    assert VOCAB_JSON and os.path.isfile(VOCAB_JSON), f"Please set BUTD_VOCAB to a valid vocab.json. Got: {VOCAB_JSON}"
    itos_list, tok = load_vocab(VOCAB_JSON)
    predictor = build_butd_predictor(YAML, WEIGHT, device=DEVICE)
    return predictor, itos_list, tok

@lru_cache()
def boot_captioner_ce():
    _, itos_list, tok = boot_backbone_and_vocab()
    if not CE_CKPT or not os.path.isfile(CE_CKPT):
        raise FileNotFoundError(f"CE checkpoint not found: {CE_CKPT}")
    return load_updown_from_ckpt(CE_CKPT, itos_list, tok, DEVICE)

@lru_cache()
def boot_captioner_scst():
    _, itos_list, tok = boot_backbone_and_vocab()
    if not SCST_CKPT or not os.path.isfile(SCST_CKPT):
        raise FileNotFoundError(f"SCST checkpoint not found: {SCST_CKPT}")
    return load_updown_from_ckpt(SCST_CKPT, itos_list, tok, DEVICE)

# ========== Visualization helpers ==========
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
            lab = f"{obj} ({sc:.2f}) | {att} ({asc:.2f})" if sc is not None else f"{obj} | {att} ({asc:.2f})"
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

# ========== Inference pipeline ==========
_feat_cache = {}
def _img_hash(pil_image) -> str:
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return hashlib.md5(buf.getvalue()).hexdigest()

@torch.no_grad()
def run_pipeline(pil_image, k_draw, draw_labels, cache_feats, decode_strategy, beam_size, len_penalty, no_repeat, max_len):
    predictor, itos_list, tok = boot_backbone_and_vocab()
    cap_ce = boot_captioner_ce()
    cap_scst = boot_captioner_scst()

    if pil_image is None:
        return None, "Please upload an image.", "Please upload an image.", ""

    t0 = time.time()
    img_rgb = np.array(pil_image)
    img_bgr = img_rgb[:, :, ::-1].copy()

    key = _img_hash(pil_image) if cache_feats else None
    if key and key in _feat_cache:
        inst, feats = _feat_cache[key]
    else:
        inst, feats = butd_extract_36x2048(predictor, img_bgr, num_objects=NUM_OBJECTS)
        if key: _feat_cache[key] = (inst, feats)
    t_feat = time.time() - t0

    inst_cpu = inst.to("cpu")
    if hasattr(inst_cpu, "scores") and len(inst_cpu) > 0:
        order = torch.argsort(inst_cpu.scores, descending=True)
        inst_cpu = inst_cpu[order]

    labels = _build_labels(inst_cpu, getattr(predictor, "meta", None)) if draw_labels else None
    boxes = inst_cpu.pred_boxes.tensor.numpy().astype(np.int32) if len(inst_cpu)>0 else np.zeros((0,4),dtype=np.int32)
    vis = draw_boxes(img_bgr, boxes, labels, topk=int(k_draw) if k_draw else None, lw=2)

    feats_t = torch.from_numpy(feats.copy()).unsqueeze(0).to(DEVICE)
    t1 = time.time()
    if decode_strategy == "beam":
        seq_ce = cap_ce.beam_search(feats_t, bos=tok["BOS"], eos=tok["EOS"], pad=tok["PAD"],
                                    max_len=int(max_len), beam_size=int(beam_size),
                                    length_penalty=float(len_penalty), no_repeat_ngram=int(no_repeat))[0].tolist()
        seq_sc = cap_scst.beam_search(feats_t, bos=tok["BOS"], eos=tok["EOS"], pad=tok["PAD"],
                                      max_len=int(max_len), beam_size=int(beam_size),
                                      length_penalty=float(len_penalty), no_repeat_ngram=int(no_repeat))[0].tolist()
    else:
        seq_ce = cap_ce.greedy(feats_t, bos=tok["BOS"], eos=tok["EOS"], max_len=int(max_len))[0].tolist()
        seq_sc = cap_scst.greedy(feats_t, bos=tok["BOS"], eos=tok["EOS"], max_len=int(max_len))[0].tolist()
    t_decode = time.time() - t1

    cap_text_ce = ids_to_text(seq_ce, itos_list, BOS_id=tok["BOS"], EOS_id=tok["EOS"], PAD_id=tok["PAD"])
    cap_text_sc = ids_to_text(seq_sc, itos_list, BOS_id=tok["BOS"], EOS_id=tok["EOS"], PAD_id=tok["PAD"])
    timing = f"features: {t_feat*1000:.0f} ms | decode: {t_decode*1000:.0f} ms (strategy={decode_strategy}, beam={beam_size}, len_pen={len_penalty}, nrep={no_repeat}, L={max_len})"

    return vis, f"[CE] {cap_text_ce}", f"[SCST] {cap_text_sc}", timing

# ========== UI ==========
with gr.Blocks(title="COCO Captioning (BUTD + UpDown) – CE vs SCST", css=".gr-button {min-width: 120px;}") as demo:
    gr.Markdown("### Image Captioning on COCO using Bottom-Up Top-Down (BUTD) features with UpDown 2-LSTM decoder\n")
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(type="pil", label="Upload image")
            k_draw = gr.Slider(1, NUM_OBJECTS, value=10, step=1, label="Top-K boxes to draw")
            draw_labels = gr.Checkbox(value=True, label="Draw labels/scores")
            cache_feats = gr.Checkbox(value=True, label="Cache features by image")
            decode_strategy = gr.Dropdown(["beam","greedy"], value="beam", label="Decode strategy")
            beam_size = gr.Slider(1, 10, value=5, step=1, label="Beam size")
            len_penalty = gr.Slider(0.0, 2.0, value=0.7, step=0.05, label="Length penalty")
            no_repeat = gr.Slider(0, 6, value=3, step=1, label="No-repeat n-gram (0=off)")
            max_len = gr.Slider(5, 40, value=20, step=1, label="Max length")
            btn = gr.Button("Run", variant="primary")
        with gr.Column(scale=2):
            out_img = gr.Image(type="numpy", label="Detections")
            with gr.Row():
                out_ce = gr.Textbox(lines=2, label="Caption (CE)")
                out_sc = gr.Textbox(lines=2, label="Caption (SCST)")
            timing = gr.Textbox(lines=1, label="Timing (ms)")

    btn.click(fn=run_pipeline,
              inputs=[inp, k_draw, draw_labels, cache_feats, decode_strategy, beam_size, len_penalty, no_repeat, max_len],
              outputs=[out_img, out_ce, out_sc, timing])

    # auto-refresh on controls
    inp.change(fn=run_pipeline,
               inputs=[inp, k_draw, draw_labels, cache_feats, decode_strategy, beam_size, len_penalty, no_repeat, max_len],
               outputs=[out_img, out_ce, out_sc, timing])
    for ctrl in [k_draw, draw_labels, cache_feats, decode_strategy, beam_size, len_penalty, no_repeat, max_len]:
        ctrl.change(fn=run_pipeline,
                    inputs=[inp, k_draw, draw_labels, cache_feats, decode_strategy, beam_size, len_penalty, no_repeat, max_len],
                    outputs=[out_img, out_ce, out_sc, timing])

if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_api=False,
        max_threads=8
    )
