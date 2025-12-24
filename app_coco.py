
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
import random
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

VOCAB_JSON_EN  = os.environ.get("BUTD_VOCAB_EN",  r"/home/aurora/Image_Captioning_BUTD/.venv310/checkpoints/vocab_coco.json")
CE_CKPT_EN     = os.environ.get("BUTD_CE_CKPT_EN",   r"/home/aurora/Image_Captioning_BUTD/.venv310/checkpoints/xe_best.pt")
SCST_CKPT_EN   = os.environ.get("BUTD_SCST_CKPT_EN", r"/home/aurora/Image_Captioning_BUTD/.venv310/checkpoints/scst_best.pt")

VOCAB_JSON_VI  = os.environ.get("BUTD_VOCAB_VI",  r"/home/aurora/Image_Captioning_BUTD/.venv310/checkpoints/vocab_butd_vi.json")
CE_CKPT_VI     = os.environ.get("BUTD_CE_CKPT_VI",   r"/home/aurora/Image_Captioning_BUTD/.venv310/checkpoints/butd_vietnamese_best.pt")
SCST_CKPT_VI   = os.environ.get("BUTD_SCST_CKPT_VI", r"/home/aurora/Image_Captioning_BUTD/.venv310/checkpoints/butd_scst_optimized.pt")

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
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST   = 0.5

    # Cho phép nhiều candidate, phần sau tự NMS gộp lớp
    cfg.TEST.DETECTIONS_PER_IMAGE = max(int(cfg.TEST.DETECTIONS_PER_IMAGE), NUM_OBJECTS * 10, 400)

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

from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPModel, CLIPProcessor
# ========== BLIP MODEL ==========
blip_processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(DEVICE)
blip_model.eval()
# ========== BLIP CAPTION GENERATOR ==========
@torch.no_grad()
def blip_generate_caption(image_pil):
    inputs = blip_processor(image_pil, return_tensors="pt").to(DEVICE)

    output = blip_model.generate(
        **inputs,
        max_length=30,
        num_beams=5
    )

    caption = blip_processor.decode(
        output[0],
        skip_special_tokens=True
    )
    return caption
# ========== CLIP MODEL ==========
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
).to(device)
clip_model.eval()

clip_processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)
# ========== CLIP SCORE ON IMAGE ==========
@torch.no_grad()
def compute_clip_score(image, caption: str) -> float:
    inputs = clip_processor(
        text=[caption],
        images=image,
        return_tensors="pt",
        padding=True
    ).to(device)

    outputs = clip_model(**inputs)

    img_feat = outputs.image_embeds
    txt_feat = outputs.text_embeds
    # normalize
    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
    txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

    score = (img_feat * txt_feat).sum(dim=-1).item()
    return score
# ========== Class-agnostic selection + NMS ==========
def _class_agnostic_select(pred_cls_logits, pred_deltas, proposals, box2box, topk, iou_thr=0.45):
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
        topk_for_nms = max(num_objects * 4, 200)
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
        f_mean = torch.sum(f * torch.softmax(f.norm(dim=-1), dim=1).unsqueeze(-1), dim=1)
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
        f = self.feat_ln(self.feat_proj(feats)); f_mean = torch.sum(f * torch.softmax(f.norm(dim=-1), dim=1).unsqueeze(-1), dim=1)
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
    def beam_search(self, feats, bos=1, eos=2, pad=0, max_len=25, beam_size=5, length_penalty=1.2, no_repeat_ngram=2):
        device = feats.device
        B,R,_ = feats.shape; H = self.att_lstm.hidden_size
        f = self.feat_ln(self.feat_proj(feats))
        f_mean = torch.sum(
            f * torch.softmax(f.norm(dim=-1), dim=1).unsqueeze(-1),
            dim=1
        )
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
    predictor = build_butd_predictor(
        YAML,
        WEIGHT,
        device=DEVICE
    )
    return predictor
@lru_cache()
def boot_captioners_bilingual():
    # ================== ENGLISH ==================
    assert os.path.isfile(VOCAB_JSON_EN), f"Missing {VOCAB_JSON_EN}"
    assert os.path.isfile(CE_CKPT_EN), f"Missing {CE_CKPT_EN}"

    itos_en, tok_en = load_vocab(VOCAB_JSON_EN)

    ce_en = load_updown_from_ckpt(
        CE_CKPT_EN,
        itos_en,
        tok_en,
        DEVICE
    )

    scst_en = None
    if SCST_CKPT_EN and os.path.isfile(SCST_CKPT_EN):
        scst_en = load_updown_from_ckpt(
            SCST_CKPT_EN,
            itos_en,
            tok_en,
            DEVICE
        )

    # ================== VIETNAMESE ==================
    assert os.path.isfile(VOCAB_JSON_VI), f"Missing {VOCAB_JSON_VI}"
    assert os.path.isfile(CE_CKPT_VI), f"Missing {CE_CKPT_VI}"

    itos_vi, tok_vi = load_vocab(VOCAB_JSON_VI)

    ce_vi = load_updown_from_ckpt(
        CE_CKPT_VI,
        itos_vi,
        tok_vi,
        DEVICE
    )

    scst_vi = None
    if SCST_CKPT_VI and os.path.isfile(SCST_CKPT_VI):
        scst_vi = load_updown_from_ckpt(
            SCST_CKPT_VI,
            itos_vi,
            tok_vi,
            DEVICE
        )

    return {
        "en": {
            "itos": itos_en,
            "tok": tok_en,
            "ce": ce_en,
            "scst": scst_en
        },
        "vi": {
            "itos": itos_vi,
            "tok": tok_vi,
            "ce": ce_vi,
            "scst": scst_vi   # ✅ SCST VI HỢP LỆ
        }
    }

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

def draw_boxes(img_bgr, boxes_xyxy, labels=None, topk=None, lw=1):
    import random
    img = img_bgr.copy()
    H, W = img.shape[:2]
    n = len(boxes_xyxy)
    if topk is not None:
        n = min(n, topk)

    random.seed(0)  # ổn định màu

    for i in range(n):
        x1, y1, x2, y2 = boxes_xyxy[i]
        x1 = int(max(0, min(W-1, x1)))
        y1 = int(max(0, min(H-1, y1)))
        x2 = int(max(0, min(W-1, x2)))
        y2 = int(max(0, min(H-1, y2)))

        color = (
            random.randint(60,255),
            random.randint(60,255),
            random.randint(60,255)
        )

        cv2.rectangle(img, (x1,y1), (x2,y2), color, lw)

        if labels and i < len(labels):
            txt = labels[i]
            (tw,th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(img, (x1, y1-th-6), (x1+tw+4, y1), color, -1)
            cv2.putText(img, txt, (x1+2, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ========== Inference pipeline ==========
_feat_cache = {}
def _img_hash(pil_image) -> str:
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return hashlib.md5(buf.getvalue()).hexdigest()

@torch.no_grad()
def run_pipeline(
    pil_image,
    k_draw,
    draw_labels,
    cache_feats,
    decode_strategy,
    beam_size,
    len_penalty,
    no_repeat,
    max_len
):
    if pil_image is None:
        return None, "", "", "", ""

    # ===== 1. BLIP caption =====
    cap_blip_txt = blip_generate_caption(pil_image)

    # ===== 2. Load backbone + captioners =====
    predictor = boot_backbone_and_vocab()
    caps = boot_captioners_bilingual()

    # ===== 3. Image → BGR =====
    img_rgb = np.array(pil_image)
    img_bgr = img_rgb[:, :, ::-1].copy()

    # ===== 4. Feature extraction (cache nếu có) =====
    cache_key = None
    if cache_feats:
        cache_key = hash(pil_image.tobytes())

    if cache_feats and hasattr(run_pipeline, "_feat_cache") and cache_key in run_pipeline._feat_cache:
        inst, feats = run_pipeline._feat_cache[cache_key]
        t_feat = 0.0
    else:
        t0 = time.time()
        inst, feats = butd_extract_36x2048(predictor, img_bgr)
        t_feat = time.time() - t0

        if cache_feats:
            if not hasattr(run_pipeline, "_feat_cache"):
                run_pipeline._feat_cache = {}
            run_pipeline._feat_cache[cache_key] = (inst, feats)

    feats_t = torch.from_numpy(feats).unsqueeze(0).to(DEVICE)

    # ===== 5. Visualization =====
    inst_cpu = inst.to("cpu")
    boxes = inst_cpu.pred_boxes.tensor.numpy().astype(np.int32)
    labels = _build_labels(inst_cpu, predictor.meta) if draw_labels else None
    vis = draw_boxes(img_bgr, boxes, labels, topk=k_draw)

    # =================================================
    # ===== 6. Decode EN + VI (CE & SCST) ============
    # =================================================
    t1 = time.time()

    def decode(model, tok):
        if decode_strategy == "beam":
            return model.beam_search(
                feats_t,
                bos=tok["BOS"],
                eos=tok["EOS"],
                pad=tok["PAD"],
                max_len=int(max_len),
                beam_size=int(beam_size),
                length_penalty=float(len_penalty),
                no_repeat_ngram=int(no_repeat)
            )[0]
        else:
            return model.greedy(feats_t)[0]

    # ---------- EN ----------
    en = caps["en"]
    seq_ce_en = decode(en["ce"], en["tok"])
    cap_ce_en = ids_to_text(seq_ce_en.tolist(), en["itos"], en["tok"]["BOS"], en["tok"]["EOS"], en["tok"]["PAD"])

    cap_sc_en = ""
    if en["scst"] is not None:
        seq_sc_en = decode(en["scst"], en["tok"])
        cap_sc_en = ids_to_text(seq_sc_en.tolist(), en["itos"], en["tok"]["BOS"], en["tok"]["EOS"], en["tok"]["PAD"])

    # ---------- VI ----------
    vi = caps["vi"]
    seq_ce_vi = decode(vi["ce"], vi["tok"])
    cap_ce_vi = ids_to_text(seq_ce_vi.tolist(), vi["itos"], vi["tok"]["BOS"], vi["tok"]["EOS"], vi["tok"]["PAD"])

    cap_sc_vi = ""
    if vi["scst"] is not None:
        seq_sc_vi = decode(vi["scst"], vi["tok"])
        cap_sc_vi = ids_to_text(seq_sc_vi.tolist(), vi["itos"], vi["tok"]["BOS"], vi["tok"]["EOS"], vi["tok"]["PAD"])

    t_dec = time.time() - t1

    # ===== 7. CLIPScore =====
    clip_ce_en = compute_clip_score(pil_image, cap_ce_en)
    clip_ce_vi = compute_clip_score(pil_image, cap_ce_vi)
    clip_sc_en = compute_clip_score(pil_image, cap_sc_en) if cap_sc_en else None
    clip_sc_vi = compute_clip_score(pil_image, cap_sc_vi) if cap_sc_vi else None
    clip_blip  = compute_clip_score(pil_image, cap_blip_txt)

    # ===== 8. Outputs =====
    out_ce_txt = (
        f"[BUTD-CE | EN]\n{cap_ce_en}\nCLIPScore: {clip_ce_en:.3f}\n\n"
        f"[BUTD-CE | VI]\n{cap_ce_vi}\nCLIPScore: {clip_ce_vi:.3f}"
    )

    out_sc_txt = ""
    if cap_sc_en:
        out_sc_txt += f"[BUTD-SCST | EN]\n{cap_sc_en}\nCLIPScore: {clip_sc_en:.3f}\n\n"
    if cap_sc_vi:
        out_sc_txt += f"[BUTD-SCST | VI]\n{cap_sc_vi}\nCLIPScore: {clip_sc_vi:.3f}"

    out_bl_txt = f"[BLIP]\n{cap_blip_txt}\nCLIPScore: {clip_blip:.3f}"

    timing = f"features {t_feat*1000:.0f} ms | decode {t_dec*1000:.0f} ms"

    return vis, out_ce_txt, out_sc_txt, out_bl_txt, timing

# ========== UI ==========
with gr.Blocks(title="Image Captioning") as demo:
    gr.Markdown(
        "## BOTTOM-UP AND TOP-DOWN IMAGE CAPTIONING\n"
        "**CE + SCST: English + Vietnamese **"
    )

    with gr.Row():
        # -------- Controls --------
        with gr.Column(scale=1):
            inp = gr.Image(type="pil", label="Upload image")

            k_draw = gr.Slider(
                minimum=1,
                maximum=36,
                value=10,
                step=1,
                label="Top-K detected regions"
            )

            draw_labels = gr.Checkbox(
                value=True,
                label="Draw object / attribute labels"
            )

            cache_feats = gr.Checkbox(
                value=True,
                label="Cache extracted features"
            )

            decode_strategy = gr.Dropdown(
                choices=["beam", "greedy"],
                value="beam",
                label="Decoding strategy"
            )

            beam_size = gr.Slider(
                minimum=1,
                maximum=10,
                value=7,
                step=1,
                label="Beam size (for beam search)"
            )

            len_penalty = gr.Slider(
                minimum=0.0,
                maximum=2.0,
                value=1.2,
                step=0.05,
                label="Length penalty"
            )

            no_repeat = gr.Slider(
                minimum=0,
                maximum=6,
                value=2,
                step=1,
                label="No-repeat ngram size"
            )

            max_len = gr.Slider(
                minimum=5,
                maximum=40,
                value=25,
                step=1,
                label="Maximum caption length"
            )

            btn = gr.Button("Run Captioning", variant="primary")

        # -------- Outputs --------
        with gr.Column(scale=2):
            out_img = gr.Image(label="Detected regions (Bottom-Up Attention)")

            out_ce = gr.Textbox(
                label="BUTD – Cross Entropy (English + Vietnamese)",
                lines=6
            )

            out_sc = gr.Textbox(
                label="BUTD – SCST (English + Vietnamese)",
                lines=4
            )

            out_blip = gr.Textbox(
                label="BLIP baseline (English)",
                lines=3
            )

            timing = gr.Textbox(label="Runtime")

    # -------- Events --------
    btn.click(
        fn=run_pipeline,
        inputs=[
            inp,
            k_draw,
            draw_labels,
            cache_feats,
            decode_strategy,
            beam_size,
            len_penalty,
            no_repeat,
            max_len
        ],
        outputs=[
            out_img,
            out_ce,
            out_sc,
            out_blip,
            timing
        ]
    )

    inp.change(
        fn=run_pipeline,
        inputs=[
            inp,
            k_draw,
            draw_labels,
            cache_feats,
            decode_strategy,
            beam_size,
            len_penalty,
            no_repeat,
            max_len
        ],
        outputs=[
            out_img,
            out_ce,
            out_sc,
            out_blip,
            timing
        ]
    )

if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_api=False,
        max_threads=8
    )
