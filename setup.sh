set -euo pipefail

GPU="${GPU:-1}"          # 1: CUDA 12.1, 0: CPU
VENV=".venv310"

echo "[1/9] apt update & base packages ..."
sudo apt update
sudo apt install -y --no-install-recommends \
  build-essential cmake ninja-build git wget curl unzip \
  python3.10 python3.10-venv python3.10-dev \
  libjpeg-dev zlib1g-dev libgl1

echo "[2/9] Create venv ${VENV} (py310) + upgrade pip ..."
if [ ! -d "${VENV}" ]; then
  python3.10 -m venv "${VENV}"
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python -m pip install --upgrade pip setuptools wheel

echo "[3/9] Install torch + torchvision ..."
if [[ "${GPU}" == "1" ]]; then
  python -m pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121
else
  python -m pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

python - <<'PY'
import torch
print("torch:", torch.__version__, "cuda_available:", torch.cuda.is_available(), "cuda:", torch.version.cuda)
PY

echo "[4/9] Install fvcore + iopath (pin đúng phiên bản D2==0.6 cần) ..."
python -m pip install --no-cache-dir "fvcore==0.1.5.post20221221" "iopath==0.1.9"

echo "[5/9] Install detectron2==0.6 (no build isolation để dùng đúng deps đã pin) ..."
python -m pip install --no-cache-dir --no-build-isolation \
  'git+https://github.com/facebookresearch/detectron2.git@v0.6'

echo "[6/9] Cài common CV/utils + Web UI stack (pin huggingface_hub để tránh lỗi Gradio) ..."
python -m pip install --no-cache-dir \
  "numpy>=1.23,<2.0" "opencv-python>=4.8,<5" "tqdm>=4.66,<5" "packaging>=23.0" \
  yacs cloudpickle pyyaml termcolor tabulate matplotlib pycocotools \
  gradio==4.44.0 fastapi==0.115.0 starlette==0.38.2 h11==0.14.0 uvicorn==0.23.2 \
  huggingface_hub==0.19.4

echo "[7/9] Viết Detectron2 compatibility layer + smoke test ..."
cat > butd_d2_compat.py << 'PY'
"""
butd_d2_compat.py — Compat layer cho detectron2 0.6 (Python 3.10)
- Cung cấp API build_predictor(...) & extract_roi_features(...).
- Shim FastRCNNOutputs chỉ để tránh vỡ import trong code cũ (không dùng thực tế).
"""

import numpy as np
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data.transforms import ResizeShortestEdge
from detectron2.modeling.postprocessing import detector_postprocess

# --- Shim tránh vỡ import ở code cũ ---
class FastRCNNOutputs:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("FastRCNNOutputs đã bị remove. Hãy dùng pipeline mới (roi_heads._forward_box).")

def build_predictor(yaml_path: str, weight_path: str, device: str = None) -> DefaultPredictor:
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(yaml_path)
    cfg.set_new_allowed(False)
    cfg.MODEL.WEIGHTS = weight_path
    cfg.MODEL.DEVICE = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg.INPUT.MIN_SIZE_TEST = 600
    cfg.INPUT.MAX_SIZE_TEST = 1000
    predictor = DefaultPredictor(cfg)
    predictor.aug = ResizeShortestEdge(600, 1000)
    return predictor

@torch.no_grad()
def extract_roi_features(predictor: DefaultPredictor, img_bgr: np.ndarray, topk: int = 36) -> np.ndarray:
    """
    Trích ROI features (K, 2048) theo API detectron2 0.6 (không dùng FastRCNNOutputs):
      backbone -> proposal_generator -> roi_heads._forward_box -> detector_postprocess
      -> _shared_roi_transform + box_head -> pooled 2048-d vectors
    """
    H, W = img_bgr.shape[:2]
    img_rgb = img_bgr[:, :, ::-1].copy()
    tfm = predictor.aug.get_transform(img_rgb)
    img_tfm = tfm.apply_image(img_rgb)
    chw = np.ascontiguousarray(img_tfm.transpose(2, 0, 1))
    tensor = torch.from_numpy(chw).to(predictor.model.device, non_blocking=True).float()
    inputs = [{"image": tensor, "height": H, "width": W}]

    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        images = predictor.model.preprocess_image(inputs)
        feats_all = predictor.model.backbone(images.tensor)
        proposals, _ = predictor.model.proposal_generator(images, feats_all, None)
        pred_instances, _ = predictor.model.roi_heads._forward_box(feats_all, proposals)
        inst_rsz = pred_instances[0]
        inst = detector_postprocess(inst_rsz, H, W)

        if hasattr(inst, "scores") and len(inst) > 0:
            keep = torch.argsort(inst.scores, descending=True)[:topk]
            inst = inst[keep]; inst_rsz = inst_rsz[keep]
        else:
            keep = torch.arange(min(topk, len(inst_rsz)), device=images.tensor.device)
            inst = inst[keep]; inst_rsz = inst_rsz[keep]

        in_feats = [feats_all[f] for f in predictor.model.roi_heads.in_features if f in feats_all]
        box_feats = predictor.model.roi_heads._shared_roi_transform(in_feats, [inst_rsz.pred_boxes])
        pooled = predictor.model.roi_heads.box_head(box_feats)  # (K, 2048)
        return pooled.detach().cpu().numpy().astype("float32")
PY

cat > d2_compat_smoke.py << 'PY'
import torch
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
print("Imports OK. Dùng butd_d2_compat.build_predictor(...) và extract_roi_features(...) trong app của bạn.")
PY

echo "[8/9] In phiên bản để xác nhận ..."
python - <<'PY'
import fvcore, iopath, detectron2, torch, pkgutil
print("torch:", torch.__version__)
print("fvcore:", fvcore.__version__, "(expect <0.1.6)")
print("iopath:", iopath.__version__, "(expect <0.1.10)")
print("detectron2:", getattr(detectron2, "__version__", "local"), "(expect 0.6)")
PY

echo "[9/9] Gợi ý env & run (in ra để bạn copy) ..."
cat <<'EOS'

# ======= GỢI Ý: Export env trước khi chạy app của bạn =======
# export BUTD_YAML="/path/to/faster_rcnn_R_101_C4_attr_caffemaxpool.yaml"
# export BUTD_WEIGHT="/path/to/faster_rcnn_from_caffe_attr.pkl"
# export BUTD_VOCAB="/path/to/vocab_coco.json"
# export BUTD_CE_CKPT="/path/to/ce_best.pt"         # optional
# export BUTD_SCST_CKPT="/path/to/scst_best.pt"     # optional

# Kích hoạt và test:
#   source .venv310/bin/activate
#   python d2_compat_smoke.py

# Trong code của bạn (detectron2 0.6):
#   from butd_d2_compat import build_predictor, extract_roi_features
#   predictor = build_predictor(BUTD_YAML, BUTD_WEIGHT)
#   feats = extract_roi_features(predictor, img_bgr, topk=36)   # (K, 2048)
#   # -> đưa feats vào UpDownDecoder.beam_search như cũ

EOS

echo
echo "============================================================"
echo " DONE: py310 + detectron2 0.6 + deps chuẩn + compat layer"
echo " Activate:   source ${VENV}/bin/activate"
echo " Test:       python d2_compat_smoke.py"
echo "============================================================"