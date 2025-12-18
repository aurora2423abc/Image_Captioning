# Image Captioning – BUTD + UpDown (WSL)

> **Mục tiêu**: Chạy demo caption ảnh (BUTD + UpDown 2‑LSTM, hai mô hình CE & SCST) trên **WSL Ubuntu**, dùng sẵn các checkpoint bạn đã để trong thư mục `checkpoints/`. README này hướng dẫn **từ A→Z**: cài môi trường bằng `setup.sh`, đặt file nặng đúng chỗ, chạy Gradio UI, và (tùy chọn) train trên Kaggle bằng **SCAN features**.

---

## 1) Yêu cầu hệ thống (WSL)

- **Windows 10/11** + **WSL2**, khuyến nghị Ubuntu **20.04/22.04**.
- Python **3.10** (script `setup.sh` sẽ tự cài venv).
- (Tùy chọn) GPU CUDA cho inference/training nhanh hơn. Nếu không có CUDA vẫn chạy được CPU.

> Lưu ý: Toàn bộ hướng dẫn này **dành cho WSL**. Không cần Anaconda.

---

## 2) Cấu trúc thư mục (chuẩn để chạy)

Tại thư mục dự án (ví dụ: `~/Image_Captioning_BUTD/`) bạn nên có:

```
Image_Captioning_BUTD/
├─ app_coco.py
├─ setup.sh
├─ README.md  (file này)
├─ .venv310/  (tự tạo sau khi chạy setup.sh)
├─ checkpoints/
│  ├─ faster_rcnn_from_caffe_attr.pkl
│  ├─ faster_rcnn_R_101_C4_attr_caffemaxpool.yaml
│  ├─ objects_vocab.txt
│  ├─ attributes_vocab.txt
│  ├─ vocab_coco.json
│  ├─ xe_best.pt
│  └─ scst_best.pt
└─ (tùy chọn) iamge-captioning-butd.ipynb, d2_compat_smoke.py, ...
```

> **Bạn đã có sẵn** những file ở `checkpoints/` (theo ảnh Drive bạn gửi). Nếu còn thiếu **duy nhất** file nặng `faster_rcnn_from_caffe_attr.pkl`, tải ở link dưới rồi đặt vào `checkpoints/`.

- Link TẢI trực tiếp (BUTD Caffe weights):  
  **http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe_attr.pkl**  
  (Đặt đúng tên file trong `checkpoints/` như trên)

- Thư mục Drive bạn đã public chứa đủ file (nếu muốn dùng):  
  **https://drive.google.com/drive/folders/13q0RGBR-XyaHXQwd2LH7zw_7BmUC4MkR**

---

## 3) Cài môi trường bằng `setup.sh` (1 lệnh duy nhất)

Trong WSL, đứng ở thư mục dự án:

```bash
chmod +x setup.sh
./setup.sh .venv310
```

Script sẽ:
- Cập nhật apt, cài công cụ build cần thiết.
- Tạo **virtualenv** tại `.venv310/` (Python 3.10).
- Cài các gói đúng phiên bản (đã **pin** để tránh xung đột):
  - `torch==1.10.2+cu113`, `torchvision==0.11.3+cu113` (hoặc bản CPU nếu không có CUDA).
  - `detectron2==0.6` (build từ source khớp Torch).
  - `fvcore==0.1.5.post20221221`, `iopath==0.1.9`, `pycocotools`…
  - `pillow<10` để tránh lỗi `Image.LINEAR` (đã xử lý trong mã nhưng vẫn pin cho chắc).
  - `gradio==4.44.1` (ổn định) + patch nhỏ chống lỗi schema.
- Kiểm tra việc nạp Detectron2.

> Nếu cuối script báo OK, bạn đã sẵn sàng chạy demo.

**Kích hoạt môi trường (khi mở terminal mới):**

```bash
source .venv310/bin/activate
```

---

## 4) Chạy demo Gradio

Vẫn trong thư mục dự án (đã `source .venv310/bin/activate`):

```bash
python app_coco.py
```

- Mặc định server mở ở: **http://0.0.0.0:7860**  
  (trên Windows, mở trình duyệt vào `http://localhost:7860`)

### Thay đổi đường dẫn/tuỳ chọn bằng biến môi trường (không bắt buộc)

Bạn có thể override các đường dẫn nếu để file ở vị trí khác:

```bash
export BUTD_YAML=./checkpoints/faster_rcnn_R_101_C4_attr_caffemaxpool.yaml
export BUTD_WEIGHT=./checkpoints/faster_rcnn_from_caffe_attr.pkl
export BUTD_VOCAB=./checkpoints/vocab_coco.json
export BUTD_OBJ_VOCAB=./checkpoints/objects_vocab.txt
export BUTD_ATTR_VOCAB=./checkpoints/attributes_vocab.txt
export BUTD_CE_CKPT=./checkpoints/xe_best.pt
export BUTD_SCST_CKPT=./checkpoints/scst_best.pt
python app_coco.py
```

Một số tham số khác (mặc định hợp lý):
- `BUTD_NUM_OBJECTS` (mặc định `36`): số box/ảnh đưa vào decoder.
- `BUTD_MIN_TEST`, `BUTD_MAX_TEST`: resize khung ngắn/dài.
- `BUTD_RPN_TOPK`: top‑K proposals sau NMS ở RPN.

---

## 5) Lưu ý & xử lý lỗi thường gặp

### 5.1 Lỗi Pillow `Image.LINEAR` / `Image.ANTIALIAS`

- Script đã **pin** `pillow<10`. Nếu bạn lỡ nâng cấp, chạy:  
  ```bash
  pip install "pillow<10" --upgrade
  ```
- Trong `app_coco.py` cũng đã có shim để dùng `Resampling.BILINEAR/BICUBIC/LANCZOS` khi cần.

### 5.2 Detectron2 báo skip một số tham số khi nạp checkpoint

- Do khác biệt nhỏ giữa head trong YAML và model Caffe‑style; **không ảnh hưởng inference**. Đoạn code đã:

  - Ép `C4` backbone với `RPN.CONV_DIMS = [512]`.
  - Dùng **NMS class‑agnostic** để tránh “một vật nhiều khung”.

### 5.3 UI thay đổi tham số nhưng caption không đổi?

- Bật “Cache features by image” thì thay đổi tham số decode **không cần** trích xuất lại đặc trưng; caption sẽ thay đổi khi bạn bấm **Run** hoặc chỉnh `decode strategy/beam size/...` (trang đã đăng ký auto refresh cho các control).

---

## 6) (Tùy chọn) Train trên **Kaggle** bằng **SCAN features**

Nếu bạn chỉ muốn demo train/finetune nhanh mà **không xử lý ảnh thô**, dùng dataset:  
**https://www.kaggle.com/datasets/kuanghueilee/scan-features**

Cách làm tối giản:
1. Tạo **Kaggle Notebook** (GPU bật **on**).
2. Ở tab **Add data** → thêm dataset “`kuanghueilee/scan-features`”.
3. Upload notebook có sẵn trong repo: **`iamge-captioning-butd.ipynb`** (đã có các hàm chia dữ liệu, tạo vocab, loader từ features).  
   - Chỉ cần sửa **đường dẫn base** tới thư mục features của Kaggle nếu notebook yêu cầu (`/kaggle/input/scan-features/…`).  
   - Chạy toàn bộ cell → huấn luyện **CE** trước, sau đó **SCST** (đã có helper CIDEr/RL).
4. Checkpoint sinh ra có thể tải về và đặt vào thư mục `checkpoints/` để dùng với `app_coco.py`.

> Lưu ý: SCST cần `pycocoevalcap` (đã nằm trong `setup.sh` và notebook).

---

## 7) Lệnh nhanh (copy/paste)

```bash
# 0) vào thư mục dự án
cd ~/Image_Captioning_BUTD

# 1) cài môi trường
chmod +x setup.sh
./setup.sh .venv310

# 2) kích hoạt venv mỗi lần mở terminal mới
source .venv310/bin/activate

# 3) (đảm bảo file nặng đã có)
#    - checkpoints/faster_rcnn_from_caffe_attr.pkl
#    - các file vocab + checkpoint CE/SCST
#    nếu thiếu file nặng, tải:
#    http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe_attr.pkl

# 4) chạy demo
python app_coco.py
# -> mở http://localhost:7860
```

---

## 8) Góp ý

Nếu bạn muốn đóng gói lại (đổi cổng, bật `share=True`, …), sửa phần cuối `app_coco.py` trong `demo.queue().launch(...)`.

Chúc bạn chạy mượt! 🚀
