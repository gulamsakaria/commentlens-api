---
title: CommentLens API
emoji: 🔎
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# 🔎 CommentLens

**An AI system that reads any Bangla/Banglish comment section and turns it into a civic-intelligence report.**

Drop in a link or a batch of comments from a Facebook post, YouTube video, or news article, and CommentLens tells you what the crowd is actually saying — not just sentiment, but structure:

- 📊 **Opinion-share breakdown** — how the comment section splits across opinion, factual claim, spam, toxicity, and general chatter
- 🩺 **Comment Health Score** — one score summarizing how healthy vs. toxic a conversation is
- 🧩 **Claim clustering** — groups repeated/similar factual claims together instead of showing 500 duplicates
- ✅ **News-backed claim verification** — cross-checks clustered claims against real news sources to flag likely misinformation

Built end-to-end by me for the **International AI Builders Congress 2026** — including collecting and labeling the training data, fine-tuning the classification model, and shipping it as a production inference service.

## AI/ML powering this

At the core is **`gulamsakaria/commentlens-banglishbert`** — a BanglishBERT transformer I fine-tuned on self-collected, hand-labeled Bangla/Banglish comments to classify each one into `claim`, `opinion`, `spam-scam`, `toxic`, or `general`. The opinion-share breakdown, health score, and downstream claim clustering/verification pipeline are all built on top of this model's output.

---

## This repo: the inference API

This repo is the **AI inference layer** of CommentLens — a FastAPI service that serves the fine-tuned BanglishBERT classifier as a plain HTTP JSON API, so it can be called directly from a PHP (or any other) backend. No Gradio UI, no client-side JS SDK required.

Runs the ONNX-quantized export of the model via `onnxruntime` for a lighter, faster CPU footprint (no PyTorch dependency at inference time).

### Classes

`claim`, `general`, `opinion`, `spam-scam`, `toxic`

### Endpoints

- `GET /` — status + metadata
- `GET /health` — liveness check
- `POST /predict` — classify a comment

#### `POST /predict`

Request:
```json
{ "text": "আপনার মন্তব্য এখানে" }
```

Response:
```json
{
  "label": "opinion",
  "confidence": 0.9993,
  "scores": {
    "claim": 0.0002,
    "general": 0.0003,
    "opinion": 0.9993,
    "spam-scam": 0.0001,
    "toxic": 0.0001
  },
  "inference_ms": 12.4
}
```

### PHP example

```php
<?php
$ch = curl_init("https://commentlens-api.onrender.com/predict");
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
curl_setopt($ch, CURLOPT_HTTPHEADER, ["Content-Type: application/json"]);
curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode(["text" => $comment]));
$response = curl_exec($ch);
curl_close($ch);
$result = json_decode($response, true);
// $result['label'], $result['confidence'], $result['scores']
```

---
*Built by Gulam Sakaria.*
