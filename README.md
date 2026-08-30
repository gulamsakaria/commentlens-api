---
title: CommentLens API
emoji: 🔎
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# CommentLens API

A small FastAPI wrapper around the fine-tuned BanglishBERT classifier
[`gulamsakaria/commentlens-banglishbert`](https://huggingface.co/gulamsakaria/commentlens-banglishbert),
served as a plain HTTP JSON API so it can be called directly from a PHP
(or any other) backend — no Gradio UI, no client-side JS SDK required.

Runs the ONNX-quantized export of the model via `onnxruntime` for a
lighter, faster CPU footprint (no PyTorch dependency).

## Classes

`claim`, `general`, `opinion`, `spam-scam`, `toxic`

## Endpoints

- `GET /` — status + metadata
- `GET /health` — liveness check
- `POST /predict` — classify a comment

### `POST /predict`

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

## PHP example

```php
<?php
$ch = curl_init("https://gulamsakaria-commentlens-api.hf.space/predict");
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
curl_setopt($ch, CURLOPT_HTTPHEADER, ["Content-Type: application/json"]);
curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode(["text" => $comment]));
$response = curl_exec($ch);
curl_close($ch);
$result = json_decode($response, true);
// $result['label'], $result['confidence'], $result['scores']
```
