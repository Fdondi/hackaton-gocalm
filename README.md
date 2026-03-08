# How to run

* (Suggested): create virtual python environment.
* `pip install -r requirements.txt`
* add an .env file with OPENAI_API_KEY for generation (or otherwise set the environment variable)
* generate data with ` python generate_challenging_span_data.py --train-size 1500 --valid-size 300 --model gpt-5-nano --max_dollars 3` (adjust the parameters)
* (suggested) inspect data quality with `python export_declared_spans.py`, which puts it in human/LLM readable format (.review.md). 
* run the `modernbert_span_classifier_notebook.ipynb` to train and evaluate the model. It should take ~minutes per epoch on any GPU with 8+GB. Cpu fallback available.

## Local generation with LM Studio

Use this when you want local generation instead of OpenAI-hosted models.

1. Start LM Studio.
2. Load a model in LM Studio.
3. Start the OpenAI-compatible local server in LM Studio (`http://127.0.0.1:1234/v1` by default).
4. Copy the loaded model id from LM Studio (this is the value for `--model` when using `--local-base-url`).

Run:

```bash
python generate_challenging_span_data.py \
  --local \
  --model "<lmstudio-model-id>" \
  --local-base-url "http://127.0.0.1:1234/v1" \
  --train-size 1500 \
  --valid-size 300
```

Direct in-process GGUF loading (without LM Studio API):

```bash
python generate_challenging_span_data.py \
  --local \
  --model "E:\path\to\your-model.gguf" \
  --train-size 1500 \
  --valid-size 300
```

Notes:

- `--model` is used for both hosted and local runs.
- `--local` forces local mode.
- `--local-base-url` also enables local mode automatically.
- With `--local-base-url`, `--model` must be the model id served by LM Studio (not a `.gguf` file path).
- `--local-api-key` is optional; default is `lm-studio`.
- If `--local` is set and `--local-base-url` is omitted, the script tries `http://127.0.0.1:1234/v1` automatically and uses it when `--model` is found in `/v1/models`.
- If direct in-process GGUF loading fails on your machine (for example with a native `llama-cpp` crash), use LM Studio API mode as above.
>>>>>>> 729b852 (support local generation)
