# How to run

* (Suggested): create virtual python environment.
* `pip install -r requirements.txt`
* add an .env file with OPENAI_API_KEY for generation (or otherwise set the environment variable)
* generate data with ` python generate_challenging_span_data.py --train-size 1500 --valid-size 300 --model gpt-5-nano --max_dollars 3` (adjust the parameters)
* (suggested) inspect data quality with `python export_declared_spans.py`, which puts it in human/LLM readable format (.review.md). 
* run the `modernbert_span_classifier_notebook.ipynb` to train and evaluate the model. It should take ~minutes per epoch on any GPU with 8+GB. Cpu fallback available.
>>>>>>> d2ace78 (first version)
