# Kinyarwanda news topic classification

I built a system that reads a news article written in Kinyarwanda and predicts which of 14 topics it belongs to, such as politics, sport, health, or economy. This is my NLP capstone project for the Machine Learning Techniques course at African Leadership University. The task is text classification for an African language.

## Project links

* Live app: https://huggingface.co/spaces/DavidPontientKayumba/kinyarwanda-news-classifier (it runs on a free Hugging Face Space, so it can take a minute to wake up)
* Fine tuned model: https://huggingface.co/DavidPontientKayumba/afroxlmr-kinnews-topic-classifier
* GitHub repository: https://github.com/DavidPontient/kinyarwanda-news-topic-classification

## Why I chose this problem

Kinyarwanda is widely spoken in Rwanda, where I study, but there is much less labeled text for it than for English. News topic classification is a clear, measurable task, and it lets me work with real Kinyarwanda text and its real problems.

My longer term goal is to build a digital platform that makes land ownership and home construction in Rwanda more transparent and accessible. Much of the information people need is written in Kinyarwanda, so tools that can read and sort Kinyarwanda text are a useful building block. This project is a first step. It does not use land data, and I do not claim that it solves a land problem.

## Dataset

I used KINNEWS (Niyongabo et al., 2020), a collection of Kinyarwanda news articles from Rwandan news websites and newspapers, labeled with 14 topics. I loaded the cleaned version from the Hugging Face dataset `andreniyongabo/kinnews_kirnews`. The Hugging Face page lists the license as MIT. The original split has 17,014 training and 4,254 test articles.

### A data quality problem I found and fixed

While checking the data I found:

* 7,939 of the 17,014 training articles were exact duplicates.
* 678 articles appeared in both the training set and the test set.
* 109 training texts and 13 test texts appeared with two different labels.

If I had kept these, a model could score well just by remembering articles it had already seen. My best baseline on the original split scored 0.8905 test accuracy. After cleaning, my best baseline scored 0.7982. I trust the lower number, because it no longer rewards memorizing.

The duplicates were not spread evenly. After cleaning, economy kept only about 30 percent of its training articles, tourism about 34 percent, and politics about 43 percent, while education and relationship lost almost nothing.

My cleaning steps:

1. Lowercase the text, remove web links, remove punctuation except the apostrophe, and join the title and the content.
2. Remove texts that appear with more than one label.
3. Remove exact duplicate texts inside the training set and inside the test set.
4. Remove test articles whose text also appears in the training set (646 articles at this step).
5. Take 10 percent of the cleaned training data as a validation set, split so that every topic keeps its share (seed 42).

Final sizes: 8,069 training, 897 validation, and 2,027 test articles.

| Topic | Train (before the validation split) | Test |
|---|---|---|
| politics | 1705 | 409 |
| entertainment | 1430 | 355 |
| sport | 1420 | 311 |
| health | 1101 | 256 |
| economy | 1057 | 227 |
| relationship | 639 | 123 |
| religion | 570 | 131 |
| technology | 212 | 41 |
| history | 175 | 54 |
| culture | 173 | 42 |
| education | 152 | 34 |
| environment | 122 | 16 |
| fashion | 109 | 20 |
| tourism | 101 | 8 |

The data is unbalanced. Some test topics have very few articles (tourism has 8, environment 16), so their scores are noisy.

Articles are also long. With the AfroXLMR tokenizer the median article has 589 tokens. 93.2 percent of the training articles are longer than 256 tokens, and 60.3 percent are longer than 512.

## Method

### Baseline

TF-IDF features with a linear classifier. TF-IDF gives each piece of text a weight that is high when it is common in one article but rare across all articles. I tried five baseline setups on the validation set. Each one changes one thing from the one before it.

| Name | Features | Classifier | Validation accuracy | Validation macro F1 |
|---|---|---|---|---|
| B1 | word unigrams | Logistic Regression | 0.7703 | 0.6888 |
| B2 | word unigrams and bigrams | Logistic Regression | 0.7781 | 0.6856 |
| B3 | word unigrams and bigrams | Linear SVM | 0.7949 | 0.7065 |
| B4 | character n-grams (3 to 5) | Linear SVM | 0.7904 | 0.7189 |
| B5 | character n-grams (3 to 5) | Linear SVM with balanced class weights | 0.7915 | 0.7324 |

B5 was the best on validation, so it is the baseline I compare against.

### Fine tuned transformer

I fine tuned AfroXLMR base (Alabi et al., 2022). This is XLM-R base (Conneau et al., 2020) that was further trained on text from 17 African languages, including Kinyarwanda and Kirundi. Fine tuning works like this:

1. The tokenizer splits each article into word pieces and cuts it at the maximum length.
2. The pretrained Transformer encoder reads the tokens.
3. A small classification layer with 14 outputs sits on top. It starts with random weights.
4. All weights are trained on my labeled articles with cross entropy loss.
5. After every epoch I score the model on the validation set and keep the best epoch.

Settings for the main run (T1): maximum length 256 tokens, batch size 16, learning rate 2e-5, 3 epochs, AdamW with weight decay 0.01, a warmup of 10 percent of the steps followed by linear decay, gradient clipping at 1.0, mixed precision (float16) on a Colab T4 GPU, and seed 42. I wrote the training loop myself in PyTorch instead of using a ready made trainer.

## Experiments and results

Models are chosen on validation macro F1. The test set is only used to report. Macro F1 gives every topic the same weight, so small topics count as much as politics. I use it because the data is unbalanced and accuracy alone would hide poor results on small topics.

| Run | What it tests | Validation macro F1 | Test accuracy | Test macro F1 |
|---|---|---|---|---|
| Baseline B5 | TF-IDF character n-grams, linear SVM, balanced weights | 0.7324 | 0.7982 | 0.7189 |
| T1 | AfroXLMR base, 3 epochs, 256 tokens | 0.6890 | 0.7834 | 0.6666 |
| T2 | T1 with class weighted loss | 0.6620 | 0.7593 | 0.6657 |
| T3 | XLM-R base, without the African language adaptation | 0.4602 | 0.6877 | 0.4573 |
| T4 | AfroXLMR small | 0.6049 | 0.7716 | 0.6337 |
| T5 | T1 trained for 8 epochs (best epoch was 5) | 0.6847 | 0.7745 | 0.6762 |
| T6 | T1 with 512 tokens and batch size 8 | 0.6846 | 0.7884 | 0.6800 |

What I learned:

* The baseline still wins. The gap in accuracy is small, but the gap in macro F1 is larger, and it comes mostly from the small topics.
* African language adaptation matters a lot. T1 and T3 have the same size and settings, and T3 scores far lower. T3 scored zero on history, education, and environment.
* A class weighted loss did not help the transformer. It raised recall and lowered precision, and the macro F1 stayed about the same.
* Training longer did not help. In T5 the validation score peaked at epoch 5 and did not improve after that, while the training loss kept falling.
* Reading more of each article (T6) gave a small gain on the test set, but T6 also changed the batch size, and the gain is about the size of the run to run variation. So truncation is probably not the main reason the baseline wins.
* Runs vary. When I ran T1 twice with the same settings, test accuracy differed by about one point. I used one seed per experiment because of time, so small differences should not be read as proof.

### Scores per topic on the test set (F1)

| Topic | Test articles | Baseline | T1 | T6 |
|---|---|---|---|---|
| politics | 409 | 0.744 | 0.757 | 0.753 |
| entertainment | 355 | 0.880 | 0.882 | 0.886 |
| sport | 311 | 0.951 | 0.946 | 0.951 |
| health | 256 | 0.718 | 0.675 | 0.693 |
| economy | 227 | 0.724 | 0.733 | 0.715 |
| religion | 131 | 0.893 | 0.821 | 0.844 |
| relationship | 123 | 0.843 | 0.839 | 0.842 |
| history | 54 | 0.527 | 0.261 | 0.324 |
| culture | 42 | 0.632 | 0.500 | 0.424 |
| technology | 41 | 0.684 | 0.694 | 0.757 |
| education | 34 | 0.627 | 0.493 | 0.500 |
| fashion | 20 | 0.750 | 0.718 | 0.698 |
| environment | 16 | 0.424 | 0.348 | 0.467 |
| tourism | 8 | 0.667 | 0.667 | 0.667 |

### Error analysis

* On the same 2,027 test articles, the baseline and T1 are both correct on 1,464 articles and both wrong on 285. The baseline alone is correct on 154 and T1 alone on 124.
* The most common mistakes of T1 are health predicted as politics (38), politics predicted as health (35), economy predicted as politics (29), politics predicted as economy (26), and history predicted as politics (21). These topics overlap in real news.
* The most confident mistakes of T1 are often not clear model failures. Several articles that T1 and the baseline both labeled sport (about football matches and sports competitions) carry the labels entertainment, economy, or health in the dataset. This suggests the labels may follow the section of the news site more than the content, which fits the contradictory labels I found earlier. I list examples in `results/error_examples.csv`.

## The web app

The app is a Gradio app on a Hugging Face Space. A user pastes a Kinyarwanda news text and sees the prediction of both models side by side: the baseline (loaded from `best_baseline.joblib`) and the fine tuned AfroXLMR (loaded from the Hugging Face Hub). The text is cleaned the same way as for training. The app says whether the models agree and warns when a text is longer than the 256 tokens that the fine tuned model reads.

The Space uses Hugging Face's free ZeroGPU setting, which is the free option available to new accounts. Both models run on the CPU, and a small placeholder function satisfies ZeroGPU's requirement for a GPU function. It is never called.

## Repository contents

* `notebooks/01_data_and_baseline.ipynb`: loads the data, finds and removes the quality problems, makes the splits, and runs the baseline experiments.
* `notebooks/02_finetune_transformers.ipynb`: token length analysis, the fine tuning runs T1 to T6, the comparison with the baseline, the error analysis, and the upload of the chosen model to Hugging Face.
* `app/`: `app.py`, `requirements.txt`, and `best_baseline.joblib` for the Space.
* `results/`: small result files (comparison tables, per topic scores, error examples, and the result file of each run).
* `figures/`: charts and confusion matrices.

I did not upload the model weights (they are on Hugging Face) or the data splits (the notebook downloads KINNEWS itself).

## How to reproduce

1. Open `01_data_and_baseline.ipynb` in Google Colab and run all cells. It saves its files in `MyDrive/kinyarwanda_news_project`.
2. Open `02_finetune_transformers.ipynb` in Colab with a T4 GPU runtime and run all cells. The runs take about 75 minutes in total. The first section removes the results of any older run in `transformer_runs`, so nothing is duplicated.
3. I ran notebook 1 twice, and both runs gave identical results, for example a test accuracy of 0.7982 and a macro F1 of 0.7189 for the best baseline.
4. The upload to Hugging Face at the end is switched off. To use it, add your own Hugging Face token as a Colab secret named `HF_TOKEN` and set `UPLOAD_TO_HUB = True`.

## What I used and what I did myself

| Existing resources | My own work |
|---|---|
| KINNEWS dataset (Niyongabo et al., 2020) | Checking the data and finding the duplicates, overlap, and contradictory labels |
| Pretrained models AfroXLMR base and small, and XLM-R base | The cleaning function and the deduplicated, stratified splits |
| Libraries: Hugging Face transformers and huggingface_hub, PyTorch, scikit-learn, pandas, matplotlib, joblib, Gradio | The five baseline experiments and their design |
| Google Colab and Google Drive for training and storage | The fine tuning loop: warmup, linear decay, mixed precision, gradient clipping, class weighted loss |
| Hugging Face Hub and Spaces for hosting | The design of runs T1 to T6, the token length analysis, and the per topic comparison |
| | The error analysis and the choice of the final model |
| | The web app that shows both models side by side, and its deployment |

I used AI assistance (chatgpt) for some of the research, while building the project. I am responsible for what I submit, and I can explain the data, code, experiments, and results.

## Limitations

* One random seed per experiment, so differences of about one point between runs are not reliable.
* Articles are cut at 256 tokens for the main model, so it reads less than half of a typical article.
* Several test topics have very few articles, so their scores are noisy.
* The labels are noisy. Some articles carry labels that look wrong, and some appear with two labels.
* The model can only choose one of the 14 topics. It has no "other" option, so an out of domain text still gets a topic.
* The data comes from news websites, so the model may not work as well on other kinds of Kinyarwanda text.

## References

Alabi, J. O., Adelani, D. I., Mosbach, M., & Klakow, D. (2022). Adapting pre-trained language models to African languages via multilingual adaptive fine-tuning. In *Proceedings of the 29th International Conference on Computational Linguistics* (pp. 4336 to 4349). International Committee on Computational Linguistics. https://aclanthology.org/2022.coling-1.382

Conneau, A., Khandelwal, K., Goyal, N., Chaudhary, V., Wenzek, G., Guzmán, F., Grave, E., Ott, M., Zettlemoyer, L., & Stoyanov, V. (2020). Unsupervised cross-lingual representation learning at scale. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics* (pp. 8440 to 8451). Association for Computational Linguistics. https://doi.org/10.18653/v1/2020.acl-main.747

Niyongabo, R. A., Qu, H., Kreutzer, J., & Huang, L. (2020). KINNEWS and KIRNEWS: Benchmarking cross-lingual text classification for Kinyarwanda and Kirundi. *Proceedings of the 28th International Conference on Computational Linguistics*. https://aclanthology.org/2020.coling-main.480 (preprint: https://arxiv.org/abs/2010.12174)

Please check the license terms of the dataset and of each pretrained model before you reuse them.
