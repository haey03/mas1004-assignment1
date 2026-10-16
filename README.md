# Animal Image Classifier

**MAS1004 Data & AI — Assignment 1, Fall 2026**

A three-class image classifier built with ResNet18. This project explores how transfer learning performs on a small, self-collected image dataset and investigates whether changing an image's background can affect a prediction.

## Demo

- **Try the classifier:** https://haey03.github.io/mas1004-assignment1/
- **Source code:** https://github.com/haey03/mas1004-assignment1

The demo supports webcam input, image upload, and drawing. Inference runs in the browser using ONNX Runtime Web; images are not uploaded to a server.

## Categories and dataset

The model predicts one of three classes:

| Class | Images after cleaning |
|---|---:|
| `cartoon_animal` | 97 |
| `plush_animal` | 97 |
| `real_animal` | 95 |
| **Total** | **289** |

I initially collected 300 images (100 per class) and removed 11 unsuitable images during manual review. The dataset was split into 232 training images and 57 test images for the recorded experiments.

## Model and experiments

The classifier uses **ResNet18** with three output classes. I compared ImageNet-pretrained weights, random initialization, freezing the pretrained backbone, and a longer training run.

| Experiment | Initialization | Trainable layers | Epochs | Learning rate | Train accuracy | Test accuracy |
|---|---|---|---:|---:|---:|---:|
| `run` | ImageNet | All layers | 10 | 0.0001 | 100.0% | 93.0% |
| `scratch` | Random | All layers | 10 | 0.0001 | 84.9% | 64.9% |
| `frozen` | ImageNet | Final layer only | 10 | 0.001 | 99.6% | 94.7% |
| `longer` | ImageNet | All layers | 20 | 0.0003 | 97.4% | 84.2% |

The frozen-backbone experiment achieved the highest test accuracy among these runs (94.7%), while training from random initialization performed worse (64.9%). The longer run did not improve test accuracy, illustrating that more epochs do not necessarily improve generalization. Results are from one train/test split and should not be treated as a guarantee of performance on new images.

## Background hypothesis

Before training, I expected the model to use object shape, texture, and illustration style. I also suspected that it might use background or photographic context as a shortcut—for example, outdoor scenes for real animals and studio/product backgrounds for plush toys.

The script `src/background_experiment.py` supports a controlled check using a PNG with transparency: it composites the same foreground onto white and blue backgrounds and compares predictions from the frozen model. This is an investigation method, not proof by itself; the paired images and output predictions must be inspected before drawing a conclusion.

## Project structure

- `src/data.py` — image preprocessing, folder loading, and train/test split
- `src/train.py` — model construction and training
- `src/evaluate.py` — prediction, accuracy, confusion matrix, and error analysis
- `src/run.py` — run training experiments and save metrics/plots
- `src/collect_new_images.py` — collect images for evaluation on a new source
- `src/background_experiment.py` — controlled background-change experiment
- `src/export_web.py` — export a trained model for the browser
- `docs/` — static web demo and exported ONNX model
- `long_report/` and `short_report/` — assignment report sources and recorded outputs

## Run locally

Python dependencies are managed with `uv`. From the repository root:

```bash
uv sync
uv run pytest
```

Run the default experiment:

```bash
uv run python src/run.py
```

Run the other recorded configurations:

```bash
uv run python src/run.py --scratch --tag scratch
uv run python src/run.py --freeze --lr 1e-3 --tag frozen
uv run python src/run.py --epochs 20 --lr 3e-4 --tag longer
```

The scripts expect the cleaned dataset under `data/clean/`, arranged in one folder per class. The image dataset and generated training outputs may not be included in the repository; place the dataset locally before rerunning the experiments.

To export a model for the web demo, run:

```bash
uv run python src/export_web.py --tag frozen
```

Check the script output and export instructions before replacing the model files in `docs/`.

## Limitations

- The dataset is small, so test accuracy can vary with the split and training settings.
- The downloaded-image test set may not represent photos from a different source.
- High accuracy alone does not reveal which visual features the model relies on. Background changes should be controlled and interpreted carefully.

## Course

Created for **MAS1004 Data & AI, Sogang University**.
