# Project description: Nutrition Predictor

The overall goal of the project is to train a neural network (NN) model that can predict the nutritional value of a plate of food based on an image of the plate.

The overall goal of the project is to train a model that can predict the
nutritional value of a plate of food based on an image of the plate.

The model is trained on a subset of the
[Nutrition5k dataset](https://github.com/google-research-datasets/Nutrition5k?tab=readme-ov-file)
containing
[side angle images](https://www.kaggle.com/datasets/zygmuntyt/nutrition5k-dataset-side-angle-images/data).
The kaggle dataset contains 20 samples per plate of food: four cameras, out of
which were extracted five consecutive frames. We will use only one of
these angles per plate (configurable via Hydra).

After processing, our dataset will contain a single image of 5000 different
dishes and metadata for each dish corresponding to 0.6GB of (uncompressed) data.
The images are JPEGs, and the metadata a CSV containing the total calories, mass,
fat, carb and protein associated with each dish ID.

We use ResNet18 as backbone, and we adapt it for regression by replacing the
final classification layer with a small FF network that outputs the nutritional
values. The pretrained backbone serves as a feature extractor, with its weights
frozen during training while the regression head is learned. This architecture
follows approaches we have commonly seen in Kaggle notebooks for the Nutrition5k
dataset. While published work has employed larger models like InceptionV2 and
ResNet50/101 ([Thames et al., 2021](https://arxiv.org/abs/2103.03375)), ResNet18
is a reasonable choice for purposes not focused on prediction accuracy.
