import os
from argparse import Namespace

import copy
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd

from text import prepocessing, preprocess_text
from Dataset import SurnameDataset, InferenceDataset
from model import SurnameModel
from trainer import Trainer
from inference import Inference
from Voc import SurnameVectorizer

###### helper functions ######
# Set Numpy and PyTorch seeds
def set_seeds(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)
     
# Creating directories
def create_dirs(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

# Arguments
args = Namespace(
    seed=1234,
    cuda=True,
    shuffle=True,
    data_file="exp5/surnames.csv",
    vectorizer_file="vectorizer.json",
    model_state_file="model.pth",
    save_dir="exp5/names",
    train_size=0.7,
    val_size=0.15,
    test_size=0.15,
    num_epochs=30,
    early_stopping_criteria=5,
    learning_rate=1e-3,
    batch_size=64,
    num_filters=100,
    dropout_p=0.1,
)

# Set seeds
set_seeds(seed=args.seed, cuda=args.cuda)

# Create save dir
create_dirs(args.save_dir)

# Expand filepaths
args.vectorizer_file = os.path.join(args.save_dir, args.vectorizer_file)
args.model_state_file = os.path.join(args.save_dir, args.model_state_file)

# Check CUDA
if not torch.cuda.is_available():
    args.cuda = False
args.device = torch.device("cuda" if args.cuda else "cpu")
print("Using CUDA: {}".format(args.cuda))
# import time
# print(torch.cuda.is_available())
# time.sleep(5)

split_df = prepocessing(args)
dataset = SurnameDataset.load_dataset_and_make_vectorizer(split_df)
dataset.save_vectorizer(args.vectorizer_file)




# args_test = Namespace(
#     shuffle=True,
#     data_file="exp5/surnamestest.csv",
#     train_size=0.7,
#     val_size=0.15,
#     test_size=0.15,
# )

# split_df_test = prepocessing()


vectorizer = dataset.vectorizer
model = SurnameModel(num_input_channels=len(vectorizer.surname_vocab),
                     num_output_channels=args.num_filters,
                     num_classes=len(vectorizer.nationality_vocab),
                     dropout_p=args.dropout_p)
print (model.named_modules)


# Train
trainer = Trainer(dataset=dataset, model=model, 
                  model_state_file=args.model_state_file, 
                  save_dir=args.save_dir, device=args.device,
                  shuffle=args.shuffle, num_epochs=args.num_epochs, 
                  batch_size=args.batch_size, learning_rate=args.learning_rate, 
                  early_stopping_criteria=args.early_stopping_criteria)
trainer.run_train_loop()
trainer.plot_performance()
trainer.run_test_loop()
print("Test loss: {0:.2f}".format(trainer.train_state['test_loss']))
print("Test Accuracy: {0:.1f}%".format(trainer.train_state['test_acc']))
trainer.save_train_state()



with open(args.vectorizer_file) as fp:
    vectorizer = SurnameVectorizer.from_serializable(json.load(fp))
# Load the model
model = SurnameModel(num_input_channels=len(vectorizer.surname_vocab),
                     num_output_channels=args.num_filters,
                     num_classes=len(vectorizer.nationality_vocab),
                     dropout_p=args.dropout_p)
model.load_state_dict(torch.load(args.model_state_file))
print (model.named_modules)

# Initialize
inference = Inference(model=model, vectorizer=vectorizer, device=args.device)
surname = input("Enter a surname to classify: ")
infer_df = pd.DataFrame([surname], columns=['surname'])
infer_df.surname = infer_df.surname.apply(preprocess_text)
infer_dataset = InferenceDataset(infer_df, vectorizer)
results = inference.predict_nationality(dataset=infer_dataset)
print(results)