import pandas as pd
import numpy as np
import collections
import re

# Preprocessing
def preprocess_text(text):
    text = ' '.join(word.lower() for word in text.split(" "))
    text = re.sub(r"([.,!?])", r" \1 ", text)
    text = re.sub(r"[^a-zA-Z.,!?]+", r" ", text)
    return text

def prepocessing(args):

    df = pd.read_csv(args.data_file, header=0)
    # Split by nationality
    by_nationality = collections.defaultdict(list)
    for _, row in df.iterrows():
        by_nationality[row.nationality].append(row.to_dict())
    for nationality in by_nationality:
        print ("{0}: {1}".format(nationality, len(by_nationality[nationality])))




    # Create split data
    final_list = []
    for _, item_list in sorted(by_nationality.items()):
        if args.shuffle:
            np.random.shuffle(item_list)
        n = len(item_list)
        n_train = int(args.train_size*n)
        n_val = int(args.val_size*n)
        n_test = int(args.test_size*n)

    # Give data point a split attribute
        for item in item_list[:n_train]:
            item['split'] = 'train'
        for item in item_list[n_train:n_train+n_val]:
            item['split'] = 'val'
        for item in item_list[n_train+n_val:]:
            item['split'] = 'test'  

        # Add to final list
        final_list.extend(item_list)



    # df with split datasets
    split_df = pd.DataFrame(final_list)
    split_df["split"].value_counts()



        
    split_df.surname = split_df.surname.apply(preprocess_text)
    print(split_df.head())

    return split_df
