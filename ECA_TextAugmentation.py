import random
import nlpaug.augmenter.word as naw

model_dir = "Packages/"


def text_augment(input_text, number_of_outputs=1, aug_type='Synonym'):
    """
    Perform text augmentation on input text using a specified augmentation method.

    Parameters:
    - input_text (str): The input text to be augmented.
    - number_of_outputs (int): Number of augmented outputs to generate.
                              Default is 1.
    - aug_type (str): The type of text augmentation to apply.
                      Options: 'Synonym', 'word2vec', 'contextualWord'.
                      Default is 'Synonym'.

    Returns:
    - list: The list of augmented texts.
    """
    # model_type: word2vec, glove or fasttext
    if aug_type == 'word2vec':
        aug = naw.WordEmbsAug(
            model_type=aug_type,
            model_path=model_dir + 'GoogleNews-vectors-negative300.bin',
            action="substitute"
        )
    elif aug_type == 'contextualWord':
        aug = naw.ContextualWordEmbsAug(
            model_path='bert-base-uncased',
            action="substitute"
        )
    else:
        aug = naw.SynonymAug(
            aug_src='ppdb',
            model_path=model_dir + 'ppdb-2.0-tldr'
        )
    augmented_text = aug.augment(input_text, n=number_of_outputs+1)
    # print(f"Original:\n{input_text}\nAugmented Text:\n{augmented_text}")
    return augmented_text


def augmentation(dataset, number_of_outputs=1, aug_type='Synonym'):
    """
    Augment a dataset using a specified text augmentation method.

    Parameters:
    - dataset (list): The input dataset to be augmented.
    - number_of_outputs (int): Number of augmented outputs to generate for each input.
                              Default is 1.
    - aug_type (str): The type of text augmentation to apply.
                      Options: 'Synonym', 'word2vec', 'contextualWord'.
                      Default is 'Synonym'.

    Returns:
    - list: The augmented dataset.
    """
    augmented_data = []
    for row in dataset:
        augmented_data.extend(text_augment(row, number_of_outputs+1, aug_type))
    extended_dataset = dataset + augmented_data
    random.shuffle(extended_dataset)
    return extended_dataset
