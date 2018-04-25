import numpy as np

def get_char_encoding(char_set):
    """
    Function to obtain per character encodings in a vector form

    Args:
        char_set    : Set of characters (array / set)
    Returns:
        dictionary with the representations
    """
    char_encoding = {}
    for char in char_set:
        x = np.zeros(256)
        x[ord(char)] = 1
        char_encoding[char] = x
    return char_encoding


def get_text_encoding(text_file):
    """
    Pass a file name to get the minimalistic character-one-hot-representation

    Args:
        text_file   : Name of text file
    Returns:
        Matrix with representations
    """
    f = open(text_file)
    all_text = f.read()
    char_set = set(all_text)
    char_encoding = get_char_encoding(char_set)
    char_matrix = None
    for c in all_text:
        if char_matrix is None:
            char_matrix = char_encoding[c].reshape(1, -1)
        else:
            char_matrix = np.vstack([char_matrix, char_encoding[c].reshape(1, -1)])
    return char_matrix
