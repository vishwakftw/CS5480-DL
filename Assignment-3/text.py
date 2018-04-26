import numpy as np

def get_char_to_encoding(char_list):
    """
    Function to obtain characters --> encodings

    Args:
        char_set    : List of characters
    Returns:
        char_encoding
    """
    char_encoding = []
    for c in char_list:
        char_encoding.append(ord(c))
    return char_encoding


def get_encoding_to_char(enc_list):
    """
    Function to obtain encodings --> characters

    Args:
        enc_set    : List of encodings
    Returns:
        char_list
    """
    char_list = []
    for e in enc_list:
        char_list.append(chr(e))
    return char_list
