import re
from stopwords import stop_words


# Expects String of words ex: "aaaa bbb is a way"
def remove_stop_words(text):
    final_words = [word for word in text if word not in stop_words]
    final_text = " ".join(final_words)
    return final_text


def tokenizer_part1(text):
    word_list = []
    for word in text.split():
        word = word.lower()
        word = re.sub(r"[^\w\s]", "", word)
        word_list.append(word)
    cleaned_list = remove_stop_words(word_list)
    return cleaned_list


# Expects a Single word
def handle_plurals(word):
    if word.endswith("sses"):
        word = word.removesuffix("sses")
        word = word + "ss"
    if word.endswith("ies"):
        word = word.removesuffix("ies")
        word = word + "i"
    if not word.endswith("ss") and word.endswith("s"):
        word = word.removesuffix("s")

    return word


def handle_action_suffix(word):
    if word.endswith("ing"):
        if len(word) >= 6:
            new_word = word.removesuffix("ing")
            if not set(new_word).isdisjoint({"a", "e", "i", "o", "u"}):
                word = new_word
    elif word.endswith("ed"):
        if len(word) >= 5:
            new_word = word.removesuffix("ed")
            if not set(new_word).isdisjoint({"a", "e", "i", "o", "u"}):
                word = new_word
    return word


def handle_double_consonants(word):
    if len(word) < 2:
        return word
    last_two = word[-2:].lower()
    is_double_consonant = (last_two[0] == last_two[1]) and (last_two[0] not in "aeiou")
    if is_double_consonant and last_two not in ("ll", "ss", "ff"):
        word = word[0 : (len(word) - 1)]
    return word


def stem_that_bad_boy(word):
    if len(word) <= 3:
        return word
    word = handle_plurals(word)  # Ending with sses/ies/ss/s
    word = handle_action_suffix(word)  # Ending with ing/ed
    word = handle_double_consonants(
        word
    )  # twin consonants at the end, excluding ll, ss, ff

    return word


# Expects a String
def stemming_algo(text):
    after_stem = []
    for word in text.split():
        word = stem_that_bad_boy(word)
        after_stem.append(word)
    return after_stem


# Expects a String a line
def tokenizer(text):
    after_token = tokenizer_part1(text)
    after_stem = stemming_algo(after_token)
    return after_stem

