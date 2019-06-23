import re
from stemming.porter2 import stem
from os import listdir
from os.path import isfile, join, abspath
from nltk.util import ngrams

# the data is stored in different folders, with the same directory except the last part.
FOLDER_PREFIX = 'C:/Corpus/norq_ texts_with_normalised_quotes/norq_ texts with normalised quotes/norq/'

def child_files(folder_name: str) -> list:
    full_folder_name = f'{FOLDER_PREFIX}/{folder_name}'
    return [file for file in listdir(full_folder_name) if isfile(join(full_folder_name, file))]


# this is the function for ngram method which merge words together according to the length requirement
def word_grams(words, min=1, max=3):
    """
    Merges ajacent words together, in this case ["in", "the", "house"】 becomes
    ["in", "the", "house", "in the"," the house"]
    :param words: entry words
    :param min: minimal length of word
    :param max: maximum length of word add 1
    :return: list of words ("tokens")
    """
    result = []
    for n in range(min, max):
        for ngram in ngrams(words, n):
            result.append(' '.join(str(i) for i in ngram))
    return result

#function to process the words including symbol removal, tokenization, bigram as well as stemming
def process_files(files):
    result = []
    for file in files:
        with open(file, errors='ignore') as opened_file:
            lines = [line.rstrip('\n') for line in opened_file]
            for cur_line in lines:
                cur_line = re.split(
                    r"[' ' |\( | \) | \/ |,|;|: |\-|\"|\\ |\? |\• |\t |\n |\» |\« |\/|\(|\)|,"
                    r"|;|\.\s | \� | \& | \! |\< |\> |\€ |\% |\~ |\= |\| |\` |\[ |\] |\… | ]+",
                    cur_line
                )
                cur_line = [token.lower() for token in cur_line]
                cur_line = word_grams(cur_line)
                cur_line = [stem(token) for token in cur_line]
                for a in range(0, len(cur_line)):
                    result.append(cur_line[a])
    return result


final_list = process_files(child_files('ad2002'))
final_list += process_files(child_files('nrc2002'))
final_list += process_files(child_files('t2002'))
final_list += process_files(child_files('tr2002'))
final_list += process_files(child_files('vk2002'))

with open('general_corpus_ngram', 'w') as filehandle:
    for listitem in final_list:
        filehandle.write('%s\n' % listitem)
