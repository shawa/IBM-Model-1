import clize
import json


def dictify(lang_a, sentences_a, lang_b, sentences_b):
    for (sentence_a, sentence_b) in zip(sentences_a, sentences_b):
        yield {lang_a: sentence_a, lang_b: sentence_b}


def main(lang_a, infile_a, lang_b, infile_b):
    '''
    infile_a: language a of the europarl corpora
    infile_a: language b of the europarl corpora
    '''
    print('[')
    with open(infile_a, 'r') as a, open(infile_b, 'r') as b:
        while True:
            try:
                pair = next(dictify(lang_a, a, lang_b, b))
                print(json.dumps(pair))
                print(',')
            except StopIteration:
                print(']')
                break

if __name__ == '__main__':
    clize.run(main)
