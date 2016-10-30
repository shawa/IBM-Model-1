import json
import pprint
import clize
from copy import deepcopy

from table_distance import distance

def get_corpus(filename):
    with open(filename, 'r') as f:
        sentence_pairs = json.load(f)
    return sentence_pairs


def get_words(corpus):
    def source_words(lang):
        for pair in corpus:
            for word in pair[lang].split():
                yield word
    return {lang: set(source_words(lang)) for lang in ('en', 'fr')}


def init_translation_probabilities(corpus):
    words = get_words(corpus)
    return {
        word_en: {word_fr: 1/len(words['en'])
                  for word_fr in words['fr']}
        for word_en in words['en']}


def train_iteration(corpus, words, total_s, prev_translation_probabilities):
    translation_probabilities = deepcopy(prev_translation_probabilities)

    counts = {word_en: {word_fr: 0 for word_fr in words['fr']}
              for word_en in words['en']}

    totals = {word_fr: 0 for word_fr in words['fr']}

    for (es, fs) in [(pair['en'].split(), pair['fr'].split())
                     for pair in corpus]:
        for e in es:
            total_s[e] = 0

            for f in fs:
                total_s[e] += translation_probabilities[e][f]

        for e in es:
            for f in fs:
                counts[e][f] += (translation_probabilities[e][f] /
                                 total_s[e])
                totals[f] += translation_probabilities[e][f] / total_s[e]

    for f in words['fr']:
        for e in words['en']:
            translation_probabilities[e][f] = counts[e][f] / totals[f]

    return translation_probabilities


def is_converged(probabilties_prev, probabilties_curr, epsilon):
    return distance(probabilties_prev, probabilties_curr) < epsilon


def train_model(corpus, epsilon):
    words = get_words(corpus)

    total_s = {word_en: 0 for word_en in words['en']}
    prev_translation_probabilities = init_translation_probabilities(corpus)

    converged = False
    iterations = 0
    while not converged:
        translation_probabilities = train_iteration(
                                        # this is a disgusting way
                                        # to indent code
                                        corpus, words, total_s,
                                        prev_translation_probabilities
                                    )

        converged = is_converged(prev_translation_probabilities,
                                 translation_probabilities, epsilon)
        prev_translation_probabilities = translation_probabilities
        iterations += 1
    return translation_probabilities, iterations


def main(infile, *, epsilon:'e'=0.1):
    '''
    IBM Model 1 SMT Training Example

    infile: json file containing english-french sentence pairs
            in the form [ {"en": <sentence>, "fr": <sentence>}, ... ]

    epsilon: acceptable euclidian distance between translation probability
             vectors across iterations
    '''
    corpus = get_corpus(infile)
    result, iterations = train_model(corpus, epsilon)
    pprint.pprint(result)

if __name__ == '__main__':
    clize.run(main)
