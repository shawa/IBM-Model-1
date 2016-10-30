import json

with open('data/sentences.json', 'r') as f:
    SENTENCE_PAIRS = json.load(f)


def source_words(lang):
    for pair in SENTENCE_PAIRS:
        for word in pair[lang].split():
            yield word

words = {lang: set(source_words(lang)) for lang in
         ('en', 'fr')}


translation_probabilities = {
    word_en: {
        word_fr: 1/len(words['en'])
        for word_fr in words['fr']}
    for word_en in words['en']
}

total_s = {word_en: 0 for word_en in words['en']}

converged = False
# do until convergence
for i in range(10):
    print(i)
    # set count(e|f) to 0 for all e,f
    counts = {word_en: {word_fr: 0 for word_fr in words['fr']}
              for word_en in words['en']}
    # set total(f) to 0 for all f
    totals = {word_fr: 0 for word_fr in words['fr']}

    # for all sentence pairs (e_s,f_s)
    for (es, fs) in [(pair['en'].split(), pair['fr'].split())
                     for pair in SENTENCE_PAIRS]:
        # for all words e in e_s
        for e in es:
            # total_s(e) = 0
            total_s[e] = 0

            # for all words f in f_s
            for f in fs:
                # total_s(e) += t(e|f)
                total_s[e] += translation_probabilities[e][f]

        # for all words e in e_s
        for e in es:
            # for all words f in f_s
            for f in fs:
                # count(e|f) += t(e|f) / total_s(e)
                counts[e][f] += (translation_probabilities[e][f] /
                                 total_s[e])

                # total(f)   += t(e|f) / total_s(e)
                totals[f] += translation_probabilities[e][f] / total_s[e]

    # for all f
    for f in words['fr']:
        # for all e
        for e in words['en']:
            translation_probabilities[e][f] = counts[e][f] / totals[f]
        # t(e|f) = count(e|f) / total(f)
