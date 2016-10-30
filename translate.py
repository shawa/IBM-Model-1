import clize
import json


def tokenize(sentence):
    return sentence.split()


def translate(tokens, model):
    return [model[word] if word in model else word for word in tokens]


def main(model_file, sentence):
    with open(model_file, 'r') as f:
        model = json.load(f)

    tokens = tokenize(sentence)
    translated_tokens = translate(tokens, model)

    print(" ".join(translated_tokens))

if __name__ == '__main__':
    clize.run(main)
