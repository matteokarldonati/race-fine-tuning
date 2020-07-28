import random
from difflib import SequenceMatcher

import neuralcoref
import nltk
import spacy
import torch
from nltk.corpus import names

nlp = spacy.load("en_core_web_sm")
neuralcoref.add_to_pipe(nlp)

nltk.download('names')

MALE_NAMES = names.words('male.txt')
FEMALE_NAMES = names.words('female.txt')
#AFRICAN_NAMES = torch.load('./data/african_names')
#CHINESE_NAMES = torch.load('./data/chinese_names')

NAMES = MALE_NAMES + FEMALE_NAMES


def group_names(names):
    result = []
    for sentence in names:
        if len(result) == 0:
            result.append([sentence])
        else:
            for i in range(0, len(result)):
                score = SequenceMatcher(None, sentence, result[i][0]).ratio()
                if score < 0.5:
                    if i == (len(result) - 1):
                        result.append([sentence])
                else:
                    if score != 1:
                        result[i].append(sentence)
    return result


def get_names(text):
    doc = nlp(text)
    names = []
    for x in doc.ents:
        if x.label_ == 'PERSON':
            names.append(x.text)

    names = group_names(list(set(names)))
    return names


def get_names_(text):
    doc = nlp(text)
    names = []
    for x in doc.ents:
        if x.label_ == 'PERSON':
            names.append(x.text)

    names = list(set(names))
    return names


def get_names_groups(text):
    doc = nlp(text)
    clusters = doc._.coref_clusters
    groups = []
    if clusters:
        for cluster in clusters:
            name_group = get_names_(' '.join(list(set([str(i) for i in cluster.mentions]))))
            if name_group:
                groups.append(name_group)

    return groups


def get_names_groups_(text):
    doc = nlp(text)
    clusters = doc._.coref_clusters
    groups = []
    if clusters:
        for cluster in clusters:
            name_group = get_names_('. '.join(list(set([str(i) for i in cluster.mentions]))))

            flag = True
            for name in name_group:
                for group in groups:
                    if name in group:
                        flag = False
                    elif name in ' '.join(group):
                        group.append(name)
                        flag = False

            if name_group and flag:
                groups.append(name_group)

    return groups


def get_adv_names(names_num, name_gender_or_race):
    adv_names = []
    for _ in range(names_num):
        if name_gender_or_race == 'males':
            adv_names.append(random.choice(MALE_NAMES))
        elif name_gender_or_race == 'female':
            adv_names.append(random.choice(FEMALE_NAMES))
        elif name_gender_or_race == 'african':
            adv_names.append(random.choice(AFRICAN_NAMES))
        elif name_gender_or_race == 'chinese':
            adv_names.append(random.choice(CHINESE_NAMES))
        else:
            adv_names.append(random.choice(NAMES))
    return adv_names


def replace_names(text, names, adv_names):
    if not names:
        return text

    for i, name_group in enumerate(names):
        for name in name_group:
            adv_text = text.replace(name, adv_names[i])
            text = adv_text

    return adv_text
