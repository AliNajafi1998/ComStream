import os
from pathlib import Path

pred_dir = os.path.join(Path(os.getcwd()).parent, 'Algo/output/X2020-03-29T23_59_59Z--19963/clusters')
texts = []
for label_dir in os.listdir(pred_dir):
    f = open(pred_dir + '/' + label_dir, "r")  # , encoding='utf8')
    lines = f.readlines()
    text = ''
    count = 0
    for line in lines:
        if line.strip() == '':
            continue
        # print(f':{line.strip()}:')
        text += line.strip() + ' '
        count += 1
    texts.append((count, text.strip()))
texts.sort(reverse=True)
texts = texts[:min(10, len(texts))]
# pip install pytextrank
# python -m spacy download en_core_web_sm
# pip3 install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz
import spacy
import pytextrank

for count, text in texts:

    # load a spaCy model, depending on language, scale, etc.
    nlp = spacy.load("en_core_web_sm")

    # add PyTextRank to the spaCy pipeline
    tr = pytextrank.TextRank()
    nlp.add_pipe(tr.PipelineComponent, name="textrank", last=True)

    doc = nlp(text)

    # examine the top-ranked phrases in the document
    for p in doc._.phrases:
        print("{:.4f} {:5d}  {}".format(p.rank, p.count, p.text))
        # print(p.chunks)
    break

# to see the topic keywords count with eyes
# word_dic = {}
# for line in outputs:
#     for word in line.split():
#         word_dic[word] = word_dic.get(word, 0) + 1
#
# print(sorted([(j, i) for i, j in word_dic.items()], reverse=True))
#
# topic1 = 'state, numbers, united, states, leader'
# topic2 = 'infected, die, selfish, idiots'
# for keyword in topic1.split(', '):
#     print(keyword, word_dic[keyword])
# print()
# for keyword in topic2.split(', '):
#     print(keyword, word_dic[keyword])
