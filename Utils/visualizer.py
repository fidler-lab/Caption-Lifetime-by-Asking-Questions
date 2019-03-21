import os
from bs4 import BeautifulSoup
from dominate.tags import *


def dom2soup(tag):
    return BeautifulSoup(tag.render(), 'html.parser')


class Visualizer(object):
    def __init__(self, html, css):

        self.html = html
        self.css = css

        if not os.path.isfile(html):
            open(html, 'a').close()
            self.soup = self.initialize()
            self.update_html()
        else:
            with open(self.html, "r") as f:
                self.soup = BeautifulSoup(f.read(), 'html.parser')

    def initialize(self):
        # load the file
        with open(self.html, "r") as f:
            soup = BeautifulSoup(f.read(), 'html.parser')

        # set up html file
        soup.append(dom2soup(html(head(), body())))
        soup.head.append(dom2soup(link(rel='stylesheet', href="https://fonts.googleapis.com/css?family=Open+Sans")))
        soup.head.append(dom2soup(link(rel='stylesheet', href=self.css)))
        soup.body.append(dom2soup(table()))

        return soup

    def update_html(self):
        with open(self.html, "w") as outf:
            outf.write(str(self.soup))


class CaptionVisualizer(Visualizer):
    def __init__(self, html, css):
        super(CaptionVisualizer, self).__init__(html, css)

    def add_entry(self, sample):

        text = []
        text.append(p(b("Epoch {}".format(sample['epoch']))))
        text.append(p(b("Greedy sample: "), sample['greedy_sample']))
        text.append(p(b("POS predictions: "), sample['pos_pred']))
        beamsearch = [p(b("Beam search {}: ".format(i)), s) for i, s in enumerate(sample['beamsearch'])],
        references = [p(b("Ref {}: ".format(i)), s) for i, s in enumerate(sample['refs'])]
        text.extend(beamsearch)
        text.extend(references)

        entry = tr(
            td(
                div(
                    img(src=sample['img'], style="max-height: 300px; max-width: 400px;"),
                    cls='img'
                )
            ),
            td(
                div(cls='textb', *text)
            )
        )

        self.soup.body.table.append(dom2soup(entry))


class VQAVisualizer(Visualizer):
    def __init__(self, html, css):
        super(VQAVisualizer, self).__init__(html, css)

    def add_entry(self, sample):
        text = []
        text.append(p(b("Epoch {}".format(sample['epoch']))))
        text.append(p(b("Question: "), sample['question']))
        text.append(p(b("GT answer: "), sample['gt_ans']))
        text.append(p(b("Predicted answers: "), ', '.join(["{} ({:.2f})".format(ans, prob) for prob, ans in sample['predictions']])))
        references = [p(b("Ref {}: ".format(i)), s) for i, s in enumerate(sample['refs'])]
        text.extend(references)

        entry = tr(
            td(
                div(
                    img(src=sample['img'], style="max-height: 300px; max-width: 400px;"),
                    cls='img'
                )
            ),
            td(
                div(cls='textb', *text)
            )
        )

        self.soup.body.table.append(dom2soup(entry))


class QGenVisualizer(Visualizer):
    def __init__(self, html, css):
        super(QGenVisualizer, self).__init__(html, css)

    def add_entry(self, sample):
        text = []
        text.append(p(b("Epoch {}".format(sample['epoch']))))
        text.append(p(b("Reference question: "), sample['gt_question']))
        text.append(p(b("answer: "), sample['answer']))
        text.append(p(b("Greedy sample: "), sample['greedy_question']))
        beamsearch = [p(b("Beam search {}: ".format(i)), s) for i, s in enumerate(sample['beamsearch'])],
        text.extend(beamsearch)

        entry = tr(
            td(
                div(
                    img(src=sample['img'], style="max-height: 300px; max-width: 400px;"),
                    cls='img'
                )
            ),
            td(
                div(cls='textb', *text)
            )
        )

        self.soup.body.table.append(dom2soup(entry))


class LLVisualizer(Visualizer):
    def __init__(self, html, css):
        super(LLVisualizer, self).__init__(html, css)

    def add_entry(self, sample):
        text = []
        text.append(p(b("Epoch {}".format(sample['epoch']))))
        text.append(p(b("Original caption: "), sample['caption']))
        text.append(p(b("Question asking probabilities: "), sample['qaskprobs']))
        text.append(p(b("Rollout caption: "), sample['rollout_caption']))
        text.append(p(b("Replace caption: "), sample['replace_caption']))

        text.append(p(b("Decision made: "), "Asked Q: {}, index {}, word {}".format(
            sample['flag'], sample['index'], sample['word'])))
        text.append(p(b("Predicted Part-of-speech: "), sample['pos']))
        text.append(p(b("Question: "), sample['question']))
        text.append(p(b("Top 3 answers: "), sample['answers']))
        text.append(p(b("Captioner's predicted words: "), sample['words']))

        # text.append(p(b("Greedy decision: "), "index {} word {}".format(sample['gindex'], sample['gword'])))
        # text.append(p(b("Predicted Part-of-speech: "), sample['gpos']))
        # text.append(p(b("Question: "), sample['gquestion']))
        # text.append(p(b("Top 3 answers: "), sample['ganswers']))
        # text.append(p(b("Captioner's predicted words: "), sample['gwords']))

        references = [p(b("Reference {}: ".format(i)), s) for i, s in enumerate(sample['refs'])]
        text.extend(references)

        entry = tr(
            td(
                div(
                    img(src=sample['img'], style="max-height: 300px; max-width: 400px;"),
                    cls='img'
                )
            ),
            td(
                div(cls='textb', *text)
            )
        )

        self.soup.body.table.append(dom2soup(entry))