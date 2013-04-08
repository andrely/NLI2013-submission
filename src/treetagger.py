import logging
import re
import subprocess
from config import get_config

def simple_tokenize(fn, out_fn, sentence_markers=['<s>', '</s>']):
    with open(fn, 'r') as f:
        with open(out_fn, 'w') as out_f:
            for line in f.readlines():
                tokens = line.split()

                if sentence_markers:
                    out_f.write(sentence_markers[0] + "\n")

                for tok in tokens:
                    out_f.write(tok.strip() + "\n")

                if sentence_markers:
                    out_f.write(sentence_markers[-1] + "\n")

TREETAGGER_NO_ERROR_OUTPUT = 'reading parameters...tagging ...finished.'

def tree_tag(input, language='en'):
    conf = get_config()
    cmd = conf['treetagger'][language]

    p = subprocess.Popen([cmd],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        shell=True)

    p.stdin.write(input)
    p.stdin.close()

    output = p.stdout.read()

    err = p.stderr.read()

    if re.sub(r"\s+", "", err) != TREETAGGER_NO_ERROR_OUTPUT:
        logging.info("Treetagger stderr: %s" % err)

    ret = p.wait()

    if ret != 0:
        logging.warn('Treetagger returned error code %d' % ret)

    return output

def tree_tag_file(fn, out_fn, language='en'):

    with open(fn, 'r') as f:
        input = f.read()
        output = tree_tag(input)

        with open(out_fn, 'w') as f:
            f.write(output)
