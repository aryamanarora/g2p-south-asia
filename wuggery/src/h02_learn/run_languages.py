import os
import sys
import subprocess

sys.path.append('./src/')
from h01_data.unimorph import Unimorph
from util import argparser


def get_args():
    return argparser.parse_args()


def main():
    args = get_args()
    languages = Unimorph.get_languages()
    my_env = os.environ.copy()

    for i, lang in enumerate(languages):
        print()
        cmd = ['make', 'train', 'LANGUAGE=%s' % lang]
        print('(%02d/%02d) Training on language: %s' % (i + 1, len(languages), lang))
        subprocess.check_call(cmd, env=my_env)
        print()


if __name__ == '__main__':
    main()
