import pandas as pd

from util import util


class Unimorph:
    name = 'unimorph'
    raw_fname = 'raw.tsv'
    separator = '\t'

    languages = ['por', 'spa', 'eng', 'deu']

    def __init__(self, data_path, language):
        self.language = language
        self.data_path = '%s/%s/' % (data_path, self.name, language)

    @classmethod
    def read_data_orig(cls, data_path, language):
        df = cls.read_data_raw(data_path, language)

        df['tag'] = df[2]
        df['concept'] = df[0]
        df['word'] = df[1].apply(lambda x: str(x))
        df['id'] = range(df.shape[0])

        return df

    @classmethod
    def read_data_raw(cls, data_path, language):
        fname = '%s/%s/%s/%s' % (data_path, cls.name, language, cls.raw_fname)
        return pd.read_csv(fname, sep=cls.separator, header=None)

    @classmethod
    def write_data(cls, data_path, language, folds, alphabets):
        fname = '%s/%s/%s/processed.pckl' % (data_path, cls.name, language)
        util.write_data(fname, (folds, alphabets))

    @classmethod
    def read_data(cls, data_path, language):
        fname = '%s/%s/%s/processed.pckl' % (data_path, cls.name, language)
        return util.read_data(fname)

    @classmethod
    def get_languages(cls):
        return cls.languages