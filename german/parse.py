import pywikibot
from pywikibot import pagegenerators
from pywikibot.comms import http
import wikitextparser as wtp
from bs4 import BeautifulSoup
import urllib.request
from tqdm import tqdm

site = pywikibot.Site(code='en', fam='wiktionary')
site.login()

MAGIC = len('inflection-table-de-')

mapper = {
    'nominative': 'NOM', 'genitive': 'GEN', 'dative': 'DAT', 'accusative': 'ACC', 'ablative': 'ABL', 'vocative': 'VOC',
    'm': 'MASC', 'f': 'FEM', 'n': 'NEUT'
}

cat = pywikibot.Category(site, "Category:German nouns")
pages = cat.articles()

with tqdm(total=40883) as pbar, open('deu.tsv', 'w') as fout:
    for page in pagegenerators.PreloadingGenerator(pages, 100):
        title = page.title()
        # print(title)
        path = f'https://en.wiktionary.org/wiki/{page.title(as_url=True)}'
        with urllib.request.urlopen(path) as resp:
            soup = BeautifulSoup(resp, 'html.parser')
        for table in soup.find_all(class_="inflection-table-de"):
            for cl in table['class']:
                if 'inflection-table-de-' in cl:
                    decl_type = cl[MAGIC:]
            all_words = []
            for row in table.find_all('tr')[2:]:
                case = mapper[row.find('th').text.rstrip()]
                words = row.find_all('td')

                if decl_type == 'pl':
                    for word in words[1].find_all('a'):
                        all_words.append((f'{case};PL', word.text))
                else:
                    for word in words[2].find_all('a'):
                        all_words.append((f'{case};SG;{mapper[decl_type]}', word.text))
                    if len(words) > 3:
                        for word in words[4].find_all('a'):
                            all_words.append((f'{case};PL;{mapper[decl_type]}', word.text))

            for i in all_words:
                fout.write(f'{title}\t{i[1].rstrip()}\tN;{i[0]}\n')
        pbar.update(1)