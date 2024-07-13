import openai
import pandas as pd
import os
import numpy as np
from openai.embeddings_utils import get_embedding, cosine_similarity


#es klasi gaaketebs teqstebis embedirebas
#https://stackoverflow.com/questions/22872952/set-file-path-for-to-csv-in-pandas  -- konkretur fokldershi daseivebis tema



pd.options.display.max_colwidth = 200
openai.api_key = os.environ.get('openai_apik')
embedding_model = "text-embedding-ada-002"

path = '/Embedding Matching Programme/XLS_files'
csvpath = 'C:\\Users\\User\\PycharmProjects\\tt\\Embedding Matching Programme\\CSV_files'
embedpath = 'C:\\Users\\User\\PycharmProjects\\tt\\Embedding Matching Programme\\CSV_embeddings'
matchpath = 'C:\\Users\\User\\PycharmProjects\\tt\\Embedding Matching Programme\\Matchings'


class Text_embedder:

    def __init__(self, path, csvpath, embedpath, matchpath):
        self.path = path
        self.csvpath = csvpath
        self.embedpath = embedpath
        self.matchpath = matchpath

    def convert(self):
        self.xlfiles = os.listdir(self.path)
        for eachfile in self.xlfiles:
            if eachfile.endswith('.xlsx'):
                cleanfilename = eachfile.replace('.xlsx', '')
                xlfile = pd.ExcelFile(self.path + '\\' + eachfile)
                sheets = xlfile.sheet_names
                for eachsheet in sheets:
                    sheetsdata = xlfile.parse(eachsheet)
                    csvname = cleanfilename + '_CSV' + '.csv'
                    sheetsdata.to_csv(self.csvpath + '\\' + csvname, index=False, encoding='utf-8-sig')

    def text_embedder(self):

        self.csvfiles = os.listdir(self.csvpath)
        for filename in self.csvfiles:
            if filename.endswith('.csv'):
                embfilename = self.embedpath + '\\' + filename
                df = pd.read_csv(self.csvpath + '\\' + filename)
                # df['names'] = df['names'].astype(str)
                df['names'] = df['names'].apply(lambda x: str(x))
                df['embeddings'] = df['names'].apply(lambda x: get_embedding(x, embedding_model))
                newfilename = 'embs_' + filename
                df.to_csv(embfilename)

    def embd_match(self):

        self.embedfiles = os.listdir(self.embedpath)

        for filename in self.embedfiles:
            if 'itech' in filename and filename.endswith('.csv'):
                df = pd.read_csv(self.embedpath + '\\' + filename)
                df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)

        for filename in self.embedfiles:
            if 'itech' not in filename and filename.endswith('.csv'):

                df2 = pd.read_csv(self.embedpath + '\\' + filename)
                lookupnames = []
                names = []
                similarities = []
                # product_embedding = get_embedding(product_description, engine=embedding_model)
                df2['embeddings'] = df2['embeddings'].apply(eval).apply(np.array)

                for o, i in enumerate(df['embeddings']):
                    df2['similarity'] = df2['embeddings'].apply(lambda x: cosine_similarity(x, i))
                    names.append(df2.sort_values("similarity", ascending=False).head(1)['names'])
                    similarities.append(df2.sort_values("similarity", ascending=False).head(1)['similarity'])
                    lookupnames.append(df['names'][o])

                newname = filename.replace("_CSV.csv", '')

                product_overview = pd.DataFrame({"lkp_names": lookupnames, "names": names, 'smts': similarities})
                product_overview.to_excel(self.matchpath + '\\' + newname + '_matched' + ".xlsx", index=False)


prog = Text_embedder(path, csvpath, embedpath, matchpath)
prog.convert()
prog.text_embedder()
prog.embd_match()

