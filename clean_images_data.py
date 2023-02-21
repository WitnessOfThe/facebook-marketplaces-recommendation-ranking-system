import chardet
import csv
import pandas as pd
import cchardet as chardet

class DataImages:

    def __init__(self,name) -> None:
        self.name = name
        self.df   = self.load_images_data(name)

    def load_images_data(self,name):
        #'detect incoding'
        with open(name, 'rb') as f:
            result = chardet.detect(f.read())
        encoding = result['encoding']
        print(encoding)        
        #'Sniff cvs and convert to pandas'       
        with open(name, 'r',encoding=encoding,newline='') as csvfile:
            reader   = csv.reader(  csvfile, delimiter=',',quotechar='"')# delimiter=',',quotechar='"',lineterminator='\n')#,
            data = list(reader)
            return pd.DataFrame(data[1:], columns=data[0])
if __name__ == '__main__':
    print(DataImages('Images.csv').df.head())
    print(DataImages('Images.csv').df.info())
    pass
