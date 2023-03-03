import chardet
import csv
import pandas as pd
import chardet as chardet

class DataTabular:

    def __init__(self,name) -> None:
        self.name = name
        self.df   = self.load_data(name)
#        self.df   = self.clean_price_column()
        
    def load_data(self,name):
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

    def clean_price_column(self):
        self.df = self.df.drop(self.df.columns[0],axis=1)
        self.df.dropna(subset='price',how='any',inplace=True)
        self.df.reset_index(inplace=True)
        self.df['price'] = self.df['price'].str.replace('£','')
        self.df['price'] = self.df['price'].str.replace(',','')
        self.df['price'] = pd.to_numeric(self.df['price'])
        return self.df

    def remove_char_from_string(self,value):
        return re.sub(r'\D', '',value)

if __name__ == '__main__':
    dt = DataTabular('Products.csv')
    print(dt.df.info())    
 #   print(dt.clean_price_column().info())
