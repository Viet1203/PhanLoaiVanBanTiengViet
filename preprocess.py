import pandas as pd
from pyvi import ViTokenizer
import re

class TextPreprocessor:
    # Tạo 1 set stop_w đọc từ file csv 'stopword'
    def __init__(self, stopwords_file):
        self.stopword_set = self.load_stopwords(stopwords_file)
    def load_stopwords(self, stopwords_file):
        stopword_df = pd.read_csv(stopwords_file, header=None, names=['stopword'])
        return set(stopword_df['stopword'])
    # Loại bỏ các stopword ra khỏi văn bản.
    def remove_stopwords(self, line):
        words = [] 
        for word in line.strip().split(): 
            if word not in self.stopword_set: 
                words.append(word) 
        return ' '.join(words)

    def preprocess_text(self, text):
        # Chuyển văn bản về chữ thường.
        text = text.lower()
        # Loại voe các kí tự đặc biệt, ngày tháng năm, đường dẫn, sđt,...
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'\d{1,2}/\d{1,2}', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        # Loại bỏ các stopword
        text = self.remove_stopwords(text)
        # Tách từ tiếng Việt sử dụng ViTokenizer
        text = ViTokenizer.tokenize(str(text))
        return text

    # Sử dụng các hàm trên cho toàn tập dữ liệu.
    def preprocess_data(self, data):
        data['Content'] = data['Content'].apply(self.preprocess_text)
        return data

