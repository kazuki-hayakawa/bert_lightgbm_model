import os
import glob
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split


def read_text(text_filepath):
    """ livedoor ニュースの形式に合わせて、4行目以降の本文のみを読み取る """
    with open(text_filepath, 'r') as f:
        lines = f.readlines()
        lines = lines[3:]

    text = ' '.join(lines)
    # 全角スペース、改行コードの削除
    text = text.replace('\u3000', '').replace('\n', '')
    return text


def main():
    # 事前に download_livedoor_news.sh を実行してデータを取得しておく
    exclude_files = ['CHANGES.txt', 'README.txt', 'LICENSE.txt']
    all_file_paths = glob.glob('../../data/raw/text/**/*.txt', recursive=True)
    all_file_paths = [p for p in all_file_paths
                      if os.path.basename(p) not in exclude_files]

    df_processed = pd.DataFrame(columns=['id', 'media', 'text'])
    for idx, filepath in enumerate(tqdm(all_file_paths)):
        media = os.path.dirname(filepath).replace('../../data/raw/text/', '')
        text = read_text(filepath)
        row = pd.Series([idx + 1, media, text], index=df_processed.columns)
        df_processed = df_processed.append(row, ignore_index=True)

    df_train, df_test, _, _ = train_test_split(
        df_processed, df_processed['media'], test_size=0.1, random_state=0,
        stratify=df_processed['media']
    )
    df_train.to_csv('../../data/processed/train_dataset.csv', index=False)
    df_test.to_csv('../../data/processed/test_dataset.csv', index=False)


if __name__ == '__main__':
    main()
