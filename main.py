import os

def main():
    if os.path.exists('./dataset.csv'):
        df = pd.read_csv('dataset.csv', index_col=None)
        df


if __name__ == '__main__':
    main()
