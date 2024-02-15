from os import listdir

from config import LABELED_DATA_PDFS_PATH


if __name__ == "__main__":
    for pdf_name in listdir(LABELED_DATA_PDFS_PATH):
        print(pdf_name)
