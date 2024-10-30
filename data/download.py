import os
import urllib.request
import zipfile
import gzip
import shutil

WIKI_DUMP_URL = "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2"
TEXT8_URL = "http://mattmahoney.net/dc/text8.zip"
CORPUS_DIR = os.path.join(os.path.dirname(__file__), "corpus")


def download_text8(dest_dir=CORPUS_DIR):
    os.makedirs(dest_dir, exist_ok=True)
    out_path = os.path.join(dest_dir, "text8")
    if os.path.exists(out_path):
        return out_path
    zip_path = os.path.join(dest_dir, "text8.zip")
    print(f"downloading text8 from {TEXT8_URL}")
    urllib.request.urlretrieve(TEXT8_URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    os.remove(zip_path)
    return out_path


def download_wikipedia(dest_dir=CORPUS_DIR):
    os.makedirs(dest_dir, exist_ok=True)
    out_path = os.path.join(dest_dir, "wiki.txt")
    if os.path.exists(out_path):
        return out_path
    print("wikipedia dump must be preprocessed externally.")
    print(f"place the tokenized text at {out_path}")
    return out_path


if __name__ == "__main__":
    download_text8()
