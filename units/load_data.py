from langchain.document_loaders import PyPDFLoader, TextLoader


def load_data(path, file_type):
    if file_type == 'application/pdf':
        loader_first = PyPDFLoader(path)
        pages = loader_first.load_and_split()
        return pages
    elif file_type == 'text/plain':
        loader_first = TextLoader(path, encoding='utf-8')
        pages = loader_first.load_and_split()
        return pages


def lazy_load_data(path, file_type):
    if file_type == 'application/pdf':
        loader_first = PyPDFLoader(path)
        pages = loader_first.load()
        return pages
    elif file_type == 'text/plain':
        loader_first = TextLoader(path, encoding='utf-8')
        pages = loader_first.load()
        return pages
