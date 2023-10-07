import pdfplumber

with pdfplumber.open("chatle.pdf") as pdf:

    for page in pdf.pages:

        text = page.extract_text()  # 提取文本

        txt_file = open("chatle.txt",

                        mode='a', encoding='utf-8')

        txt_file.write(text)
