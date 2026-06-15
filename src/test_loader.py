# test_loader.py

from src.document_loader import load_and_split_pdf, get_pdf_info

# Change this to your actual PDF filename
chunks = load_and_split_pdf('data/Dharun_resume (1).pdf')
info = get_pdf_info(chunks)

print('\n📊 PDF Stats:')
for key, value in info.items():
    print(f'  {key}: {value}')