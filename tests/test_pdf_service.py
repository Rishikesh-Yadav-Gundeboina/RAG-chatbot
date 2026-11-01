from backend.services.pdf_service import extract_pages

file_path = 'data/uploads/32dd937b-9a5a-419b-9255-1d34173edbcf.pdf'

pages = extract_pages(file_path)

if pages:
    print(f"Successfully extracted {len(pages)} pages!")
    print("--- First Page Snippet ---")
    print(f"Page Number: {pages[0]['page_number']}")
    print(f"Text Preview: {pages[0]['text'][:200]}...") # Only print first 200 characters
else:
    print("No pages were extracted. Check if the PDF path is correct.")