import re

def clean_text(text:str) -> str:
    text = text.replace("\x00",'')
    text = re.sub(r"\s+",' ',text)
    return text.strip()


if __name__ == "__main__":
    messy_pdf_text = "Hello\x00 World!   \n\nThis   is   a   test   string.\nIt has  \r\nrandom  newlines   in  \n\n\n  weird places.   \nAnd   multiple     spaces.\t\t\t\n\nGoodbye."
    print("--- ORIGINAL TEXT ---")
    print(repr(messy_pdf_text)) # repr() shows you the hidden \n and \t characters

    print("\n--- CLEANED TEXT (Your current version - NO SPACES) ---")
    print(clean_text(messy_pdf_text)) 

