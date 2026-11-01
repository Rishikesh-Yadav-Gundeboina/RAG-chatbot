from pathlib import Path
import uuid

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True,exist_ok=True)

def save_files(filename:str,content:bytes) -> str :
    file_id = str(uuid.uuid4())
    extension = Path(filename).suffix
    file_path = UPLOAD_DIR/f"{file_id}{extension}"
    with open(file_path,'wb') as file:
        file.write(content)
    return str(file_path)

