import os
import gdown


def test_gdown():
    gdown.download(
        "https://drive.google.com/file/d/1zeN4nY3Q7O16mRdiL6vTOT-me6TTBX3a/view?usp=sharing",
        "backend/qlora_3e.zip",
    )
    os.system("cd backend; unzip qlora_3e.zip; cd ..")
