FROM nvidia/cuda:12.4.1-devel-ubuntu22.04


RUN apt update
RUN apt install -y python3 python3-pip unzip git


# TODO: fill in the blank
RUN git clone https://github.com/hoang1645/cinnamon-ai-pre-entrance-test-submission

RUN cd cinnamon-ai-pre-entrance-test-submission; pip install -r requirements.txt

RUN gdown 1zeN4nY3Q7O16mRdiL6vTOT-me6TTBX3a -O backend/qlora-3e.zip
RUN cd backend/; unzip qlora-3e.zip; cd ..

RUN cd backend/application; fastapi run app.py & cd ../../frontend; streamlit run app.py&
