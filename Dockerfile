FROM python:3.10

RUN pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install transformers sentence-transformers==2.2.2 fastapi "uvicorn[standard]" pydantic numpy

WORKDIR /app
COPY . /app

EXPOSE 8000

CMD ["uvicorn", "app.routes.neural_network:app", "--host", "0.0.0.0", "--port", "8000"]