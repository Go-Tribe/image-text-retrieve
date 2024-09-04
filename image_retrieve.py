import os
import uuid
from tqdm import tqdm
from qdrant_client import models, QdrantClient

from clip_client import ChineseClipTorch

from loguru import logger

def get_dir_files_path(file_dir, filetype='.txt'):
    """
    Obtain the absolute paths of all files with the specified file type in the directory.
    
    Args:
        file_dir (str): The directory path.
        filetype (str): The file type, e.g. '.txt'. If None, all files will be included.
    
    Returns:
        list: The absolute paths of all files with the specified file type.
    """
    files_path = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if filetype is None or (os.path.splitext(file)[1] == filetype):
                files_path.append(os.path.join(root, file))
    return files_path


class ImageRetrieve:
    def __init__(self) -> None:
        self.clip_client = ChineseClipTorch()
        self.db_path = "./data/vector_db"
        self.qdrant = QdrantClient(path=self.db_path)
        self.collection_name = "image_retrieve"
        
        self.vector_dim = 768
        self.topn = 10
        
    def create_collection(self):
        """
        Create a new collection in the Qdrant database if it doesn't already exist.
        """
        if not self.qdrant.collection_exists(self.collection_name):
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_dim, # Vector size is defined by used model
                    distance=models.Distance.COSINE
                )
            )
            logger.info(f"Created {self.collection_name} successfully")
        else:
            logger.info(f"{self.collection_name} collection already exists")

    def insert_single_data(self, id, doc):
        """
        Insert a single data record into the Qdrant database.
        
        Args:
            id (str): The unique identifier for the data record.
            doc (dict): The data record, including the image path.
        """
        self.qdrant.upload_records(
            collection_name=self.collection_name,
            records=[
                models.Record(
                    id=id,
                    vector=self.clip_client.compute_image_features(doc["image_path"])[0],
                    payload=doc
                )
            ]
        )

    def batch_insert_data(self, documents):
        """
        Insert a batch of data records into the Qdrant database.
        
        Args:
            documents (list): A list of data records, each containing an image path and a unique identifier.
        """
        self.qdrant.upload_records(
            collection_name=self.collection_name,
            records=[
                models.Record(
                    id=doc["id"],
                    vector=self.clip_client.compute_image_features(doc["image_path"])[0],
                    payload=doc
                ) for idx, doc in tqdm(enumerate(documents), total=len(documents))
            ]
        )

    def text2image(self, text):
        """
        Retrieve the most similar images to the given text query.
        
        Args:
            text (str): The text query.
        
        Returns:
            list: The paths of the top-n most similar images.
        """
        result = []
        hits = self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=self.clip_client.compute_text_features(text)[0],
            limit=self.topn
        )
        for hit in hits:
            image_path = hit.payload["image_path"]
            score = hit.score
            result.append(image_path)
        
        return result

    def image2image(self, image_path):
        """
        Retrieve the most similar images to the given image.
        
        Args:
            image_path (str): The path of the input image.
        
        Returns:
            list: The paths of the top-n most similar images.
        """
        result = []
        hits = self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=self.clip_client.compute_image_features(image_path)[0],
            limit=self.topn
        )
        for hit in hits:
            image_path = hit.payload["image_path"]
            score = hit.score
            result.append(image_path)
        
        return result
    

if __name__ == "__main__":
    
    image_retrieve = ImageRetrieve()
    image_retrieve.create_collection()
    
    image_dir = "./data/images"
    files = get_dir_files_path(image_dir, '.png')
    documents = []
    for idx, file in enumerate(files):
        document = {
            "image_path": file,
            "id": str(uuid.uuid4())
        }
        documents.append(document)

    image_retrieve.batch_insert_data(documents)
    
    query = "书包"
    result = image_retrieve.text2image(query)
    print(result)