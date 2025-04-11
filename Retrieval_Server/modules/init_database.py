import json, sys, uuid, hashlib, ast, requests, os
sys.path.append(".")
sys.path.append("..")
sys.path.append("...")
from PIL import Image
from sqlalchemy import create_engine, MetaData
from sqlalchemy.sql import text
from llama_index.core import SQLDatabase
from llama_index.core.schema import TextNode, ImageNode
import urllib.parse
from tqdm import tqdm
# from Retrieval_Server.modules.load_images import load_svsortiff_image, load_pngorjpg_image
from sqlalchemy.sql import text
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class Initial_Database():
    def __init__(self, struc_file, data_file, 
                #  text_json_file, image_json_file 
                 ):
        # user_name, password, ip, port, db_name = "postgres", "sr19970203", "127.0.0.1", "5432", "abp_medical"  # local
        user_name, password, ip, port, db_name = r'aladdin_remap_ro', r'[rFPe!8|b(~lrnADYI91r>H:@S~z3x', "localhost", "5432", "aladdin_remap"
        self.uuid_namespace = uuid.NAMESPACE_URL
        encoded_password = urllib.parse.quote_plus(password)
        print(f'postgresql://{user_name}:{encoded_password}@{ip}:{port}/{db_name}')

        self.engine = create_engine(f'postgresql://{user_name}:{encoded_password}@{ip}:{port}/{db_name}')
        metadata = MetaData()
        # self.init_structure(struc_file)
        # self.load_sql_data(data_file)
        metadata.create_all(self.engine)
        # with open(text_json_file, "r") as text_file:
        #     text_json = json.load(text_file)
        # with open(image_json_file, "r") as image_file:
        #     image_json = json.load(image_file)
        self.sql_database = SQLDatabase(self.engine)
        self.text_nodes = self.create_text_node_from_sql(
            # text_json
        )
        # local_not_exist = not os.path.exists("/home/mdi/suri/MDI_RAG_project/Retrieval_Server/caches/images")
        # self.load_image_from_filecode()   #   WSI
        # self.image_nodes = self.create_image_node_from_sql(
        #     # image_json,
        #     local_not_exist=local_not_exist
        # )
        print("Finish Initialized SQL Databases.")

    def init_structure(self, struc_file):
        with open(struc_file, 'r') as struc_file:
            struc_sql = struc_file.read()
        with self.engine.connect() as connection:
            sql_command = struc_sql.split(";")
            for command in sql_command:
                if command.strip():
                    connection.execute(text(command))
            # connection.commit()       

    def load_sql_data(self, data_file):
        with open(data_file, 'r') as data_file:
            data_sql = data_file.read()
        with self.engine.connect() as connection:
            sql_command = data_sql.split("\n")
            for command in sql_command:
                if command != "":
                    connection.execute(text(command.replace("0001-01-01", "1970-01-01")))       #  
            # connection.commit()        

    # def create_text_node_from_sql(self):
    #     nodes = []
    #     name, column = "disease_data", "data"

    #     with self.engine.connect() as connection:
    #         contents = connection.execute(text(f"SELECT {column} FROM {name}")).fetchall()
    #         for row in contents:
    #             single_row = str(row[0])
    #             row_dict = ast.literal_eval(single_row)
    #             for key in row_dict:
    #                 if isinstance(row_dict[key], str):
    #                     nodes.append(TextNode(id=uuid.uuid5(self.uuid_namespace, row_dict[key]),
    #                                         text=row_dict[key],
    #                                         metadata=row_dict))
    #     return nodes
    # def create_text_node_from_sql(self):
    #     nodes = []
    #     with self.engine.connect() as connection:
    #         #  
    #         tables = connection.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema='public'")).fetchall()
    #         for table in tables:
    #             table_name = table[0]
    #             #  
    #             contents = connection.execute(text(f"SELECT * FROM {table_name}")).fetchall()
    #             for row in contents:
    #                 row_dict = {}
    #                 for i in range(len(row)):
    #                     row_dict[str(i)] = str(row[i])
    #                 nodes.append(TextNode(id=uuid.uuid5(self.uuid_namespace, str(row_dict)),
    #                                     text=str(row_dict),
    #                                     metadata=row_dict))
    #     return nodes

    def create_text_node_from_sql(self):
        nodes = []
        with self.engine.connect() as connection:
            #  
            tables = connection.execute(
                text("SELECT table_name FROM information_schema.tables WHERE table_schema='public'")
            ).fetchall()
            
            for table in tables:
                table_name = table[0]
                
                #  
                columns = connection.execute(
                    text(f"SELECT column_name FROM information_schema.columns WHERE table_name='{table_name}' AND table_schema='public'")
                ).fetchall()
                column_names = [column[0] for column in columns]

                #  
                contents = connection.execute(text(f"SELECT * FROM {table_name}")).fetchall()
                for row in contents:
                    row_dict = dict(zip(column_names, row))  #  
                    #  {table_name} 
                    row_dict["table_name"] = table_name
                    nodes.append(TextNode(id=uuid.uuid5(self.uuid_namespace, str(row_dict)),
                                          text=str(row_dict),
                                          metadata=row_dict))
        return nodes

    def create_image_node_from_sql(self, local_not_exist=True):
        nodes = []
        name, column = "file_source", "file_path"
        with self.engine.connect() as connection:
            contents = connection.execute(text(f"SELECT {column} FROM {name}"))
            for row in contents:
                img_path = str(row[0])
                if img_path.endswith(".svs") or img_path.endswith(".tiff"):
                    total_img_pathes, positiones, levels = load_svsortiff_image(img_path, open_image=local_not_exist)
                # elif img_path.endswith(".svs") or img_path.endswith(".tiff"):
                #     total_img_pathes, positiones = load_pngorjpg_image(img_path, open_image=local_not_exist)
                source_path = total_img_pathes[0]
                for i in range(1, len(total_img_pathes)):
                    cur_img_path = total_img_pathes[i]
                    nodes.append(ImageNode(id=uuid.uuid5(self.uuid_namespace, cur_img_path),
                                           image_path=cur_img_path,
                                           metadata={
                                               "position":positiones[i],
                                               "level":levels[i],
                                               "size":(2048, 2048),
                                               "wsi_image_source":source_path,
                                               "database_source":f"column:{column} in tabel:{name}"
                                            }))
        return nodes

    def load_image_from_filecode(self):
        """  WSI  """
        save_path = "/hpc2hdd/home/ysi538/retrieval/caches/wsi_image"
        os.makedirs(save_path, exist_ok=True)
        name = "file_source"
        with self.engine.connect() as connection:
            contents_id = connection.execute(text(f"SELECT id FROM {name}"))
            contents_code = connection.execute(text(f"SELECT source_code FROM {name}"))
            contents_name = connection.execute(text(f"SELECT file_name FROM {name}"))
            file_list = os.listdir(save_path)
            for id, code, name in tqdm(zip(contents_id, contents_code, contents_name)):
                #  str(name[0]) svs tiff
                if str(name[0]).endswith(".svs") or str(name[0]).endswith(".tiff"):
                    if str(name[0]) not in file_list:
                        image_url = self.request_image_url(str(code[0]))
                        with open(os.path.join(save_path, str(name[0])), 'wb') as f:
                            f.write(requests.get(image_url).content)
                    else:
                        old_path = os.path.join(save_path, str(name[0]))
                        new_path = os.path.join(save_path, f"{str(id[0])}_{str(name[0])}")
                        if os.path.exists(old_path):
                            os.rename(old_path, new_path)
                        else:
                            print(f" : {old_path}")

    def request_image_url(self, source_code):
        url = 'https://re_map.ibingli.com/diseaselib/api/v1/lib/common/getFileUrl'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhdWQiOiIiLCJleHAiOjIwMzQyMjI5MDMsImlhdCI6MTcxODg2MjkwMywiaXNzIjoiZGlzZWFzZWxpYkxvZ2luIiwibmJmIjoxNzE4ODYyOTAzLCJ1aWQiOjEwNX0.osomnuzu9Q5YNYrZsLwK6CC-FnTgepj0nd866a_-rjo'  #  Bearer Token
        }
        data = {
            "fileSourceCode": source_code
        }

        response = requests.post(url, headers=headers, json=data, verify=False)
        json_response = json.loads(response.text)

        return json_response["data"]["fileUrl"]
                



if __name__ == "__main__":
    struc_file = "/home/mdi/suri/MDI_RAG_project/Retrieval_Server/data/struc.sql"
    data_file = "/home/mdi/suri/MDI_RAG_project/Retrieval_Server/data/data.sql"
    # text_json_file = "Retrieval_Server/data/text_columns.json"
    # image_json_file = "Retrieval_Server/data/image_columns.json"
    database = Initial_Database(struc_file,
                                data_file,
                                # text_json_file,
                                # image_json_file
                                )
    