import sys
sys.path.append(".")
sys.path.append("..")
from flask import Flask, request, jsonify
import threading
from Retrieval_Server.modules.init_database import Initial_Database
from Retrieval_Server.modules.T2T_Retrieval.vector_retriever_build_rapid import Text2Text_Retriever_Builder
from Retrieval_Server.modules.T2T_Retrieval.text2sql_retriever_asyncio import Text2Sql_Retriever
from MDI_RAG_Image2Image_Research.src.build.basic.search_rapid import Image2Image_Retriever_Rapid
from Retrieval_Server.pipelines.T2T_Retrieval_Pipeline import main_t2t_retrieval_server_pipeline
from datetime import datetime, timezone
import concurrent.futures
import time

app = Flask(__name__)

database = None
text2text_retriever = None
text2sql_retriever = None
image2image_retrieval = None
text2image_retrieval = None

def get_database():
    global database
    if database is None:
        print("==============初始化数据库==============")
        struc_file = "Retrieval_Server/data/struc.sql"
        data_file = "Retrieval_Server/data/data.sql"
        try:
            begin_time = datetime.now()
            database = Initial_Database(struc_file, data_file)
            end_time = datetime.now()
            # database = None
            print("==============数据库初始化成功=============="+"用时：", end_time - begin_time)
        except:
            print("==============数据库初始化失败==============")
            database = None
    return database

def get_text2text_retriever(database):
    global text2text_retriever
    if text2text_retriever is None:
        print("==============初始化text2text检索器==============")
        begin_time = datetime.now()
        text2text_retriever = Text2Text_Retriever_Builder(database=None)
        end_time = datetime.now()
        print("==============text2text检索器初始化成功=============="+"用时：", end_time - begin_time)
        
    return text2text_retriever

def get_text2sql_retriever(database, llm="gpt4o"):
    global text2sql_retriever
    if text2sql_retriever is None:
        print("==============初始化text2sql检索器==============")
        begin_time = datetime.now()
        text2sql_retriever = Text2Sql_Retriever(database, llm=llm)
        end_time = datetime.now()
        print("==============text2sql检索器初始化成功=============="+"用时：", end_time - begin_time)
    return text2sql_retriever

# def get_image2image_retriever(database):
#     """加载旧版图像检索器"""
#     global image2image_retrieval
#     if image2image_retrieval is None:
#         image2image_retrieval = Image2Image_Retriever(database)
#     return image2image_retrieval

def get_image2image_retriever():
    """加载新版图像检索器（大规模）"""
    global image2image_retrieval
    if image2image_retrieval is None:
        print("==============初始化image2image检索器==============")
        begin_time = datetime.now()
        image2image_retrieval = Image2Image_Retriever_Rapid()
        end_time = datetime.now()
        print("==============image2image检索器初始化成功=============="+"用时：", end_time - begin_time)
    return image2image_retrieval

def validate_datetime(value):
    if isinstance(value, datetime):
        try:
            # 确保日期时间值在有效范围内
            value = value.astimezone(timezone.utc)
            # value = None

        except OverflowError:
            # 如果日期时间值无效，设置为 None 或其他默认值
            value = None
    return value


@app.route('/text2text_retrieval', methods=['POST'])
def process_text2text_retrieval():
    query_text = request.json.get('origin_query')
    llm = request.json.get('llm') 
    top_k = request.json.get('top_k')

    try:
        top_k = int(top_k)
    except:
        top_k = 5
    if llm is None:
        llm = "gpt4o"

    database = get_database()
    vector_retriever = get_text2text_retriever(database)
    # vector_retriever = None
    if database is None:
        pass
    else:
        text2sql_retriever = get_text2sql_retriever(database, llm)
    
    answer, metadata = main_t2t_retrieval_server_pipeline(query_text, vector_retriever, text2sql_retriever, top_k, llm)
    data_table = answer['data_table']

    result = {
            'answer': '以下是 Text2Text Retrieval 的结果。',
            'data_table': data_table,
            'metadata': metadata
        }
    
    print("生成完成")
    print("result",result)

    # 遍历并验证所有日期时间值
    try:
        for key, value in result['metadata'].items():
            print("key",key)
            
            if isinstance(value, list):
                for item in value:
                    for k, v in item.items():
                        item[k] = validate_datetime(v)
    except:
        pass

    return jsonify(result)


@app.route('/query/image2image_retrieval', methods=['POST'])
def process_image2image_retrieval():
    """新版图像检索（大规模）"""
    query_img_path = request.json.get('query_img_path')
    top_k = request.json.get('top_k')

    image2image_retriever = get_image2image_retriever()
    distances, neighbors,results = image2image_retriever.search(query_img_path, top_k)
    # print(retrieved_images_payload)
    # neighbor "47062_EGFR-肺癌-230214010TFXA-0.7-LBP.ibl.tiff_5120_15872_256_256_1.png" 
    # results = [image2image_retriever.all_infos[neighbor] for neighbor in neighbors]


    retrieved_images_information = [{
            "wsi_level": node.split("_")[-1].split(".")[0],
            "metadata_position": (node.split("_")[-5], node.split("_")[-4]),
            "patch_size": (node.split("_")[-3], node.split("_")[-2]),
            "source_wsi_name":"_".join(node.split("_")[:-5]),
            # "wsi_id":node.split("_")[0],
        } for node in results]

    return jsonify({  
        'answer':"以下是 Image2Image Retrieval 的结果。",        # 文字回答为定式
        'retrieved_images_information':retrieved_images_information
    })


@app.route('/query/health', methods=['POST','GET'])
def health():
    return "hello world!"



# if __name__ == '__main__':
#     begin_time = datetime.now()
#     database = get_database()
#     image2image_retriever = get_image2image_retriever()
#     text2text_retriever = get_text2text_retriever(database)
#     text2sql_retriever = get_text2sql_retriever(database)
#     end_time = datetime.now()
#     print("Initialization time:", end_time - begin_time)
#     app.run(debug=True, host='0.0.0.0', port="9999", use_reloader=False)

def initialize_retrievers(database):
    global text2text_retriever, text2sql_retriever, image2image_retrieval
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            # executor.submit(get_text2text_retriever, database): 'text2text_retriever',
            # executor.submit(get_text2sql_retriever, database): 'text2sql_retriever',
            executor.submit(get_image2image_retriever): 'image2image_retrieval'
        }
        for future in concurrent.futures.as_completed(futures):
            retriever_name = futures[future]
            try:
                result = future.result()
                if retriever_name == 'text2text_retriever':
                    text2text_retriever = result
                elif retriever_name == 'text2sql_retriever':
                    text2sql_retriever = result
                elif retriever_name == 'image2image_retrieval':
                    image2image_retrieval = result
            except Exception as exc:
                print(f'{retriever_name} generated an exception: {exc}')


def initialize_all():
    global database, text2text_retriever, text2sql_retriever, image2image_retrieval
    database = get_database()
    initialize_retrievers(database)

if __name__ == '__main__':
    # multiprocessing.set_start_method('spawn')
    begin_time = datetime.now()
    init_thread = threading.Thread(target=initialize_all)
    init_thread.start()
    
    app.run(debug=True, host='0.0.0.0', port="9998", use_reloader=False)
    end_time = datetime.now()
    print("本次服务运行时长:", end_time - begin_time)
    