import sys
sys.path.append(".")
sys.path.append("..")
from flask import Flask, request, jsonify
import datetime
import time
import logging
import concurrent.futures
from MDI_RAG_Image2Image_Research.src.build.basic.search_rapid import Image2Image_Retriever_Rapid


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)


image2image_retrieval = None

def get_image2image_retriever():

    global image2image_retrieval
    if image2image_retrieval is None:

        begin_time = datetime.datetime.now()
        try:
            image2image_retrieval = Image2Image_Retriever_Rapid()
            end_time = datetime.datetime.now()
            logger.info(f" {end_time - begin_time}")
        except Exception as e:
            logger.error(f" {str(e)}")
            image2image_retrieval = None
    return image2image_retrieval

@app.route('/query/image2image_retrieval', methods=['POST'])
def process_image2image_retrieval():

    
    query_img_path = request.json.get('query_img_path')
    top_k = request.json.get('top_k', 20)
    
    try:
        top_k = int(top_k)
    except (ValueError, TypeError):
        top_k = 20
        
    
    if not query_img_path:
        return jsonify({
            'error': 'query_img_path',
            'status': 400
        }), 400
    
    
    try:
        image2image_retriever = get_image2image_retriever()
        if not image2image_retriever:
            return jsonify({
                'error': 'failed',
                'status': 500
            }), 500
            
        
        start_time = time.time()
        distances, neighbors, results = image2image_retriever.search(query_img_path, top_k)
        end_time = time.time()
        search_time = end_time - start_time
        
        
        retrieved_images_information = []
        for node in results:
            try:
                level = node.split("_")[-1].split(".")[0]
                position = (node.split("_")[-5], node.split("_")[-4])
                patch_size = (node.split("_")[-3], node.split("_")[-2])
                wsi_name = "_".join(node.split("_")[:-5])
                
                retrieved_images_information.append({
                    "wsi_level": level,
                    "metadata_position": position,
                    "patch_size": patch_size,
                    "source_wsi_name": wsi_name,
                })
            except Exception as e:
                logger.warning(f" {str(e)}")
        
        return jsonify({  
            'status': 200,
            'message': "ok",
            'search_time': search_time,
            'retrieved_images_count': len(retrieved_images_information),
            'retrieved_images_information': retrieved_images_information
        })
    except Exception as e:
        logger.error(f": {str(e)}")
        return jsonify({
            'error': f': {str(e)}',
            'status': 500
        }), 500

@app.route('/query/health', methods=['GET'])
def health():

    return jsonify({
        'status': 'healthy',
        'service': 'image2image_retrieval',
        'timestamp': datetime.datetime.now().isoformat(),
    })

def initialize_retriever():

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(get_image2image_retriever)
        try:
            future.result(timeout=300)  
            logger.info("")
        except concurrent.futures.TimeoutError:
            logger.error("")
        except Exception as e:
            logger.error(f": {str(e)}")

if __name__ == '__main__':
    
    initialize_retriever()
    
    app.run(debug=False, host='0.0.0.0', port=9999, threaded=True)
    