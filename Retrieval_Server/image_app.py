import sys
import os
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional

sys.path.append(".")
sys.path.append("..")

from flask import Flask, request, jsonify, Response
from flask_cors import CORS


from MDI_RAG_Image2Image_Research.src.build.basic.search_rapid import Image2Image_Retriever_Rapid


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('image_retrieval_service.log')
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  


image_retriever = None

def get_image_retriever() -> Optional[Image2Image_Retriever_Rapid]:
    global image_retriever
    if image_retriever is None:
        logger.info("...")
        try:
            start_time = time.time()
            image_retriever = Image2Image_Retriever_Rapid()
            elapsed_time = time.time() - start_time
            logger.info(f": {elapsed_time:.2f}")
        except Exception as e:
            logger.error(f": {str(e)}")
            return None
    return image_retriever

@app.route('/health', methods=['GET'])
def health_check() -> Response:

    return jsonify({
        'status': 'ok',
        'service': 'image-retrieval',
        'time': datetime.now().isoformat(),
        'retriever_initialized': image_retriever is not None
    })

@app.route('/search', methods=['POST'])
def search_image() -> Response:

    req_data = request.get_json()
    if not req_data:
        return jsonify({
            'error': '',
            'status': 400
        }), 400
    
    query_img_path = req_data.get('query_img_path')
    top_k = req_data.get('top_k', 20)
    include_metadata = req_data.get('include_metadata', True)

    if not query_img_path:
        return jsonify({
            'error': ' query_img_path',
            'status': 400
        }), 400
    
    try:
        top_k = int(top_k)
        if top_k <= 0:
            top_k = 20
    except (ValueError, TypeError):
        top_k = 20

    retriever = get_image_retriever()
    if retriever is None:
        return jsonify({
            'error': 'fail',
            'status': 500
        }), 500

    try:
        logger.info(f" {query_img_path}")
        start_time = time.time()
        distances, neighbors, results = retriever.search(query_img_path, top_k)
        search_time = time.time() - start_time

        retrieved_results = []
        for i, (distance, neighbor, result) in enumerate(zip(distances, neighbors, results)):
            try:
                parts = result.split("_")
                level = parts[-1].split(".")[0]
                position = (parts[-5], parts[-4])
                patch_size = (parts[-3], parts[-2])
                wsi_name = "_".join(parts[1:-5])
                wsi_id = parts[0]
                
                result_item = {
                    "rank": i + 1,
                    "distance": float(distance),
                    "wsi_level": level,
                    "position": position,
                    "patch_size": patch_size,
                    "wsi_name": wsi_name,
                    "wsi_id": wsi_id,
                }
                
                try:
                    result_item["image_url"] = f""
                except Exception as e:
                    logger.warning(f": {str(e)}")
                
                retrieved_results.append(result_item)
            except Exception as e:
                logger.warning(f": {str(e)}")
        
        
        connected_regions = []
        if include_metadata:
            try:
                connected_regions = retriever.get_combined_regions(results)
                
                
                formatted_regions = []
                for region_index, region in enumerate(connected_regions):
                    formatted_region = {
                        "region_id": region_index + 1,
                        "tile_count": len(region),
                        "wsi_name": region[0]["name"] if region else "",
                        "tiles": region
                    }
                    formatted_regions.append(formatted_region)
                
                connected_regions = formatted_regions
            except Exception as e:
                logger.error(f": {str(e)}")
        
        
        response = {
            'status': 200,
            'message': 'ok',
            'search_time': search_time,
            'query_image': query_img_path,
            'result_count': len(retrieved_results),
            'results': retrieved_results
        }
        
        if include_metadata:
            response['connected_regions'] = connected_regions
            response['connected_region_count'] = len(connected_regions)
        
        logger.info(f" {len(retrieved_results)}  {search_time:.2f}")
        return jsonify(response)
    
    except Exception as e:
        logger.error(f": {str(e)}")
        return jsonify({
            'error': f': {str(e)}',
            'status': 500
        }), 500

@app.route('/batch_search', methods=['POST'])
def batch_search_images() -> Response:


    req_data = request.get_json()
    if not req_data:
        return jsonify({
            'error': '',
            'status': 400
        }), 400
    
    query_img_paths = req_data.get('query_img_paths', [])
    top_k = req_data.get('top_k', 20)
    combine_results = req_data.get('combine_results', True)
    
    
    if not query_img_paths or not isinstance(query_img_paths, list):
        return jsonify({
            'error': 'query_img_paths',
            'status': 400
        }), 400
    
    try:
        top_k = int(top_k)
        if top_k <= 0:
            top_k = 20
    except (ValueError, TypeError):
        top_k = 20

    retriever = get_image_retriever()
    if retriever is None:
        return jsonify({
            'error': '',
            'status': 500
        }), 500

    try:
        logger.info(f" {len(query_img_paths)} ")
        start_time = time.time()
        
        if combine_results:
            
            distances, neighbors, connected_regions, results = retriever.search_multi_imgs(
                query_img_paths, m=1, n=1, top_k=top_k
            )
            search_time = time.time() - start_time
            
            
            retrieved_results = format_search_results(distances, neighbors, results)
            
            
            response = {
                'status': 200,
                'message': 'ok',
                'search_time': search_time,
                'query_count': len(query_img_paths),
                'result_count': len(retrieved_results),
                'results': retrieved_results,
                'connected_regions': connected_regions,
                'connected_region_count': len(connected_regions),
            }
        else:
            
            all_results = []
            for i, query_path in enumerate(query_img_paths):
                individual_start = time.time()
                distances, neighbors, results = retriever.search(query_path, top_k)
                individual_time = time.time() - individual_start
                
                retrieved_results = format_search_results(distances, neighbors, results)
                
                all_results.append({
                    'query_image': query_path,
                    'search_time': individual_time,
                    'result_count': len(retrieved_results),
                    'results': retrieved_results
                })
            
            search_time = time.time() - start_time
            
            
            response = {
                'status': 200,
                'message': 'ok',
                'total_search_time': search_time,
                'query_count': len(query_img_paths),
                'individual_results': all_results
            }
        
        logger.info(f" {len(query_img_paths)} {search_time:.2f}")
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"{str(e)}")
        return jsonify({
            'error': f' {str(e)}',
            'status': 500
        }), 500

def format_search_results(distances, neighbors, results):

    formatted_results = []
    for i, (distance, neighbor, result) in enumerate(zip(distances, neighbors, results)):
        try:
            parts = result.split("_")
            level = parts[-1].split(".")[0]
            position = (parts[-5], parts[-4])
            patch_size = (parts[-3], parts[-2])
            wsi_name = "_".join(parts[1:-5])
            wsi_id = parts[0]
            
            result_item = {
                "rank": i + 1,
                "distance": float(distance),
                "wsi_level": level,
                "position": position,
                "patch_size": patch_size,
                "wsi_name": wsi_name,
                "wsi_id": wsi_id,
                "image_url": f""
            }
            formatted_results.append(result_item)
        except Exception as e:
            logger.warning(f" {str(e)}")
    
    return formatted_results

if __name__ == '__main__':
    
    logger.info("...")
    get_image_retriever()
    
    
    port = int(os.environ.get('PORT', 9999))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
