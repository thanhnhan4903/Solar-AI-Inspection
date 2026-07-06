import requests
try:
    res = requests.get('http://127.0.0.1:8000/api/v1/review/items')
    items = res.json().get('items', [])
    if items:
        print(f"Items count: {len(items)}")
        print(f"First item image_url: {items[0].get('image_url')}")
        print(f"First item annotated_image_url: {items[0].get('annotated_image_url')}")
        print(f"First item review_item_id: {items[0].get('review_item_id')}")
except Exception as e:
    print(e)
