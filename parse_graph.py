import json

with open('user/default/workflows/video_ltx2_3_t2v.json') as f:
    data = json.load(f)

nodes = {n['id']: n for n in data['nodes']}
links = {l[0]: l for l in data['links']} # link_id -> [id, origin_id, origin_slot, target_id, target_slot, type]

api_prompt = {}

def get_input_value(node, input_name):
    # Check if it's a link
    if 'inputs' in node:
        for inp in node['inputs']:
            if inp['name'] == input_name:
                if inp.get('link') is not None:
                    link = links[inp['link']]
                    return [str(link[1]), link[2]]
    # Check if it's a widget value
    if 'widgets_values' in node:
        # We need to map widget index to name. This is tricky.
        pass
    return None

# Actually, the user's json already HAS an API format prompt embedded in it maybe?
# Wait, I noticed a "prompt" key in the JSON you showed earlier. Let's print it.
if "prompt" in data:
    with open("extracted_api.json", "w") as out:
        json.dump(data["prompt"], out, indent=2)
    print("Extracted prompt!")
else:
    print("No prompt found in JSON.")
