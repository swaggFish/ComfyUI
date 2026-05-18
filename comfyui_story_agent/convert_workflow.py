from playwright.sync_api import sync_playwright
import json
import time

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto("http://127.0.0.1:8188")
        
        # Wait for the graph canvas and the app object
        page.wait_for_selector("#graph-canvas", timeout=10000)
        
        # ComfyUI's app object is initialized asynchronously
        page.wait_for_function("typeof window.app !== 'undefined'", timeout=10000)
        
        with open("/home/mikeyb/Documents/AI/ComfyUI/comfyui_story_agent/skills/LTX2.3 video reasoning final.json", "r") as f:
            workflow_data = f.read()
            
        page.evaluate(f"""
            (workflowData) => {{
                window.app.loadGraphData(JSON.parse(workflowData));
            }}
        """, workflow_data)
        
        # Wait a moment for graph to process
        time.sleep(1)
        
        api_format = page.evaluate("""
            () => {
                return window.app.graphToPrompt();
            }
        """)
        
        if api_format and "output" in api_format:
            with open("/home/mikeyb/Documents/AI/ComfyUI/comfyui_story_agent/skills/LTX2.3 video reasoning final.json", "w") as f:
                json.dump(api_format["output"], f, indent=2)
            print("Successfully converted to API format!")
        else:
            print("Conversion failed.")
            
        browser.close()

if __name__ == "__main__":
    run()
