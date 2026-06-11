import json
import threading
import time
import urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"
STATUS_URL = "http://127.0.0.1:8000/v1/status"

def send_request(model="chat", prompt="Write a single sentence about software engineering.", stream=False):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream
    }
    req = urllib.request.Request(
        URL, 
        data=json.dumps(payload).encode("utf-8"), 
        headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req) as res:
            if stream:
                text = ""
                for line in res:
                    line_str = line.decode("utf-8").strip()
                    if line_str.startswith("data: "):
                        data_payload = line_str[6:]
                        if data_payload == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_payload)
                            delta = chunk["choices"][0]["delta"].get("content", "")
                            text += delta
                        except Exception:
                            pass
                duration = time.time() - t0
                return text, duration
            else:
                resp_data = json.loads(res.read().decode("utf-8"))
                text = resp_data["choices"][0]["message"]["content"]
                duration = time.time() - t0
                return text, duration
    except Exception as e:
        print(f"Request failed: {e}")
        return None, 0.0

def get_status():
    try:
        with urllib.request.urlopen(STATUS_URL) as res:
            return json.loads(res.read().decode("utf-8"))
    except Exception as e:
        print(f"Status check failed: {e}")
        return None

print("=== VERIFICATION OF PERSISTENT EMBEDDED ENGINE ===")

# Check initial status
status = get_status()
print(f"Active model: {status.get('active_model')}")
print(f"Registered models: {status.get('registered_models')}")
print(f"Residency: {status.get('residency')}")

# 1. Warm request test (chat model was pre-warmed)
print("\n--- Test 1: Warm request (should be fast, ~1-2s) ---")
text, duration = send_request(model="chat", prompt="Name three fruits.")
print(f"Latency: {duration:.4f}s")
print(f"Response: {text.strip()}")

# 2. Consecutive request test (should be even faster because model is resident)
print("\n--- Test 2: Consecutive request (resident check) ---")
text, duration = send_request(model="chat", prompt="What is 2 + 2?")
print(f"Latency: {duration:.4f}s")
print(f"Response: {text.strip()}")

# 3. Model Swap request test (chat -> reasoning)
print("\n--- Test 3: Model Swap (chat -> reasoning) ---")
text, duration = send_request(model="reasoning", prompt="What is logic in one sentence?")
print(f"Latency: {duration:.4f}s")
print(f"Response: {text.strip()}")

# Check status after swap
status = get_status()
print(f"Active model after swap: {status.get('active_model')}")

# 4. Swap back (reasoning -> chat)
print("\n--- Test 4: Swap Back (reasoning -> chat) ---")
text, duration = send_request(model="chat", prompt="What is biology in one sentence?")
print(f"Latency: {duration:.4f}s")
print(f"Response: {text.strip()}")

# 5. Concurrency lock test
print("\n--- Test 5: Concurrency Protection (asyncio.Lock) ---")
results = []
def worker(name, prompt):
    print(f"Worker {name} starting...")
    txt, dur = send_request(model="chat", prompt=prompt)
    print(f"Worker {name} finished in {dur:.2f}s")
    results.append((name, dur))

t1 = threading.Thread(target=worker, args=("A", "Say hello."))
t2 = threading.Thread(target=worker, args=("B", "Say goodbye."))

t1.start()
t2.start()
t1.join()
t2.join()

print("Concurrency execution completed.")
print("=== VERIFICATION COMPLETED ===")
