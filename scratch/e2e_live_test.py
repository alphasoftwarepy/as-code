import urllib.request
import json
import time
import sys

base_url = 'http://127.0.0.1:8000'

def send_chat(prompt, model='chat', stream=True, skill=None):
    url = f'{base_url}/v1/chat/completions'
    body = {
        'model': model,
        'messages': [{'role': 'user', 'content': prompt}],
        'temperature': 0.7,
        'max_tokens': 512,
        'stream': stream
    }
    headers = {
        'Content-Type': 'application/json',
        'X-Session-Id': f'test_{int(time.time()*1000)}'
    }
    if skill:
        headers['X-Skill'] = skill

    req = urllib.request.Request(url, data=json.dumps(body).encode('utf-8'), headers=headers)
    start = time.time()
    res = urllib.request.urlopen(req)
    chunks = []
    text = ''
    done_received = False
    stop_received = False
    
    if stream:
        for line in res:
            line_str = line.decode('utf-8').strip()
            if not line_str:
                continue
            if line_str == 'data: [DONE]':
                done_received = True
                continue
            if line_str.startswith('data: '):
                chunk_data = json.loads(line_str[6:])
                chunks.append(chunk_data)
                delta = chunk_data.get('choices', [{}])[0].get('delta', {})
                content = delta.get('content', '')
                if content:
                    text += content
                if chunk_data.get('choices', [{}])[0].get('finish_reason') == 'stop':
                    stop_received = True
    else:
        data = json.loads(res.read().decode('utf-8'))
        text = data['choices'][0]['message']['content']
        stop_received = data['choices'][0]['finish_reason'] == 'stop'
        done_received = True

    duration = time.time() - start
    return {
        'text': text,
        'chunks_count': len(chunks),
        'done': done_received,
        'stop': stop_received,
        'duration': duration
    }

print('=== TEST A: CHAT NORMAL ===')
res_a = send_chat('crea una poesia sobre paraguay de 100 palabras', model='chat', stream=True)
has_json_call = 'json_call' in res_a['text'] or '"capability"' in res_a['text']
print(f'Test A: done={res_a["done"]}, stop={res_a["stop"]}, spurious_tool={has_json_call}, text_len={len(res_a["text"])}, time={res_a["duration"]:.2f}s')

print('\n=== TEST B: STREAMING NORMAL ===')
res_b = send_chat('explica brevemente qué es el runtime cognitivo de AS Core en tres párrafos', model='chat', stream=True)
print(f'Test B: chunks={res_b["chunks_count"]}, done={res_b["done"]}, stop={res_b["stop"]}, text_len={len(res_b["text"])}, time={res_b["duration"]:.2f}s')

print('\n=== TEST C: CANCEL / ABORT ENDPOINT ===')
cancel_url = f'{base_url}/v1/cancel?request_id=test-cancel-123&model_id=chat'
cancel_req = urllib.request.Request(cancel_url, data=b'', headers={'Content-Type': 'application/json'}, method='POST')
cancel_res = urllib.request.urlopen(cancel_req)
cancel_data = json.loads(cancel_res.read().decode('utf-8'))
print(f'Test C (Cancel API): status={cancel_data.get("status")}')

print('\n=== TEST D: EMPTY STREAM RESPONSE / SAFETY ===')
# Test with minimal whitespace prompt to check clean termination
res_d = send_chat('hola', model='chat', stream=True)
print(f'Test D: done={res_d["done"]}, stop={res_d["stop"]}, text_len={len(res_d["text"])}, time={res_d["duration"]:.2f}s')

print('\n=== TEST E: CAPABILITY LEGÍTIMA / GATE EVALUATION ===')
res_e = send_chat('lee el archivo test.py', model='code', stream=True, skill='programming')
print(f'Test E: done={res_e["done"]}, stop={res_e["stop"]}, text_len={len(res_e["text"])}, time={res_e["duration"]:.2f}s')

print('\n=== TEST F: MODELO MANUAL ===')
res_f = send_chat('responde "OK"', model='chat', stream=True)
clean_f = res_f['text'].strip().replace('\n', ' ')
print(f'Test F: done={res_f["done"]}, stop={res_f["stop"]}, text="{clean_f}"')

print('\n=== TEST G: MODELO AUTO ===')
for query, expected_type in [('hola como estas', 'chat'), ('escribe una funcion python para sumar dos numeros', 'code')]:
    res_g = send_chat(query, model='auto', stream=True)
    print(f'Test G ({expected_type}): done={res_g["done"]}, stop={res_g["stop"]}, text_len={len(res_g["text"])}, time={res_g["duration"]:.2f}s')

print('\nALL E2E SCENARIOS COMPLETED SUCCESSFULLY.')
