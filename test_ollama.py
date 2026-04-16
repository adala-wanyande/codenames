import ollama

def test_local_llm():
    print("🤖 Sending prompt to local Qwen model via Ollama...")
    
    # We ask it a Codenames-style question
    response = ollama.chat(model='qwen2.5', messages=[
        {
            'role': 'system',
            'content': 'You are a helpful AI assistant playing a word association game.'
        },
        {
            'role': 'user',
            'content': 'Give me one single word that connects "ocean" and "pool".'
        }
    ])
    
    print("\n✅ Response from Qwen:")
    print(response['message']['content'])

if __name__ == "__main__":
    test_local_llm()