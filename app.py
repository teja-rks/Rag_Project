from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn
from src.api.routes import router

app = FastAPI(title="RAG Chatbot API", version="1.0.0")
app.include_router(router, prefix="/api")

HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Minimal RAG Chat</title>
    <style>
        :root {
            --bg-color: #0f172a;
            --panel-bg: #1e293b;
            --accent: #3b82f6;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --border: #334155;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            display: flex;
            justify-content: center;
            height: 100vh;
            padding: 20px;
        }
        .chat-container {
            width: 100%;
            max-width: 760px;
            display: flex;
            flex-direction: column;
            background: var(--panel-bg);
            border-radius: 12px;
            border: 1px solid var(--border);
            overflow: hidden;
        }
        .header {
            padding: 16px 20px;
            background: #111827;
            border-bottom: 1px solid var(--border);
            font-size: 1.1rem;
            font-weight: 600;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .status-dot {
            height: 8px; width: 8px;
            background-color: #10b981;
            border-radius: 50%;
            display: inline-block;
            margin-right: 6px;
        }
        .messages {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        .msg {
            max-width: 80%;
            padding: 12px 16px;
            border-radius: 10px;
            line-height: 1.5;
            font-size: 0.95rem;
            word-break: break-word;
        }
        .msg.user {
            align-self: flex-end;
            background: var(--accent);
            color: #fff;
        }
        .msg.bot {
            align-self: flex-start;
            background: #334155;
            color: var(--text-main);
        }
        .input-area {
            display: flex;
            padding: 14px;
            background: #111827;
            border-top: 1px solid var(--border);
            gap: 10px;
        }
        input {
            flex: 1;
            padding: 12px 16px;
            background: #1e293b;
            border: 1px solid var(--border);
            border-radius: 8px;
            color: #fff;
            font-size: 0.95rem;
            outline: none;
        }
        input:focus { border-color: var(--accent); }
        button {
            padding: 12px 20px;
            background: var(--accent);
            border: none;
            border-radius: 8px;
            color: #fff;
            font-weight: 600;
            cursor: pointer;
            transition: opacity 0.2s;
        }
        button:hover { opacity: 0.9; }
        button:disabled { background: #64748b; cursor: not-allowed; }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="header">
            <span>RAG Assistant</span>
            <span style="font-size: 0.85rem; color: var(--text-muted);"><span class="status-dot"></span>Llama 3 Local</span>
        </div>
        <div class="messages" id="chatBox">
            <div class="msg bot">Hello! Ask me any question grounded in your ingested documents.</div>
        </div>
        <div class="input-area">
            <input type="text" id="userInput" placeholder="Ask a question..." onkeydown="if(event.key==='Enter') sendQuery()" />
            <button id="sendBtn" onclick="sendQuery()">Send</button>
        </div>
    </div>

    <script>
        async function sendQuery() {
            const input = document.getElementById("userInput");
            const btn = document.getElementById("sendBtn");
            const chatBox = document.getElementById("chatBox");
            const question = input.value.trim();

            if (!question) return;

            // Render User Message
            const userDiv = document.createElement("div");
            userDiv.className = "msg user";
            userDiv.innerText = question;
            chatBox.appendChild(userDiv);
            input.value = "";
            input.disabled = true;
            btn.disabled = true;
            chatBox.scrollTop = chatBox.scrollHeight;

            // Render Loading Placeholder
            const botDiv = document.createElement("div");
            botDiv.className = "msg bot";
            botDiv.innerText = "Thinking...";
            chatBox.appendChild(botDiv);
            chatBox.scrollTop = chatBox.scrollHeight;

            try {
                const res = await fetch("/api/query", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ question: question, top_k: 2 })
                });

                if (!res.ok) throw new Error("API request failed");
                const data = await res.json();
                botDiv.innerText = data.answer;
            } catch (err) {
                botDiv.innerText = "⚠️ Error fetching response. Ensure Ollama is running.";
            } finally {
                input.disabled = false;
                btn.disabled = false;
                input.focus();
                chatBox.scrollTop = chatBox.scrollHeight;
            }
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return HTML_CONTENT

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)