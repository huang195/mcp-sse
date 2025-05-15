import asyncio
import json
import os
import requests
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession
from mcp.client.sse import sse_client

from dotenv import load_dotenv

OLLAMA_ENDPOINT = "http://localhost:11434/v1/chat/completions"
MODEL = "qwen3:1.7b"

def wrap_mcp_tool_for_openai(tool) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema,
        }
    }

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

    async def connect_to_sse_server(self, server_url: str):
        """Connect to an MCP server running with SSE transport"""
        # Store the context managers so they stay alive
        self._streams_context = sse_client(url=server_url)
        streams = await self._streams_context.__aenter__()

        self._session_context = ClientSession(*streams)
        self.session: ClientSession = await self._session_context.__aenter__()

        # Initialize
        await self.session.initialize()

        # List available tools to verify connection
        print("Initialized SSE client...")
        print("Listing tools...")
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def cleanup(self):
        """Properly clean up the session and streams"""
        if self._session_context:
            await self._session_context.__aexit__(None, None, None)
        if self._streams_context:
            await self._streams_context.__aexit__(None, None, None)

    async def process_query(self, query: str) -> str:
        """Process a query using Ollama and available tools"""
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        response = await self.session.list_tools()
        available_tools = [wrap_mcp_tool_for_openai(t) for t in response.tools]


        # Process response and handle tool calls
        final_text = []

        while True:

            print(f"messages: {messages}")

            # Ask Ollama with tool definitions
            response = requests.post(
                OLLAMA_ENDPOINT,
                headers={"Content-Type": "application/json"},
                json={
                    "model": MODEL,
                    "max_tokens": 1000,
                    "messages": messages,
                    "tools": available_tools,
                    "tool_choice": "auto"
                }
            )

            message = response.json()["choices"][0]["message"]

            print(f"llm response: {message}")

            # Capture regular assistant reply
            if "content" in message:
                final_text.append(message["content"])
                if len(message["content"]) > 0:
                    break

            if "tool_calls" in message:
                # Handle tool calls
                for tool_call in message["tool_calls"]:
                    tool_name = tool_call["function"]["name"]
                    tool_args = json.loads(tool_call["function"]["arguments"])

                    # Call the tool via your session
                    result = await self.session.call_tool(tool_name, tool_args)

                    print(f"tool response: {result}")

                    messages.append({
                        "role": "assistant",
                        "tool_calls": [tool_call]
                    })

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": tool_name,
                        "content": result.content[0].text  # must be a string
                    })

        return "\n".join(final_text) 

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                
                if query.lower() == 'quit':
                    break
                    
                response = await self.process_query(query)
                print("\n" + response)
                    
            except Exception as e:
                print(f"\nError: {str(e)}")


async def main():
    if len(sys.argv) < 2:
        print("Usage: uv run client.py <URL of SSE MCP server (i.e. http://localhost:8080/sse)>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_sse_server(server_url=sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    import sys
    asyncio.run(main())
