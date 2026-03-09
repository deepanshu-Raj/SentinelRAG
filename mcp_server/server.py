from mcp_server.tools import search_hybrid

TOOLS = {
    "search_hybrid": search_hybrid,
}


def call_tool(tool_name: str, **kwargs):
    if tool_name not in TOOLS:
        raise ValueError(f"Unknow tool {tool_name}")

    return TOOLS[tool_name](**kwargs)
