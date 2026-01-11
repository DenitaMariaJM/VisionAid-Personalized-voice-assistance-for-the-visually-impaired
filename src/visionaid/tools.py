
"""
Tool definitions and local execution for realtime tool-calling.
All tool schemas are OpenAI-compatible; only one schema is maintained.
"""


import json
from typing import Any, Dict


from .memory import build_memory_entry, clear_memory, list_memory, search_memory, store_memory
from .vision import analyze_image, capture_image


TOOLS_OPENAI = [
    {
        "name": "capture_image",
        "description": "Capture an image from the camera and return its path.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "analyze_image",
        "description": "Analyze a captured image for obstacles and navigation details.",
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": {"type": "string"},
                "query": {"type": "string"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_memory",
        "description": "Search semantic memory for relevant past items.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "default": 2},
            },
            "required": ["query"],
        },
    },
    {
        "name": "store_memory",
        "description": "Store a memory entry for future retrieval.",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "clear_memory",
        "description": "Clear all stored semantic memory entries.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "list_memory",
        "description": "List recent memory entries.",
        "parameters": {
            "type": "object",
            "properties": {"limit": {"type": "integer", "default": 3}},
            "required": [],
        },
    },
]



def _tool_capture_image() -> Dict[str, Any]:
    """Capture an image and return the file path."""
    path = capture_image()
    if not path:
        return {"ok": False, "error": "Camera capture failed."}
    return {"ok": True, "image_path": path}



def _tool_analyze_image(image_path: str, query: str) -> Dict[str, Any]:
    """Analyze an image for obstacles and navigation details."""
    if not image_path:
        image_path = capture_image()
    if not image_path:
        return {"ok": False, "error": "Camera capture failed."}
    analysis = analyze_image(image_path, query)
    if not analysis:
        return {"ok": False, "error": "Vision analysis failed.", "image_path": image_path}
    return {"ok": True, "image_path": image_path, "analysis": analysis}



def _tool_search_memory(query: str, k: int) -> Dict[str, Any]:
    """Search semantic memory for relevant past items."""
    results = search_memory(query, k=k)
    return {"ok": True, "results": results}



def _tool_store_memory(text: str) -> Dict[str, Any]:
    """Store a memory entry for future retrieval."""
    entry = build_memory_entry(text)
    store_memory(entry)
    return {"ok": True, "stored": entry}



def _tool_clear_memory() -> Dict[str, Any]:
    """Clear all stored semantic memory entries."""
    clear_memory()
    return {"ok": True, "cleared": True}



def _tool_list_memory(limit: int) -> Dict[str, Any]:
    """List recent memory entries."""
    items = list_memory(limit=limit)
    return {"ok": True, "items": items}



def execute_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch tool execution by name."""
    if name == "capture_image":
        return _tool_capture_image()
    if name == "analyze_image":
        return _tool_analyze_image(args.get("image_path"), args.get("query", ""))
    if name == "search_memory":
        return _tool_search_memory(args.get("query", ""), int(args.get("k", 2)))
    if name == "store_memory":
        return _tool_store_memory(args.get("text", ""))
    if name == "clear_memory":
        return _tool_clear_memory()
    if name == "list_memory":
        return _tool_list_memory(int(args.get("limit", 3)))
    return {"ok": False, "error": f"Unknown tool: {name}"}



def serialize_tool_result(result: Any) -> str:
    """Serialize tool result to JSON string."""
    return json.dumps(result)
