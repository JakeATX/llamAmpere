#!/usr/bin/env python3
import pytest
import base64
import requests

from utils import *

server: ServerProcess


def get_test_image_base64() -> str:
    """Get a test image in base64 format"""
    # Use the same test image as test_vision_api.py
    IMG_URL = "https://huggingface.co/ggml-org-dev/tinygemma3-GGUF/resolve/main/test/11_truck.png"
    response = requests.get(IMG_URL)
    response.raise_for_status()
    return base64.b64encode(response.content).decode("utf-8")

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.model_alias = "tinyllama-2-anthropic"
    server.n_slots = 1
    server.n_ctx = 8192
    server.n_batch = 2048


@pytest.fixture
def vision_server():
    """Separate fixture for vision tests that require multimodal support"""
    global server
    server = ServerPreset.tinygemma3()
    server.offline = False  # Allow downloading the model
    server.model_alias = "tinygemma3-anthropic"
    server.n_slots = 1
    return server


# Basic message tests

def test_anthropic_messages_basic():
    """Test basic Anthropic messages endpoint"""
    server.start()

    res = server.make_request("POST", "/v1/messages", data={
        "model": "test",
        "max_tokens": 50,
        "messages": [
            {"role": "user", "content": "Say hello"}
        ]
    })

    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    assert res.body["type"] == "message", f"Expected type 'message', got {res.body.get('type')}"
    assert res.body["role"] == "assistant", f"Expected role 'assistant', got {res.body.get('role')}"
    assert "content" in res.body, "Missing 'content' field"
    assert isinstance(res.body["content"], list), "Content should be an array"
    assert len(res.body["content"]) > 0, "Content array should not be empty"
    assert res.body["content"][0]["type"] == "text", "First content block should be text"
    assert "text" in res.body["content"][0], "Text content block missing 'text' field"
    assert res.body["stop_reason"] in ["end_turn", "max_tokens"], f"Invalid stop_reason: {res.body.get('stop_reason')}"
    assert "usage" in res.body, "Missing 'usage' field"
    assert "cache_read_input_tokens" in res.body["usage"], "Missing usage.cache_read_input_tokens"
    assert "input_tokens" in res.body["usage"], "Missing usage.input_tokens"
    assert "output_tokens" in res.body["usage"], "Missing usage.output_tokens"
    assert isinstance(res.body["usage"]["cache_read_input_tokens"], int), "cache_read_input_tokens should be integer"
    assert isinstance(res.body["usage"]["input_tokens"], int), "input_tokens should be integer"
    assert isinstance(res.body["usage"]["output_tokens"], int), "output_tokens should be integer"
    assert res.body["usage"]["output_tokens"] > 0, "Should have generated some tokens"
    # Anthropic API should NOT include timings
    assert "timings" not in res.body, "Anthropic API should not include timings field"


def test_anthropic_messages_with_system():
    """Test messages with system prompt"""
    server.start()

    res = server.make_request("POST", "/v1/messages", data={
        "model": "test",
        "max_tokens": 50,
        "system": "You are a helpful assistant.",
        "messages": [
            {"role": "user", "content": "Hello"}
        ]
    })

    assert res.status_code == 200
    assert res.body["type"] == "message"
    assert len(res.body["content"]) > 0


def test_anthropic_messages_multipart_content():
    """Test messages with multipart content blocks"""
    server.start()

    res = server.make_request("POST", "/v1/messages", data={
        "model": "test",
        "max_tokens": 50,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is"},
                    {"type": "text", "text": " the answer?"}
                ]
            }
        ]
    })

    assert res.status_code == 200
    assert res.body["type"] == "message"


def test_anthropic_messages_conversation():
    """Test multi-turn conversation"""
    server.start()

    res = server.make_request("POST", "/v1/messages", data={
        "model": "test",
        "max_tokens": 50,
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
    })

    assert res.status_code == 200
    assert res.body["type"] == "message"


# Streaming tests

def test_anthropic_messages_streaming():
    """Test streaming messages"""
    server.start()

    res = server.make_stream_request("POST", "/v1/messages", data={
        "model": "test",
        "max_tokens": 30,
        "messages": [
            {"role": "user", "content": "Say hello"}
        ],
        "stream": True
    })

    events = []
    for data in res:
        # Each event should have type and other fields
        assert "type" in data, f"Missing 'type' in event: {data}"
        events.append(data)

    # Verify event sequence
    event_types = [e["type"] for e in events]
    assert "message_start" in event_types, "Missing message_start event"
    assert "content_block_start" in event_types, "Missing content_block_start event"
    assert "content_block_delta" in event_types, "Missing content_block_delta event"
    assert "content_block_stop" in event_types, "Missing content_block_stop event"
    assert "message_delta" in event_types, "Missing message_delta event"
    assert "message_stop" in event_types, "Missing message_stop event"

    # Check message_start structure
    message_start = next(e for e in events if e["type"] == "message_start")
    assert "message" in message_start, "message_start missing 'message' field"
    assert message_start["message"]["type"] == "message"
    assert message_start["message"]["role"] == "assistant"
    assert message_start["message"]["content"] == []
    assert "usage" in message_start["message"]
    assert message_start["message"]["usage"]["input_tokens"] > 0

    # Check content_block_start
    block_start = next(e for e in events if e["type"] == "content_block_start")
    assert "index" in block_start, "content_block_start missing 'index'"
    assert block_start["index"] == 0, "First content block should be at index 0"
    assert "content_block" in block_start
    assert block_start["content_block"]["type"] == "text"

    # Check content_block_delta
    deltas = [e for e in events if e["type"] == "content_block_delta"]
    assert len(deltas) > 0, "Should have at least one content_block_delta"
    for delta in deltas:
        assert "index" in delta
        assert "delta" in delta
        assert delta["delta"]["type"] == "text_delta"
        assert "text" in delta["delta"]

    # Check content_block_stop
    block_stop = next(e for e in events if e["type"] == "content_block_stop")
    assert "index" in block_stop
    assert block_stop["index"] == 0

    # Check message_delta
    message_delta = next(e for e in events if e["type"] == "message_delta")
    assert "delta" in message_delta
    assert "stop_reason" in message_delta["delta"]
    assert message_delta["delta"]["stop_reason"] in ["end_turn", "max_tokens"]
    assert "usage" in message_delta
    assert message_delta["usage"]["output_tokens"] > 0

    # Check message_stop
    message_stop = next(e for e in events if e["type"] == "message_stop")
    # message_stop should NOT have timings for Anthropic API
    assert "timings" not in message_stop, "Anthropic streaming should not include timings"


# Token counting tests

def test_anthropic_count_tokens():
    """Test token counting endpoint"""
    server.start()

    res = server.make_request("POST", "/v1/messages/count_tokens", data={
        "model": "test",
        "messages": [
            {"role": "user", "content": "Hello world"}
        ]
    })

    assert res.status_code == 200
    assert "input_tokens" in res.body
    assert isinstance(res.body["input_tokens"], int)
    assert res.body["input_tokens"] > 0
    # Should only have input_tokens, no other fields
    assert "output_tokens" not in res.body


def test_anthropic_count_tokens_with_system():
    """Test token counting with system prompt"""
    server.start()

    res = server.make_request("POST", "/v1/messages/count_tokens", data={
        "model": "test",
        "system": "You are a helpful assistant.",
        "messages": [
            {"role": "user", "content": "Hello"}
        ]
    })

    assert res.status_code == 200
    assert res.body["input_tokens"] > 0


def test_anthropic_count_tokens_no_max_tokens():
    """Test that count_tokens doesn't require max_tokens"""
    server.start()

    # max_tokens is NOT required for count_tokens
    res = server.make_request("POST", "/v1/messages/count_tokens", data={
        "model": "test",
        "messages": [
            {"role": "user", "content": "Hello"}
        ]
    })

    assert res.status_code == 200
    assert "input_tokens" in res.body


# Tool use tests

def test_anthropic_tool_use_basic():
    """Test basic tool use"""
    server.jinja = True
    server.start()

    res = server.make_request("POST", "/v1/messages", data={
        "model": "test",
        "max_tokens": 200,
        "tools": [{
            "name": "get_weather",
            "description": "Get the current weather in a location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name"
                    }
                },
                "required": ["location"]
            }
        }],
        "messages": [
            {"role": "user", "content": "What's the weather in Paris?"}
        ]
    })

    assert res.status_code == 200
    assert res.body["type"] == "message"
    assert len(res.body["content"]) > 0

    # Check if model used the tool (it might not always, depending on the model)
    content_types = [block.get("type") for block in res.body["content"]]

    if "tool_use" in content_types:
        # Model used the tool
        assert res.body["stop_reason"] == "tool_use"

        # Find the tool_use block
        tool_block = next(b for b in res.body["content"] if b.get("type") == "tool_use")
        assert "id" in tool_block
        assert "name" in tool_block
        assert tool_block["name"] == "get_weather"
        assert "input" in tool_block
        assert isinstance(tool_block["input"], dict)


def test_anthropic_tool_result():
    """Test sending tool results back

    This test verifies that tool_result blocks are properly converted to
    role="tool" messages internally. Without proper conversion, this would
    fail with a 500 error: "unsupported content[].type" because tool_result
    blocks would remain in the user message content array.
    """
    server.jinja = True
    server.start()

    res = server.make_request("POST", "/v1/messages", data={
        "model": "test",
        "max_tokens": 100,
        "messages": [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "test123",
                        "name": "get_weather",
                        "input": {"location": "Paris"}
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "test123",
                        "content": "The weather is sunny, 25°C"
                    }
                ]
            }
        ]
    })

    # This would be 500 with the old bug where tool_result blocks weren't converted
    assert res.status_code == 200
    assert res.body["type"] == "message"
    # Model should respond to the tool result
    assert len(res.body["content"]) > 0
    assert res.body["content"][0]["type"] == "text"


def test_anthropic_tool_result_with_text():
    """Test tool result mixed with text content

    This tests the edge case where a user message contains both text and
    tool_result blocks. The server must properly split these into separate
    messages: a user message with text, followed by tool messages.
    Without proper handling, this would fail with 500: "unsupported content[].type"
    """
    server.jinja = True
    server.start()

    res = server.make_request("POST", "/v1/messages", data={
        "model": "test",
        "max_tokens": 100,
        "messages": [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool_1",
                        "name": "get_weather",
                        "input": {"location": "Paris"}
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Here are the results:"},
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool_1",
                        "content": "Sunny, 25°C"
                    }
                ]
            }
        ]
    })

    assert res.status_code == 200
    assert res.body["type"] == "message"
    assert len(res.body["content"]) > 0


def test_anthropic_tool_result_with_image():
    """Test tool result containing mixed text and image blocks

    Verifies that image blocks inside Anthropic tool_result content are
    properly converted to OpenAI image_url format rather than being
    silently dropped. With a non-multimodal model, the converted image
    triggers a clear error message instead of being ignored.
    """
    server.jinja = True
    server.start()

    # Small 1x1 red PNG image in base64 (same as vision tests)
    red_pixel_png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="

    res = server.make_request("POST", "/v1/messages", data={
        "model": "test",
        "max_tokens": 100,
        "messages": [
            {"role": "user", "content": "What is in this image?"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool_1",
                        "name": "read",
                        "input": {"file": "test.png"}
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool_1",
                        "content": [
                            {"type": "text", "text": "File: test.png"},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": red_pixel_png
                                }
                            }
                        ]
                    }
                ]
            }
        ]
    })

    # Without the fix, image block would cause "unsupported content[].type"
    # With the fix, image is converted to image_url but tinyllama doesn't support images
    assert res.status_code == 500
    assert "image input is not supported" in res.body.get("error", {}).get("message", "").lower()


def test_anthropic_tool_result_error():
    """Test tool result with error flag"""
    server.jinja = True
    server.start()

    res = server.make_request("POST", "/v1/messages", data={
        "model": "test",
        "max_tokens": 100,
        "messages": [
            {"role": "user", "content": "Get the weather"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "test123",
                        "name": "get_weather",
                        "input": {"location": "InvalidCity"}
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "test123",
                        "is_error": True,
                        "content": "City not found"
                    }
                ]
            }
        ]
    })

    assert res.status_code == 200
    assert res.body["type"] == "message"


def test_anthropic_tool_reference():
    """Anthropic defer_loading and tool_reference convert to their OpenAI forms (behaviour: test_tool_call.py)"""
    server.jinja = True
    server.chat_template_file = '../../../models/templates/LFM3.jinja'
    server.anthropic_thinking = "template"  # compares against the OpenAI endpoint: same thinking default
    server.start()

    search = {"name": "tool_search", "description": "Search for available tools by keyword",
              "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}
    purchase = {"name": "markItemPurchased", "description": "Mark an item as purchased",
                "parameters": {"type": "object", "properties": {"id": {"type": "string"}}}}

    anthropic = server.make_request("POST", "/v1/messages/count_tokens", data={
        "model": "test",
        "tools": [
            {"name": t["name"], "description": t["description"], "input_schema": t["parameters"], **extra}
            for t, extra in ((search, {}), (purchase, {"defer_loading": True}))
        ],
        "messages": [
            {"role": "user", "content": "Mark item abc as purchased"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "call_1", "name": "tool_search", "input": {"query": "purchase"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "call_1",
                 "content": [{"type": "tool_reference", "tool_name": "markItemPurchased"}]},
            ]},
        ],
    })
    oai = server.make_request("POST", "/chat/completions/input_tokens", data={
        "tools": [
            {"type": "function", "function": search},
            {"type": "function", "function": {**purchase, "defer_loading": True}},
        ],
        "messages": [
            {"role": "user", "content": "Mark item abc as purchased"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "tool_search", "arguments": "{\"query\": \"purchase\"}"}},
            ]},
            {"role": "tool", "tool_call_id": "call_1",
             "content": [{"type": "tool_reference", "name": "markItemPurchased"}]},
        ],
    })
    assert anthropic.status_code == 200, anthropic.body
    assert oai.status_code == 200, oai.body
    assert anthropic.body["input_tokens"] == oai.body["input_tokens"]


def test_anthropic_tool_streaming():
    """Test streaming with tool use"""
    server.jinja = True
    server.start()

    res = server.make_stream_request("POST", "/v1/messages", data={
        "model": "test",
        "max_tokens": 200,
        "stream": True,
        "tools": [{
            "name": "calculator",
            "description": "Calculate math",
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"}
                },
                "required": ["expression"]
            }
        }],
        "messages": [
            {"role": "user", "content": "Calculate 2+2"}
        ]
    })

    events = []
    for data in res:
        events.append(data)

    event_types = [e["type"] for e in events]

    # Should have basic events
    assert "message_start" in event_types
    assert "message_stop" in event_types

    # If tool was used, check for proper tool streaming
    if any(e.get("type") == "content_block_start" and
           e.get("content_block", {}).get("type") == "tool_use"
           for e in events):
        # Find tool use block start
        tool_starts = [e for e in events if
                      e.get("type") == "content_block_start" and
                      e.get("content_block", {}).get("type") == "tool_use"]

        assert len(tool_starts) > 0, "Should have tool_use content_block_start"

        # Check index is correct (should be 0 if no text, 1 if there's text)
        tool_start = tool_starts[0]
        assert "index" in tool_start
        assert tool_start["content_block"]["type"] == "tool_use"
        assert "name" in tool_start["content_block"]


# Vision/multimodal tests

def test_anthropic_vision_format_accepted():
    """Test that Anthropic vision format is accepted (format validation only)"""
    server.start()

    # Small 1x1 red PNG image in base64
    red_pixel_png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="

    res = server.make_request("POST", "/v1/messages", data={
        "model": "test",
        "max_tokens": 10,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": red_pixel_png
                        }
                    },
                    {
                        "type": "text",
                        "text": "What is this?"
                    }
                ]
            }
        ]
    })

    # Server accepts the format but tinyllama doesn't support images
    # So it should return 500 with clear error message about missing mmproj
    assert res.status_code == 500
    assert "image input is not supported" in res.body.get("error", {}).get("message", "").lower()


def test_anthropic_vision_base64_with_multimodal_model(vision_server):
    """Test vision with base64 image using Anthropic format with multimodal model"""
    global server
    server = vision_server
    server.start()

    # Get test image in base64 format
    image_base64 = get_test_image_base64()

    res = server.make_request("POST", "/v1/messages", data={
        "model": "test",
        "max_tokens": 10,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_base64
                        }
                    },
                    {
                        "type": "text",
                        "text": "What is this:\n"
                    }
                ]
            }
        ]
    })

    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.body}"
    assert res.body["type"] == "message"
    assert len(res.body["content"]) > 0
    assert res.body["content"][0]["type"] == "text"
    # The model should generate some response about the image
    assert len(res.body["content"][0]["text"]) > 0


# Parameter tests

def test_anthropic_stop_sequences():
    """Test stop_sequences parameter"""
    server.start()

    res = server.make_request("POST", "/v1/messages", data={
        "model": "test",
        "max_tokens": 100,
        "stop_sequences": ["\n", "END"],
        "messages": [
            {"role": "user", "content": "Count to 10"}
        ]
    })

    assert res.status_code == 200
    assert res.body["type"] == "message"


def test_anthropic_temperature():
    """Test temperature parameter"""
    server.start()

    res = server.make_request("POST", "/v1/messages", data={
        "model": "test",
        "max_tokens": 50,
        "temperature": 0.5,
        "messages": [
            {"role": "user", "content": "Hello"}
        ]
    })

    assert res.status_code == 200
    assert res.body["type"] == "message"


def test_anthropic_top_p():
    """Test top_p parameter"""
    server.start()

    res = server.make_request("POST", "/v1/messages", data={
        "model": "test",
        "max_tokens": 50,
        "top_p": 0.9,
        "messages": [
            {"role": "user", "content": "Hello"}
        ]
    })

    assert res.status_code == 200
    assert res.body["type"] == "message"


def test_anthropic_top_k():
    """Test top_k parameter (llama.cpp specific)"""
    server.start()

    res = server.make_request("POST", "/v1/messages", data={
        "model": "test",
        "max_tokens": 50,
        "top_k": 40,
        "messages": [
            {"role": "user", "content": "Hello"}
        ]
    })

    assert res.status_code == 200
    assert res.body["type"] == "message"


# Error handling tests

def test_anthropic_missing_messages():
    """Test error when messages are missing"""
    server.start()

    res = server.make_request("POST", "/v1/messages", data={
        "model": "test",
        "max_tokens": 50
        # missing "messages" field
    })

    # Should return an error (400 or 500)
    assert res.status_code >= 400


def test_anthropic_empty_messages():
    """Test permissive handling of empty messages array"""
    server.start()

    res = server.make_request("POST", "/v1/messages", data={
        "model": "test",
        "max_tokens": 50,
        "messages": []
    })

    # Server is permissive and accepts empty messages (provides defaults)
    # This matches the permissive validation design choice
    assert res.status_code == 200
    assert res.body["type"] == "message"


# Content block index tests

def test_anthropic_streaming_content_block_indices():
    """Test that content block indices are correct in streaming"""
    server.jinja = True
    server.start()

    # Request that might produce both text and tool use
    res = server.make_stream_request("POST", "/v1/messages", data={
        "model": "test",
        "max_tokens": 400,
        "stream": True,
        "tools": [{
            "name": "test_tool",
            "description": "A test tool",
            "input_schema": {
                "type": "object",
                "properties": {
                    "param": {"type": "string"}
                },
                "required": ["param"]
            }
        }],
        "messages": [
            {"role": "user", "content": "Use the test tool"}
        ]
    })

    events = []
    for data in res:
        events.append(data)

    # Check content_block_start events have sequential indices
    block_starts = [e for e in events if e.get("type") == "content_block_start"]
    if len(block_starts) > 1:
        # If there are multiple blocks, indices should be sequential
        indices = [e["index"] for e in block_starts]
        expected_indices = list(range(len(block_starts)))
        assert indices == expected_indices, f"Expected indices {expected_indices}, got {indices}"

    # Check content_block_stop events match the starts
    block_stops = [e for e in events if e.get("type") == "content_block_stop"]
    start_indices = set(e["index"] for e in block_starts)
    stop_indices = set(e["index"] for e in block_stops)
    assert start_indices == stop_indices, "content_block_stop indices should match content_block_start indices"


# Extended features tests

def test_anthropic_thinking():
    """Test extended thinking parameter"""
    server.jinja = True
    server.start()

    res = server.make_request("POST", "/v1/messages", data={
        "model": "test",
        "max_tokens": 100,
        "thinking": {
            "type": "enabled",
            "budget_tokens": 50
        },
        "messages": [
            {"role": "user", "content": "What is 2+2?"}
        ]
    })

    assert res.status_code == 200
    assert res.body["type"] == "message"


def _anthropic_input_tokens(extra):
    body = {
        "model": "test",
        "max_tokens": 1,
        "temperature": 0,
        "messages": [{"role": "user", "content": "Write a short title for this session."}],
    }
    body.update(extra)
    res = server.make_request("POST", "/v1/messages", data=body)
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.body}"
    usage = res.body["usage"]
    # input_tokens counts only the uncached part; a repeated prompt is mostly read from the cache
    return usage["input_tokens"] + usage.get("cache_read_input_tokens", 0)


def test_anthropic_think_only_when_asked():
    """--anthropic-thinking on-request (default): no "thinking" (or type "disabled") renders the template with
    thinking off (Qwen3 adds an empty think block to the generation prompt); "enabled"/"adaptive" keep thinking on;
    an explicit chat_template_kwargs.enable_thinking from the client wins."""
    global server
    server.jinja = True
    server.chat_template_file = '../../../models/templates/Qwen-Qwen3-0.6B.jinja'
    server.start()

    n_none     = _anthropic_input_tokens({})
    n_disabled = _anthropic_input_tokens({"thinking": {"type": "disabled"}})
    n_enabled  = _anthropic_input_tokens({"thinking": {"type": "enabled", "budget_tokens": 64}})
    n_adaptive = _anthropic_input_tokens({"thinking": {"type": "adaptive"}})
    n_kwargs   = _anthropic_input_tokens({"chat_template_kwargs": {"enable_thinking": True}})

    assert n_none == n_disabled, (n_none, n_disabled)
    assert n_enabled == n_adaptive == n_kwargs, (n_enabled, n_adaptive, n_kwargs)
    # thinking off adds "<think>\n\n</think>\n\n" to the generation prompt
    assert n_none > n_enabled, (n_none, n_enabled)

    # count_tokens uses the same conversion
    def count(extra):
        body = {"model": "test", "messages": [{"role": "user", "content": "Write a short title for this session."}]}
        body.update(extra)
        res = server.make_request("POST", "/v1/messages/count_tokens", data=body)
        assert res.status_code == 200, res.body
        return res.body["input_tokens"]
    assert count({}) > count({"thinking": {"type": "enabled", "budget_tokens": 64}})


def test_anthropic_thinking_operator_flag_wins():
    """--reasoning on from the operator keeps thinking on for requests without "thinking"."""
    global server
    server.jinja = True
    server.chat_template_file = '../../../models/templates/Qwen-Qwen3-0.6B.jinja'
    server.reasoning = "on"
    server.start()

    n_none    = _anthropic_input_tokens({})
    n_enabled = _anthropic_input_tokens({"thinking": {"type": "enabled", "budget_tokens": 64}})
    assert n_none == n_enabled, (n_none, n_enabled)


def test_anthropic_thinking_template_mode():
    """--anthropic-thinking template: a request without "thinking" keeps the template default (thinking on)."""
    global server
    server.jinja = True
    server.chat_template_file = '../../../models/templates/Qwen-Qwen3-0.6B.jinja'
    server.anthropic_thinking = "template"
    server.start()

    n_none    = _anthropic_input_tokens({})
    n_enabled = _anthropic_input_tokens({"thinking": {"type": "enabled", "budget_tokens": 64}})
    assert n_none == n_enabled, (n_none, n_enabled)


QWEN38_TEMPLATE = '../../../models/templates/Qwen-Qwen3.8-27B.jinja'


def _claude_code_request(effort="medium", thinking=True, model="atx-swift-27b-8gb", turns=1):
    """Shaped like a Claude Code 2.1.327 /v1/messages request (captured 2026-11-19): system blocks with the billing
    header, a system message *inside* messages after the first user turn (the environment block), adaptive thinking
    with display omitted, output_config effort, context_management, metadata and tools with a $schema. Text is
    shortened; the structure is unchanged."""
    body = {
        "model": model,
        "max_tokens": 8,
        "temperature": 1,
        "stream": False,
        "system": [
            {"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.327.8c9; cc_entrypoint=sdk-cli; cch=a5145;"},
            {"type": "text", "text": "You are a coding agent.", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "\nUse the tools to help the user.", "cache_control": {"type": "ephemeral"}},
        ],
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "<system-reminder>\nKeep commits short.\n</system-reminder>\n"},
                {"type": "text", "text": "say hi"},
            ]},
            {"role": "system", "content": [
                {"type": "text", "text": "# Environment\n - Primary working directory: /work\n - Platform: linux\n\nToday's date is 2026-11-19.",
                 "cache_control": {"type": "ephemeral"}},
            ]},
        ],
        "tools": [
            {"name": "Bash", "description": "Executes a bash command and returns its output.",
             "input_schema": {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object",
                              "properties": {"command": {"type": "string", "description": "The command to execute"},
                                             "timeout": {"type": "number", "description": "Timeout in ms"}},
                              "required": ["command"], "additionalProperties": False}},
            {"name": "Read", "description": "Reads a file from the local filesystem.",
             "input_schema": {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object",
                              "properties": {"file_path": {"type": "string"}}, "required": ["file_path"],
                              "additionalProperties": False}},
        ],
        "context_management": {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]},
        "metadata": {"user_id": "{\"device_id\":\"0\",\"account_uuid\":\"\",\"session_id\":\"0\"}"},
    }
    if thinking:
        body["thinking"] = {"type": "adaptive", "display": "omitted"}
    if effort is not None:
        body["output_config"] = {"effort": effort}
    for i in range(1, turns):
        body["messages"].append({"role": "assistant", "content": [{"type": "text", "text": f"Hi! (turn {i})"}]})
        body["messages"].append({"role": "user", "content": [{"type": "text", "text": "list the files"}]})
    return body


def _cc_count(body):
    b = dict(body)
    for k in ("max_tokens", "stream", "temperature"):
        b.pop(k, None)
    res = server.make_request("POST", "/v1/messages/count_tokens", data=b)
    assert res.status_code == 200, res.body
    return res.body["input_tokens"]


def test_anthropic_claude_code_session_qwen38():
    """A Claude Code session against the Qwen3.8 template: the system message inside "messages" is hoisted (the
    template raises "System message must be at the beginning" otherwise), effort "high" is taken (the template knows
    low/medium/xhigh), and thinking is on by default (adaptive)."""
    global server
    server.jinja = True
    server.chat_template_file = QWEN38_TEMPLATE
    server.n_ctx = 8192
    server.start()

    for turns in (1, 2, 3):
        for effort in ("medium", "high"):
            res = server.make_request("POST", "/v1/messages", data=_claude_code_request(effort=effort, turns=turns))
            assert res.status_code == 200, f"turns={turns} effort={effort}: {res.body}"
            assert res.body["type"] == "message"

    # streaming, as Claude Code sends it
    body = _claude_code_request(effort="high", turns=2)
    body["stream"] = True
    events = [e for e in server.make_stream_request("POST", "/v1/messages", data=body)]
    assert any(e.get("type") == "message_stop" for e in events), events[-3:]

    # thinking on by default: adaptive renders "<think>\n", no thinking renders the empty think block
    n_adaptive = _cc_count(_claude_code_request(effort="medium"))
    n_nothink = _cc_count(_claude_code_request(effort="medium", thinking=False))
    assert n_nothink > n_adaptive, (n_nothink, n_adaptive)

    # effort: high and max take the xhigh level, minimal takes low, medium stays medium (no instruction line)
    n = {e: _cc_count(_claude_code_request(effort=e)) for e in ("minimal", "low", "medium", "high", "xhigh", "max")}
    assert n["high"] == n["xhigh"] == n["max"], n
    assert n["minimal"] == n["low"], n
    assert n["medium"] < n["low"] < n["xhigh"], n


def test_anthropic_inline_system_hoisted_in_order():
    """Inline system messages before the first assistant turn join the leading system message in order, after the
    "system" parameter; a later one becomes a <system-reminder> user turn in place (the earlier prefix stays cached)."""
    global server
    server.jinja = True
    server.chat_template_file = QWEN38_TEMPLATE
    server.start()

    base = {"model": "test", "system": "S0",
            "messages": [{"role": "user", "content": "hi"}, {"role": "system", "content": "S1"},
                         {"role": "system", "content": [{"type": "text", "text": "S2"}]},
                         {"role": "assistant", "content": "hello"}, {"role": "system", "content": "S3"},
                         {"role": "user", "content": "again"}]}
    flat = {"model": "test", "system": "S0\n\nS1\n\nS2",
            "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"},
                         {"role": "user", "content": "<system-reminder>\nS3\n</system-reminder>"},
                         {"role": "user", "content": "again"}]}
    assert _cc_count(base) == _cc_count(flat)
    # no "system" parameter: the inline ones become the leading system message
    only_inline = {"model": "test", "messages": [{"role": "user", "content": "hi"}, {"role": "system", "content": "S1"}]}
    assert _cc_count(only_inline) == _cc_count({"model": "test", "system": "S1", "messages": [{"role": "user", "content": "hi"}]})
    # the order is kept: swapping S1 and S2 changes nothing in the count, so check the hoisted text renders at all
    long_inline = {"model": "test", "messages": [{"role": "user", "content": "hi"},
                                                 {"role": "system", "content": "a much longer environment block " * 8}]}
    assert _cc_count(long_inline) > _cc_count(only_inline)


def test_reasoning_effort_clamped_to_template_levels():
    """common/chat: an effort the template does not take goes to the nearest level it takes (ties upward); a word
    outside the scale falls back to the template default, never an error."""
    global server
    server.jinja = True
    server.chat_template_file = QWEN38_TEMPLATE
    server.start()

    def render(effort):
        res = server.make_request("POST", "/apply-template", data={
            "messages": [{"role": "user", "content": "hi"}],
            "chat_template_kwargs": {"reasoning_effort": effort},
        })
        assert res.status_code == 200, f"{effort}: {res.body}"
        return res.body["prompt"]

    assert "effort is set to xhigh" in render("high")
    assert "effort is set to xhigh" in render("max")
    assert "effort is set to low" in render("minimal")
    assert "effort is set to" not in render("medium")
    assert "effort is set to xhigh" in render("banana")   # template default


def test_anthropic_metadata():
    """Test metadata parameter"""
    server.start()

    res = server.make_request("POST", "/v1/messages", data={
        "model": "test",
        "max_tokens": 50,
        "metadata": {
            "user_id": "test_user_123"
        },
        "messages": [
            {"role": "user", "content": "Hello"}
        ]
    })

    assert res.status_code == 200
    assert res.body["type"] == "message"


# Compatibility tests

def test_anthropic_vs_openai_different_response_format():
    """Verify Anthropic format is different from OpenAI format"""
    server.start()

    # Make OpenAI request
    openai_res = server.make_request("POST", "/v1/chat/completions", data={
        "model": "test",
        "max_tokens": 50,
        "messages": [
            {"role": "user", "content": "Hello"}
        ]
    })

    # Make Anthropic request
    anthropic_res = server.make_request("POST", "/v1/messages", data={
        "model": "test",
        "max_tokens": 50,
        "messages": [
            {"role": "user", "content": "Hello"}
        ]
    })

    assert openai_res.status_code == 200
    assert anthropic_res.status_code == 200

    # OpenAI has "object", Anthropic has "type"
    assert "object" in openai_res.body
    assert "type" in anthropic_res.body
    assert openai_res.body["object"] == "chat.completion"
    assert anthropic_res.body["type"] == "message"

    # OpenAI has "choices", Anthropic has "content"
    assert "choices" in openai_res.body
    assert "content" in anthropic_res.body

    # Different usage field names
    assert "prompt_tokens" in openai_res.body["usage"]
    assert "input_tokens" in anthropic_res.body["usage"]
    assert "completion_tokens" in openai_res.body["usage"]
    assert "output_tokens" in anthropic_res.body["usage"]


# Extended thinking tests with reasoning models

# The next two tests cover the input path (conversation history):
# Client sends thinking blocks -> convert_anthropic_to_oai -> reasoning_content -> template

def test_anthropic_thinking_history_in_count_tokens():
    """Test that interleaved thinking blocks in conversation history are not dropped during conversion."""
    global server
    server.jinja = True
    server.chat_template_file = '../../../models/templates/Qwen-Qwen3-0.6B.jinja'
    server.start()

    tool = {
        "name": "list_files",
        "description": "List files",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"]
        }
    }

    messages_without_thinking = [
        {"role": "user", "content": "Fix the bug"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "call_1", "name": "list_files", "input": {"path": "."}}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "call_1", "content": "main.py"}
            ]
        },
    ]

    messages_with_thinking = [
        {"role": "user", "content": "Fix the bug"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "I should check the project structure first to understand the codebase layout."},
                {"type": "tool_use", "id": "call_1", "name": "list_files", "input": {"path": "."}}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "call_1", "content": "main.py"}
            ]
        },
    ]

    res_without = server.make_request("POST", "/v1/messages/count_tokens", data={
        "model": "test",
        "messages": messages_without_thinking,
        "tools": [tool],
    })
    assert res_without.status_code == 200, f"Expected 200: {res_without.body}"

    res_with = server.make_request("POST", "/v1/messages/count_tokens", data={
        "model": "test",
        "messages": messages_with_thinking,
        "tools": [tool],
    })
    assert res_with.status_code == 200, f"Expected 200: {res_with.body}"

    # Thinking blocks should increase the token count
    assert res_with.body["input_tokens"] > res_without.body["input_tokens"], \
        f"Expected more tokens with thinking ({res_with.body['input_tokens']}) than without ({res_without.body['input_tokens']})"


def test_anthropic_thinking_history_in_template():
    """Test that reasoning_content from converted interleaved thinking blocks renders in the prompt."""
    global server
    server.jinja = True
    server.chat_template_file = '../../../models/templates/Qwen-Qwen3-0.6B.jinja'
    server.start()

    reasoning_1 = "I should check the project structure first."
    reasoning_2 = "Now I need to read the main file."

    res = server.make_request("POST", "/apply-template", data={
        "messages": [
            {"role": "user", "content": "Fix the bug in main.py"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": reasoning_1,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "list_files", "arguments": "{\"path\": \".\"}"}
                }]
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "main.py\nutils.py"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": reasoning_2,
                "tool_calls": [{
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{\"path\": \"main.py\"}"}
                }]
            },
            {"role": "tool", "tool_call_id": "call_2", "content": "print('hello')"},
        ],
        "tools": [{
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List files",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
            }
        }, {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
            }
        }],
    })
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.body}"
    prompt = res.body["prompt"]

    # Both reasoning_content values should be rendered in <think> tags
    assert reasoning_1 in prompt, f"Expected first reasoning text in prompt: {prompt}"
    assert reasoning_2 in prompt, f"Expected second reasoning text in prompt: {prompt}"
    assert prompt.count("<think>") >= 2, f"Expected at least 2 <think> blocks in prompt: {prompt}"


@pytest.mark.slow
@pytest.mark.parametrize("stream", [False, True])
def test_anthropic_thinking_with_reasoning_model(stream):
    """Test that thinking content blocks are properly returned for reasoning models"""
    global server
    server = ServerProcess()
    server.model_hf_repo = "bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF"
    server.model_hf_file = "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf"
    server.reasoning_format = "deepseek"
    server.jinja = True
    server.n_ctx = 8192
    server.n_predict = 1024
    server.start(timeout_seconds=600)  # large model needs time to download

    if stream:
        res = server.make_stream_request("POST", "/v1/messages", data={
            "model": "test",
            "max_tokens": 1024,
            "thinking": {
                "type": "enabled",
                "budget_tokens": 500
            },
            "messages": [
                {"role": "user", "content": "What is 2+2?"}
            ],
            "stream": True
        })

        events = list(res)

        # should have thinking content block events
        thinking_starts = [e for e in events if
            e.get("type") == "content_block_start" and
            e.get("content_block", {}).get("type") == "thinking"]
        assert len(thinking_starts) > 0, "Should have thinking content_block_start event"
        assert thinking_starts[0]["index"] == 0, "Thinking block should be at index 0"

        # should have thinking_delta events
        thinking_deltas = [e for e in events if
            e.get("type") == "content_block_delta" and
            e.get("delta", {}).get("type") == "thinking_delta"]
        assert len(thinking_deltas) > 0, "Should have thinking_delta events"

        # should have signature_delta event before thinking block closes (Anthropic API requirement)
        signature_deltas = [e for e in events if
            e.get("type") == "content_block_delta" and
            e.get("delta", {}).get("type") == "signature_delta"]
        assert len(signature_deltas) > 0, "Should have signature_delta event for thinking block"

        # should have text block after thinking
        text_starts = [e for e in events if
            e.get("type") == "content_block_start" and
            e.get("content_block", {}).get("type") == "text"]
        assert len(text_starts) > 0, "Should have text content_block_start event"
        assert text_starts[0]["index"] == 1, "Text block should be at index 1 (after thinking)"
    else:
        res = server.make_request("POST", "/v1/messages", data={
            "model": "test",
            "max_tokens": 1024,
            "thinking": {
                "type": "enabled",
                "budget_tokens": 500
            },
            "messages": [
                {"role": "user", "content": "What is 2+2?"}
            ]
        })

        assert res.status_code == 200
        assert res.body["type"] == "message"

        content = res.body["content"]
        assert len(content) >= 2, "Should have at least thinking and text blocks"

        # first block should be thinking
        thinking_blocks = [b for b in content if b.get("type") == "thinking"]
        assert len(thinking_blocks) > 0, "Should have thinking content block"
        assert "thinking" in thinking_blocks[0], "Thinking block should have 'thinking' field"
        assert len(thinking_blocks[0]["thinking"]) > 0, "Thinking content should not be empty"
        assert "signature" in thinking_blocks[0], "Thinking block should have 'signature' field (Anthropic API requirement)"

        # should also have text block
        text_blocks = [b for b in content if b.get("type") == "text"]
        assert len(text_blocks) > 0, "Should have text content block"


# Output config / reasoning effort tests

def test_anthropic_output_config_effort():
    """Test that output_config.effort is accepted"""
    server.start()

    res = server.make_request("POST", "/v1/messages", data={
        "model": "test",
        "max_tokens": 50,
        "output_config": {"effort": "low"},
        "messages": [
            {"role": "user", "content": "Hello"}
        ]
    })

    assert res.status_code == 200
    assert res.body["type"] == "message"


def test_anthropic_output_config_effort_xhigh():
    """Test that non-standard effort values are forwarded unchanged"""
    server.start()

    res = server.make_request("POST", "/v1/messages", data={
        "model": "test",
        "max_tokens": 50,
        "output_config": {"effort": "xhigh"},
        "messages": [
            {"role": "user", "content": "Hello"}
        ]
    })

    assert res.status_code == 200
    assert res.body["type"] == "message"


def test_anthropic_output_config_effort_preserves_kwargs():
    """Test that existing chat_template_kwargs are preserved alongside output_config.effort"""
    server.start()

    res = server.make_request("POST", "/v1/messages", data={
        "model": "test",
        "max_tokens": 50,
        "output_config": {"effort": "high"},
        "chat_template_kwargs": {"some_key": "some_value"},
        "messages": [
            {"role": "user", "content": "Hello"}
        ]
    })

    assert res.status_code == 200
    assert res.body["type"] == "message"


def test_anthropic_output_config_effort_reaches_template():
    """Test that output_config.effort reaches the Jinja template as reasoning_effort."""
    global server
    server.jinja = True
    server.chat_template_file = '../../../models/templates/upstage-Solar-Open-100B.jinja'
    server.start()

    res_without = server.make_request("POST", "/v1/messages/count_tokens", data={
        "model": "test",
        "messages": [
            {"role": "user", "content": "Hello"}
        ]
    })
    assert res_without.status_code == 200

    res_with = server.make_request("POST", "/v1/messages/count_tokens", data={
        "model": "test",
        "output_config": {"effort": "low"},
        "messages": [
            {"role": "user", "content": "Hello"}
        ]
    })
    assert res_with.status_code == 200

    # Solar template adds an empty think block when reasoning_effort is "low",
    # so the token count should increase
    assert res_with.body["input_tokens"] > res_without.body["input_tokens"], \
        f"Expected more tokens with effort=low ({res_with.body['input_tokens']}) than without ({res_without.body['input_tokens']})"


def test_anthropic_output_config_effort_precedence():
    """Test that output_config.effort takes precedence over chat_template_kwargs.reasoning_effort."""
    global server
    server.jinja = True
    server.chat_template_file = '../../../models/templates/upstage-Solar-Open-100B.jinja'
    server.start()

    # Baseline: reasoning_effort="high" via chat_template_kwargs (no extra think block)
    res_high = server.make_request("POST", "/v1/messages/count_tokens", data={
        "model": "test",
        "chat_template_kwargs": {"reasoning_effort": "high"},
        "messages": [
            {"role": "user", "content": "Hello"}
        ]
    })
    assert res_high.status_code == 200

    # output_config.effort="low" should override chat_template_kwargs
    res_override = server.make_request("POST", "/v1/messages/count_tokens", data={
        "model": "test",
        "output_config": {"effort": "low"},
        "chat_template_kwargs": {"reasoning_effort": "high"},
        "messages": [
            {"role": "user", "content": "Hello"}
        ]
    })
    assert res_override.status_code == 200

    # If output_config.effort takes precedence, "low" adds the extra think block
    assert res_override.body["input_tokens"] > res_high.body["input_tokens"], \
        f"Expected output_config.effort='low' to override chat_template_kwargs.reasoning_effort='high'"
