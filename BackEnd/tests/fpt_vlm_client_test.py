import json

from BackEnd.app.vqa import FPTVLMClient


class FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {"answer": "There are three people.", "confidence": 0.9}
                        )
                    }
                }
            ]
        }


class FakeHTTPClient:
    def __init__(self):
        self.url = ""
        self.payload = None
        self.authorization = ""

    def post(self, url, **kwargs):
        self.url = url
        self.payload = kwargs["json"]
        self.authorization = kwargs["headers"]["Authorization"]
        return FakeResponse()


def test_fpt_client_sends_images_and_parses_answer(tmp_path):
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"tiny-test-image")
    http = FakeHTTPClient()
    client = FPTVLMClient(
        api_key="test-secret",
        model="test-vlm",
        http_client=http,
    )

    answer = client.answer(
        question="How many people are visible?",
        prompt="Use visible evidence only.",
        image_paths=[image],
    )

    assert answer.answer == "There are three people."
    assert answer.confidence == 0.9
    assert http.url == "https://mkp-api.fptcloud.com/chat/completions"
    assert http.authorization == "Bearer test-secret"
    content = http.payload["messages"][1]["content"]
    assert content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert content[-1] == {"type": "text", "text": "How many people are visible?"}


def test_fpt_client_rejects_unsupported_image_type(tmp_path):
    image = tmp_path / "frame.gif"
    image.write_bytes(b"gif")
    client = FPTVLMClient(
        api_key="test-secret",
        model="test-vlm",
        http_client=FakeHTTPClient(),
    )
    try:
        client.answer(question="?", prompt="", image_paths=[image])
    except ValueError as error:
        assert "Unsupported" in str(error)
    else:
        raise AssertionError("Unsupported image type must fail before API call")
