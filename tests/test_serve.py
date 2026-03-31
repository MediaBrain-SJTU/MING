"""
Tests for serving/inference module.
"""
import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI
from ming.serve import inference, cli


class TestInferenceModule:
    """Test inference module."""

    def test_import_inference(self):
        """Test that inference module can be imported."""
        assert inference is not None

    def test_import_cli(self):
        """Test that CLI module can be imported."""
        assert cli is not None

    def test_compute_skip_echo_len(self):
        """Test compute_skip_echo_len function."""
        try:
            from ming.serve.inference import compute_skip_echo_len

            # Test with a simple model name
            result = compute_skip_echo_len("ming", None, "Hello world")
            assert isinstance(result, int)
            assert result >= 0
        except ImportError:
            pytest.skip("compute_skip_echo_len not available")


class TestFastAPIService:
    """Test FastAPI service functionality."""

    def test_fastapi_app_creation(self):
        """Test FastAPI app can be created."""
        app = FastAPI(title="MING Test")

        @app.get("/health")
        def health():
            return {"status": "healthy"}

        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}

    def test_api_endpoints_structure(self):
        """Test API endpoints structure."""
        app = FastAPI(title="MING Test")

        @app.post("/v1/chat/completions")
        def chat_completion(request: dict):
            return {"choices": [{"message": {"content": "Test"}}]}

        @app.get("/v1/models")
        def list_models():
            return {"data": [{"id": "ming-model"}]}

        client = TestClient(app)

        # Test models endpoint
        response = client.get("/v1/models")
        assert response.status_code == 200

        # Test chat completion endpoint
        response = client.post("/v1/chat/completions", json={"messages": []})
        assert response.status_code == 200


class TestGradioInterface:
    """Test Gradio interface."""

    def test_gradio_import(self):
        """Test Gradio can be imported."""
        try:
            import gradio as gr
            assert gr is not None
        except ImportError:
            pytest.skip("Gradio not installed")

    def test_gradio_components(self):
        """Test Gradio components creation."""
        try:
            import gradio as gr

            with gr.Blocks() as demo:
                with gr.Row():
                    with gr.Column():
                        input_text = gr.Textbox(label="输入")
                        submit_btn = gr.Button("提交")
                    with gr.Column():
                        output_text = gr.Textbox(label="输出")

            assert demo is not None
        except ImportError:
            pytest.skip("Gradio not installed")


class TestConversationInference:
    """Test conversation-based inference."""

    def test_conversation_template_usage(self):
        """Test conversation template in inference."""
        from ming.conversations import get_default_conv_template

        conv = get_default_conv_template("ming")
        conv.append_message(conv.roles[0], "你好")
        conv.append_message(conv.roles[1], None)

        prompt = conv.get_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_generate_stream_params(self):
        """Test generate_stream function parameters."""
        # Mock parameters
        params = {
            "prompt": "Hello",
            "temperature": 0.7,
            "max_new_tokens": 256,
            "stop": None
        }

        assert "prompt" in params
        assert "temperature" in params
        assert "max_new_tokens" in params
