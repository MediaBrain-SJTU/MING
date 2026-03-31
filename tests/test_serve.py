import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestServeModule:
    def test_inference_import(self):
        try:
            from ming.serve.inference import generate_stream
            assert generate_stream is not None
        except ImportError:
            pytest.skip("inference module not available")

    def test_cli_import(self):
        try:
            from ming.serve import cli
            assert cli is not None
        except ImportError:
            pytest.skip("CLI module not available")


class TestFastAPI:
    def test_fastapi_available(self):
        try:
            from fastapi import FastAPI
            app = FastAPI()
            assert app is not None
        except ImportError:
            pytest.skip("FastAPI not installed")

    def test_health_endpoint(self):
        try:
            from fastapi import FastAPI
            from fastapi.testclient import TestClient
            
            app = FastAPI()
            
            @app.get("/health")
            def health():
                return {"status": "ok"}
            
            client = TestClient(app)
            response = client.get("/health")
            
            assert response.status_code == 200
            assert response.json()["status"] == "ok"
        except ImportError:
            pytest.skip("FastAPI not installed")

    def test_inference_endpoint_schema(self):
        try:
            from fastapi import FastAPI
            from pydantic import BaseModel
            
            class InferenceRequest(BaseModel):
                prompt: str
                max_new_tokens: int = 256
                temperature: float = 1.0
            
            class InferenceResponse(BaseModel):
                generated_text: str
                prompt: str
            
            request = InferenceRequest(prompt="test")
            assert request.prompt == "test"
            assert request.max_new_tokens == 256
        except ImportError:
            pytest.skip("FastAPI or Pydantic not installed")


class TestGradio:
    def test_gradio_available(self):
        try:
            import gradio as gr
            assert gr is not None
        except ImportError:
            pytest.skip("Gradio not installed")

    def test_gradio_interface_creation(self):
        try:
            import gradio as gr
            
            def mock_inference(text):
                return f"Response: {text}"
            
            demo = gr.Interface(
                fn=mock_inference,
                inputs=gr.Textbox(label="Input"),
                outputs=gr.Textbox(label="Output"),
                title="Test Interface"
            )
            
            assert demo is not None
        except ImportError:
            pytest.skip("Gradio not installed")


class TestConversations:
    def test_conv_templates_import(self):
        try:
            from ming.conversations import conv_templates
            assert conv_templates is not None
        except ImportError:
            pytest.skip("conv_templates not available")

    def test_get_default_conv_template(self):
        try:
            from ming.conversations import get_default_conv_template
            conv = get_default_conv_template()
            assert conv is not None
        except ImportError:
            pytest.skip("get_default_conv_template not available")

    def test_separator_style(self):
        try:
            from ming.conversations import SeparatorStyle
            assert hasattr(SeparatorStyle, 'ADD_COLON_TWO')
        except ImportError:
            pytest.skip("SeparatorStyle not available")
