"""The OCR endpoints ask Mistral without blocking the event loop of the instance (every agent and chat it serves), and
never wait for it longer than a timeout.

Run from the core root: ``python -m pytest cat/plugins/cc_mistral_ocr/tests``.

The Cat imports every ``.py`` file of the plugin, tests included: at import time this module needs only the stdlib,
the Cat, httpx and the plugin are loaded in ``setUpClass`` (and in the fakes, when they run).
"""
import asyncio
import base64
import json
import os
import sys
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

PLUGIN_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOADED = set()

MODULE = None
OCR = OCR_PDF = None
OCRInput = OCRPDFInput = None


def _load():
    """Loads the plugin once, from the test classes: the Cat imports the modules of the plugin (tests included) again
    when it loads it, so ``sys.modules`` (where unittest looks for ``setUpModule``) may hold another copy of this module:
    the globals are set here, in the module of the test classes."""
    global MODULE, OCR, OCR_PDF, OCRInput, OCRPDFInput
    if PLUGIN_PATH in LOADED:
        return
    from cat.looking_glass.mad_hatter.plugin import Plugin

    plugin = Plugin(PLUGIN_PATH)
    plugin._load_decorated_functions()
    OCR = next(e.function for e in plugin.endpoints if e.path == "/ocr")
    OCR_PDF = next(e.function for e in plugin.endpoints if e.path == "/ocr-pdf")
    MODULE = sys.modules[OCR.__module__]
    OCRInput, OCRPDFInput = MODULE.OCRInput, MODULE.OCRPDFInput
    LOADED.add(PLUGIN_PATH)


PAGES = {"pages": [{"markdown": "# page 1"}, {"markdown": "# page 2"}]}


class Mistral:
    """api.mistral.ai, behind a real httpx client (``httpx.MockTransport``): the upload of a file, its signed URL, and
    the OCR. ``delay`` is the time it takes to answer, ``status`` the HTTP status of the OCR, ``unreachable`` the paths
    that do not answer, ``invalid_json`` whether the OCR answers something that is not JSON."""

    def __init__(self, delay=0.0):
        self.delay = delay
        self.status = 200
        self.unreachable = set()
        self.invalid_json = False
        self.per_agent = False
        self.requests = []

    async def handle(self, request):
        import httpx

        self.requests.append(request)
        await asyncio.sleep(self.delay)
        path = request.url.path
        if path in self.unreachable:
            raise httpx.ConnectError("injected: connection refused", request=request)
        if path == "/v1/files":
            return httpx.Response(200, json={"id": "file-1"})
        if path == "/v1/files/file-1/url":
            return httpx.Response(200, json={"url": "https://signed/file-1"})
        if self.invalid_json:
            return httpx.Response(200, text="<html>not json</html>")
        if self.per_agent:
            # the pages of the document of the agent that asks
            key = request.headers["Authorization"].removeprefix("Bearer key of ")
            return httpx.Response(self.status, json={"pages": [{"markdown": f"# page {i} of {key}"} for i in (1, 2)]})
        return httpx.Response(self.status, json=PAGES)

    def client(self):
        import httpx

        return httpx.AsyncClient(transport=httpx.MockTransport(self.handle), timeout=MODULE.TIMEOUT_SECONDS)


def _info(save=False, agent="agent"):
    info = MagicMock()
    info.cheshire_cat.agent_key = agent
    info.cheshire_cat.mad_hatter.get_plugin.return_value.load_settings = AsyncMock(
        return_value={"mistral_api_key": f"key of {agent}", "save_text_to_rabbit_hole": save}
    )
    info.lizard.rabbit_hole.ingest_file = AsyncMock()
    return info


def _image():
    return OCRInput(image="aW1n", type="image/png", tags=[{"name": "kind", "value": "invoice"}])


def _pdf():
    return OCRPDFInput(pdf=base64.b64encode(b"%PDF-1.4").decode(), filename="doc", tags=[{"name": "k", "value": ["a"]}])


def _run(coro_factory, server):
    async def main():
        with patch.object(MODULE, "_client", server.client):
            return await coro_factory()

    return asyncio.run(main())


class InTemporaryDirectory(unittest.TestCase):
    """The tests run in a temporary working directory: the endpoints must leave it empty."""

    def setUp(self):
        self.cwd = os.getcwd()
        self.tmp = tempfile.TemporaryDirectory()
        os.chdir(self.tmp.name)

    def tearDown(self):
        os.chdir(self.cwd)
        self.tmp.cleanup()


def _ingested(info):
    """(filename, content) of every page ingested: the file must be in memory (a BytesIO), never a local path."""
    import io

    ingested = []
    for call in info.lizard.rabbit_hole.ingest_file.await_args_list:
        file = call.kwargs["file"]
        assert isinstance(file, io.BytesIO), f"a local file was ingested: {file!r}"
        ingested.append((call.kwargs["filename"], file.getvalue().decode("utf-8")))
    return ingested


class TestOCR(InTemporaryDirectory):
    @classmethod
    def setUpClass(cls):
        _load()

    def test_an_image(self):
        server = Mistral()
        self.assertEqual(_run(lambda: OCR(_image(), _info()), server), PAGES)
        (request,) = server.requests
        self.assertEqual((request.method, str(request.url)), ("POST", "https://api.mistral.ai/v1/ocr"))
        self.assertEqual(request.headers["Authorization"], "Bearer key of agent")
        self.assertEqual(json.loads(request.content)["document"], {
            "type": "image_url", "image_url": "data:image/png;base64,aW1n",
        })

    def test_the_pages_of_an_image_are_ingested(self):
        info = _info(save=True)
        _run(lambda: OCR(_image(), info), Mistral())
        calls = info.lizard.rabbit_hole.ingest_file.await_args_list
        self.assertEqual(len(calls), 2)
        for call in calls:
            self.assertIs(call.kwargs["cat"], info.cheshire_cat)
            self.assertEqual(call.kwargs["metadata"], {"kind": "invoice"})
            self.assertEqual(call.kwargs["content_type"], "text/markdown")
        self.assertEqual(_ingested(info), [("ocrpage.md", "# page 1"), ("ocrpage.md", "# page 2")])
        self.assertEqual(os.listdir("."), [])

    def test_concurrent_agents_ingest_their_own_pages(self):
        """Regression: every page was written to the same local file (``ocrpage.md``, in the working directory of the
        instance) and ingested from there: two agents asking at the same time overwrote each other's page, and the text
        of an agent could be ingested in the memory of another one."""
        server = Mistral()
        server.per_agent = True
        infos = [_info(save=True, agent=f"agent {i}") for i in range(3)]
        for info in infos:
            async def slow_ingest(*args, **kwargs):
                # the rabbit hole reads the file after a while (parsing, embedding): the other agents go on
                await asyncio.sleep(0.01)
                from pathlib import Path

                file = kwargs["file"]
                kwargs["read"] = file.getvalue() if hasattr(file, "getvalue") else Path(file).read_bytes()

            info.lizard.rabbit_hole.ingest_file.side_effect = slow_ingest

        async def burst():
            with patch.object(MODULE, "_client", server.client):
                await asyncio.gather(*(OCR(_image(), info) for info in infos))

        asyncio.run(burst())
        for i, info in enumerate(infos):
            self.assertEqual([content for _, content in _ingested(info)], [f"# page 1 of agent {i}", f"# page 2 of agent {i}"])
        self.assertEqual(os.listdir("."), [])

    def test_the_endpoints_declare_what_they_return(self):
        """Regression: annotated ``-> str``, they return the answer of Mistral (a dict): FastAPI validates the response
        against the annotation."""
        import typing

        for function in (OCR, OCR_PDF):
            self.assertIs(typing.get_type_hints(function)["return"], dict, function.__name__)

    def test_an_error_of_mistral_is_raised(self):
        import httpx

        server = Mistral()
        server.status = 401
        with self.assertRaises(httpx.HTTPStatusError):
            _run(lambda: OCR(_image(), _info()), server)
        server = Mistral()
        server.unreachable.add("/v1/ocr")
        with self.assertRaises(httpx.ConnectError):
            _run(lambda: OCR(_image(), _info()), server)

    def test_an_answer_that_is_not_json(self):
        from cat.exceptions import CustomValidationException

        server = Mistral()
        server.invalid_json = True
        with self.assertRaises(CustomValidationException):
            _run(lambda: OCR(_image(), _info()), server)

    def test_an_error_of_the_ingestion_is_raised(self):
        info = _info(save=True)
        info.lizard.rabbit_hole.ingest_file.side_effect = RuntimeError("injected")
        with self.assertRaises(RuntimeError):
            _run(lambda: OCR(_image(), info), Mistral())


class TestOCRPDF(InTemporaryDirectory):
    @classmethod
    def setUpClass(cls):
        _load()

    def test_a_pdf_is_uploaded_and_read(self):
        server = Mistral()
        self.assertEqual(_run(lambda: OCR_PDF(_pdf(), _info()), server), PAGES)
        upload, url, ocr = server.requests
        self.assertEqual((upload.method, str(upload.url)), ("POST", "https://api.mistral.ai/v1/files"))
        self.assertIn(b'name="purpose"\r\n\r\nocr', upload.content)
        self.assertIn(b"%PDF-1.4", upload.content)
        self.assertIn(b'filename="doc.pdf"', upload.content)
        self.assertEqual(upload.headers["Authorization"], "Bearer key of agent")
        self.assertEqual((url.method, str(url.url)), ("GET", "https://api.mistral.ai/v1/files/file-1/url?expiry=24"))
        self.assertEqual(json.loads(ocr.content)["document"], {
            "type": "document_url", "document_url": "https://signed/file-1",
        })

    def test_the_pages_of_a_pdf_are_ingested(self):
        info = _info(save=True)
        _run(lambda: OCR_PDF(_pdf(), info), Mistral())
        self.assertEqual(_ingested(info), [("doc_0.md", "# page 1"), ("doc_1.md", "# page 2")])
        for call in info.lizard.rabbit_hole.ingest_file.await_args_list:
            self.assertIs(call.kwargs["cat"], info.cheshire_cat)
            self.assertEqual(call.kwargs["metadata"], {"k": ["a"]})
        self.assertEqual(os.listdir("."), [])

    def test_the_filename_of_the_user_is_only_a_name(self):
        """Regression: the pages were written to ``f"{filename}_{i}.md"`` with the filename chosen by the user: a path
        (``../../x``) wrote the file anywhere the Cat can write. The filename is now only the (sanitized) name of the
        pages ingested from memory, and of the file stored by the rabbit hole."""
        cases = {
            "../../etc/cron.d/evil": "evil",
            "/abs/path/Report 2024.v2.pdf": "Report_2024_v2",
            "..": "document",
            "": "document",
            "a" * 200: "a" * 80,
        }
        for filename, name in cases.items():
            with self.subTest(filename=filename):
                info = _info(save=True)
                payload = OCRPDFInput(pdf=base64.b64encode(b"%PDF-1.4").decode(), filename=filename, tags=[])
                _run(lambda: OCR_PDF(payload, info), Mistral())
                self.assertEqual([f for f, _ in _ingested(info)], [f"{name}_0.md", f"{name}_1.md"])
                self.assertEqual(os.listdir("."), [])

    def test_no_file_is_written(self):
        """The PDF is uploaded, and the pages ingested, from memory: nothing is written on the disk of the instance."""
        info = _info(save=True)
        with patch("tempfile.NamedTemporaryFile", side_effect=AssertionError("a local file")), \
                patch("pathlib.Path.write_text", side_effect=AssertionError("a local file")), \
                patch("pathlib.Path.write_bytes", side_effect=AssertionError("a local file")):
            self.assertEqual(_run(lambda: OCR_PDF(_pdf(), info), Mistral()), PAGES)
            self.assertEqual(_run(lambda: OCR(_image(), info), Mistral()), PAGES)
        self.assertEqual(len(_ingested(info)), 4)

    def test_an_error_of_mistral_is_raised(self):
        import httpx

        for path in ("/v1/files", "/v1/files/file-1/url", "/v1/ocr"):
            with self.subTest(path=path):
                server = Mistral()
                server.unreachable.add(path)
                with self.assertRaises(httpx.ConnectError):
                    _run(lambda: OCR_PDF(_pdf(), _info()), server)
        server = Mistral()
        server.status = 500
        with self.assertRaises(httpx.HTTPStatusError):
            _run(lambda: OCR_PDF(_pdf(), _info()), server)


class TestNonBlocking(InTemporaryDirectory):
    @classmethod
    def setUpClass(cls):
        _load()

    def test_a_slow_mistral_never_blocks_the_other_requests(self):
        """Regression: the requests were synchronous (``requests.post``/``get``) inside the asynchronous endpoints:
        while Mistral answered, the event loop was blocked, and so every other request served by the instance."""
        server = Mistral(delay=0.1)

        async def burst():
            with patch.object(MODULE, "_client", server.client):
                begin = time.monotonic()
                await asyncio.gather(*(
                    endpoint(payload, _info(agent=f"agent {i}"))
                    for i in range(3) for endpoint, payload in ((OCR, _image()), (OCR_PDF, _pdf()))
                ))
                return time.monotonic() - begin

        # an image takes one request of 0.1 seconds, a PDF three: the six in a row take 1.2 seconds
        self.assertLess(asyncio.run(burst()), 0.8, "the six OCRs run together")
        self.assertEqual(len(server.requests), 12)
        self.assertEqual(
            {r.headers["Authorization"] for r in server.requests}, {f"Bearer key of agent {i}" for i in range(3)}
        )

    def test_the_client_has_a_timeout(self):
        client = MODULE._client()
        self.assertEqual(client.timeout.read, MODULE.TIMEOUT_SECONDS)
        self.assertEqual(client.timeout.connect, MODULE.TIMEOUT_SECONDS)
        asyncio.run(client.aclose())


class TestPluginSafety(unittest.TestCase):
    def test_every_file_passes_the_scanner_of_the_core(self):
        from cat.looking_glass.mad_hatter.plugin_extractor import PluginExtractor

        self.assertTrue(PluginExtractor._is_safe_plugin(PLUGIN_PATH))

    def test_tests_import_only_the_stdlib_at_import_time(self):
        import ast
        from pathlib import Path

        for test_file in sorted(Path(__file__).parent.glob("*.py")):
            tree = ast.parse(test_file.read_text(encoding="utf-8"))
            for node in tree.body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                    for name in names:
                        self.assertIn(name.split(".")[0], sys.stdlib_module_names, (test_file.name, name))


if __name__ == "__main__":
    unittest.main()
