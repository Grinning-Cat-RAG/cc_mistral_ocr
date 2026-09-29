import re
from io import BytesIO
from pathlib import PurePosixPath

from cat.auth.connection import AuthorizedInfo
from cat.exceptions import CustomValidationException
from cat.log import log
from typing import List
from cat import endpoint, check_permissions, AuthPermission, AuthResource
from pydantic import BaseModel
import json
import base64
import httpx

# the OCR of a long PDF takes a while, but a request never waits for Mistral longer than this
TIMEOUT_SECONDS = 120.0


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=TIMEOUT_SECONDS)


def _document_name(filename: str) -> str:
    """The name of a document chosen by the user, reduced to a plain name (no folders, no extension, only letters,
    digits, ``_`` and ``-``): it only names the pages ingested, never a path."""
    name = PurePosixPath(filename.replace("\\", "/")).name
    name = re.sub(r"\.pdf$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"[^\w-]+", "_", name).strip("_")[:80]
    return name or "document"


async def _ingest_pages(info: AuthorizedInfo, pages: list, name, tags: List["Tag"]) -> None:
    """Ingest every page (its markdown) in the memory of the agent, from memory: nothing is written on the disk of the
    instance, shared by every agent. ``name(i)`` is the name of the i-th page."""
    metadata = {item.name: item.value for item in tags}
    for i, page in enumerate(pages):
        content = page.get("markdown", "").encode("utf-8")
        await info.lizard.rabbit_hole.ingest_file(
            cat=info.cheshire_cat,
            file=BytesIO(content),
            filename=name(i),
            metadata=metadata,
            content_type="text/markdown",
        )


class Tag(BaseModel):
    name: str
    value: str | List[str]


class OCRInput(BaseModel):
    image: str
    type: str
    tags: List[Tag]


class OCRPDFInput(BaseModel):
    pdf: str
    filename: str
    tags: List[Tag]


@endpoint.post("/ocr")
async def ocr(
    ocr_input: OCRInput,
    info: AuthorizedInfo = check_permissions(AuthResource.MEMORY, AuthPermission.DELETE),
) -> dict:
    settings = await info.cheshire_cat.mad_hatter.get_plugin().load_settings()
    api_key = settings["mistral_api_key"]
    save_rh = settings["save_text_to_rabbit_hole"]

    api_url = "https://api.mistral.ai/v1/ocr"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    data = {
        "model": "mistral-ocr-latest",
        "document": {
            "type": "image_url",
            "image_url": f"data:{ocr_input.type};base64,{ocr_input.image}",
        },
    }

    response = None
    try:
        async with _client() as client:
            response = await client.post(api_url, headers=headers, json=data)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        ocr_response = response.json()

        log.debug(f"OCR response: {ocr_response}")

        if save_rh:
            await _ingest_pages(info, ocr_response.get("pages", []), lambda i: "ocrpage.md", ocr_input.tags)

        return ocr_response
    except httpx.HTTPError as e:
        log.debug(f"Error during OCR request: {e}")
        raise e
    except json.JSONDecodeError as e:
        if response is not None:
            log.debug(
                f"Error decoding JSON response: {e}. Response text: {response.text}"
            )
        raise CustomValidationException(f"Error decoding JSON response: {e}")
    except Exception as e:
        log.debug(f"An unexpected error occurred: {e}")
        raise e


@endpoint.post("/ocr-pdf")
async def ocr_pdf(
    ocr_input: OCRPDFInput,
    info: AuthorizedInfo = check_permissions(AuthResource.MEMORY, AuthPermission.DELETE),
) -> dict:
    name = _document_name(ocr_input.filename)
    settings = await info.cheshire_cat.mad_hatter.get_plugin().load_settings()
    api_key = settings["mistral_api_key"]
    save_rh = settings["save_text_to_rabbit_hole"]

    async with _client() as client:
        document_url = await upload_pdf(client, api_key, f"{name}.pdf", base64.b64decode(ocr_input.pdf))
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "mistral-ocr-latest",
            "document": {"type": "document_url", "document_url": document_url},
            "include_image_base64": True,
        }
        response = await client.post(
            "https://api.mistral.ai/v1/ocr", headers=headers, json=payload,
        )
    response.raise_for_status()
    ocr_response = response.json()
    log.debug(f"OCR PDF response: {ocr_response}")

    if save_rh:
        await _ingest_pages(info, ocr_response.get("pages", []), lambda i: f"{name}_{i}.md", ocr_input.tags)
    return ocr_response


async def upload_pdf(client: httpx.AsyncClient, api_key: str, filename: str, content: bytes) -> str:
    files = {"file": (filename, content)}
    data = {"purpose": "ocr"}
    headers = {"Authorization": f"Bearer {api_key}"}
    response = await client.post(
        "https://api.mistral.ai/v1/files", headers=headers, files=files, data=data,
    )
    response.raise_for_status()
    uploaded = response.json()
    file_id = uploaded["id"]

    # signed URL
    response = await client.get(
        f"https://api.mistral.ai/v1/files/{file_id}/url?expiry=24", headers=headers,
    )
    response.raise_for_status()
    return response.json()["url"]
