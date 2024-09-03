"""
twelvelabs_client.py
--------------------
Python client wrapper for the TwelveLabs API.

Covers:
- Index management  (create, list, get, delete)
- Video upload & task management
- Embedding generation (Marengo / Pegasus models)
- Search  (semantic, visual, conversational)

Reference: https://docs.twelvelabs.io/v1.3/reference
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger(__name__)

_BASE_URL = "https://api.twelvelabs.io/v1.3/"
_DEFAULT_TIMEOUT = 60          # seconds
_UPLOAD_TIMEOUT = 300          # upload can be slower


class TwelveLabsAPIError(Exception):
    """Raised when the TwelveLabs API returns a non-2xx response."""

    def __init__(self, status_code: int, message: str, response_body: dict[str, Any] | None = None) -> None:
        super().__init__(f"HTTP {status_code}: {message}")
        self.status_code = status_code
        self.response_body = response_body or {}


class TwelveLabsClient:
    """
    Thread-safe HTTP client for the TwelveLabs REST API.

    Parameters
    ----------
    api_key : str
        Your TwelveLabs API key (``tlk_...``).
    base_url : str, optional
        Override the default API base URL (useful for testing).
    timeout : int, optional
        Default request timeout in seconds.
    max_retries : int, optional
        Number of times to retry on transient 5xx / connection errors.

    Example
    -------
    >>> client = TwelveLabsClient(api_key="tlk_xxxx")
    >>> index = client.create_index("my-videos", models=["marengo2.7"])
    >>> task = client.upload_video(index["_id"], video_url="https://...")
    >>> results = client.search(index["_id"], query="person laughing")
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = _BASE_URL,
        timeout: int = _DEFAULT_TIMEOUT,
        max_retries: int = 3,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout
        self._session = self._build_session(max_retries)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_session(self, max_retries: int) -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=max_retries,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST", "DELETE", "PATCH"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.headers.update(
            {
                "x-api-key": self._api_key,
                "Accept": "application/json",
            }
        )
        return session

    def _url(self, path: str) -> str:
        return urljoin(self._base_url, path.lstrip("/"))

    def _raise_for_status(self, response: requests.Response) -> None:
        if not response.ok:
            try:
                body = response.json()
            except Exception:
                body = {"raw": response.text}
            message = body.get("message") or body.get("error") or response.reason
            raise TwelveLabsAPIError(response.status_code, message, body)

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = self._url(path)
        log.debug("GET %s params=%s", url, params)
        resp = self._session.get(url, params=params, timeout=self._timeout)
        self._raise_for_status(resp)
        return resp.json()

    def _post(self, path: str, json_body: dict[str, Any] | None = None, timeout: int | None = None) -> dict[str, Any]:
        url = self._url(path)
        log.debug("POST %s body=%s", url, json_body)
        resp = self._session.post(url, json=json_body, timeout=timeout or self._timeout)
        self._raise_for_status(resp)
        return resp.json()

    def _delete(self, path: str) -> None:
        url = self._url(path)
        log.debug("DELETE %s", url)
        resp = self._session.delete(url, timeout=self._timeout)
        self._raise_for_status(resp)

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def create_index(
        self,
        name: str,
        models: Optional[list[str]] = None,
        addons: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Create a new TwelveLabs index.

        Parameters
        ----------
        name : str
            Human-readable index name.
        models : list[str], optional
            Model engine names, e.g. ``["marengo2.7", "pegasus1.2"]``.
        addons : list[str], optional
            Optional add-ons, e.g. ``["thumbnail"]``.

        Returns
        -------
        dict
            Index object containing ``_id``, ``name``, ``status``, etc.
        """
        payload: dict[str, Any] = {
            "index_name": name,
            "engines": [
                {"engine_name": m, "engine_options": ["visual", "conversation", "text_in_video", "logo"]}
                for m in (models or ["marengo2.7"])
            ],
        }
        if addons:
            payload["addons"] = addons

        result = self._post("indexes", payload)
        log.info("Created index '%s' with id=%s", name, result.get("_id"))
        return result

    def list_indexes(self, page: int = 1, page_limit: int = 50) -> dict[str, Any]:
        """List all indexes for the account."""
        return self._get("indexes", params={"page": page, "page_limit": page_limit})

    def get_index(self, index_id: str) -> dict[str, Any]:
        """Fetch a single index by ID."""
        return self._get(f"indexes/{index_id}")

    def delete_index(self, index_id: str) -> None:
        """Permanently delete an index and all its videos."""
        self._delete(f"indexes/{index_id}")
        log.info("Deleted index id=%s", index_id)

    # ------------------------------------------------------------------
    # Video upload & task management
    # ------------------------------------------------------------------

    def upload_video(
        self,
        index_id: str,
        video_url: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Submit a video for indexing via URL.

        Parameters
        ----------
        index_id : str
            Target index ID.
        video_url : str
            Publicly accessible (or presigned) URL to the video file.
        metadata : dict, optional
            Arbitrary key-value metadata to attach to the video.

        Returns
        -------
        dict
            Task object with ``_id`` and ``status`` fields.
        """
        payload: dict[str, Any] = {
            "index_id": index_id,
            "video_url": video_url,
            "language": "en",
        }
        if metadata:
            payload["metadata"] = metadata

        result = self._post("tasks", payload, timeout=_UPLOAD_TIMEOUT)
        log.info(
            "Upload task created: task_id=%s index_id=%s url=%s",
            result.get("_id"),
            index_id,
            video_url,
        )
        return result

    def get_task_status(self, task_id: str) -> dict[str, Any]:
        """
        Retrieve the current status of an upload/indexing task.

        Returns
        -------
        dict
            Task object; key field is ``status``:
            ``pending | indexing | ready | failed``.
        """
        return self._get(f"tasks/{task_id}")

    def wait_for_task(
        self,
        task_id: str,
        poll_interval: int = 30,
        timeout: int = 1200,
    ) -> dict[str, Any]:
        """
        Block until a task reaches a terminal state (``ready`` or ``failed``).

        Parameters
        ----------
        task_id : str
        poll_interval : int
            Seconds between status checks.
        timeout : int
            Maximum seconds to wait before raising ``TimeoutError``.

        Returns
        -------
        dict
            Final task object.

        Raises
        ------
        TimeoutError
            If the task does not complete within *timeout* seconds.
        """
        deadline = time.monotonic() + timeout
        while True:
            task = self.get_task_status(task_id)
            status = task.get("status")
            log.debug("Task %s status=%s", task_id, status)
            if status in ("ready", "failed"):
                return task
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Task {task_id} did not complete within {timeout}s (last status: {status})")
            time.sleep(poll_interval)

    def list_videos(self, index_id: str, page: int = 1, page_limit: int = 50) -> dict[str, Any]:
        """List all videos inside an index."""
        return self._get(f"indexes/{index_id}/videos", params={"page": page, "page_limit": page_limit})

    def get_video(self, index_id: str, video_id: str) -> dict[str, Any]:
        """Fetch metadata for a single video."""
        return self._get(f"indexes/{index_id}/videos/{video_id}")

    def delete_video(self, index_id: str, video_id: str) -> None:
        """Remove a video from an index."""
        self._delete(f"indexes/{index_id}/videos/{video_id}")
        log.info("Deleted video_id=%s from index_id=%s", video_id, index_id)

    # ------------------------------------------------------------------
    # Embedding generation
    # ------------------------------------------------------------------

    def generate_embedding(
        self,
        video_id: str,
        model: str = "Marengo-retrieval-2.7",
    ) -> dict[str, Any]:
        """
        Generate a video-level embedding using the TwelveLabs Embed API.

        Parameters
        ----------
        video_id : str
            The TwelveLabs video ID (``tlv_...``).
        model : str
            Embedding model name.

        Returns
        -------
        dict
            Embedding object containing ``embedding`` (list[float]),
            ``model``, ``video_id``, and ``dimension``.
        """
        payload = {"video_id": video_id, "model_name": model}
        result = self._post("embed/tasks", payload)
        log.info(
            "Embedding request submitted: video_id=%s model=%s task_id=%s",
            video_id,
            model,
            result.get("_id"),
        )
        # Poll until embedding task is done
        embed_task = self._wait_for_embed_task(result["_id"])
        return embed_task

    def _wait_for_embed_task(
        self,
        embed_task_id: str,
        poll_interval: int = 10,
        timeout: int = 600,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while True:
            task = self._get(f"embed/tasks/{embed_task_id}")
            status = task.get("status")
            if status == "ready":
                return task
            if status == "failed":
                raise TwelveLabsAPIError(422, f"Embed task {embed_task_id} failed", task)
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Embed task {embed_task_id} timed out after {timeout}s")
            time.sleep(poll_interval)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        index_id: str,
        query: str,
        search_options: Optional[list[str]] = None,
        group_by: str = "clip",
        threshold: str = "medium",
        page_limit: int = 10,
        filter: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Execute a semantic search over a TwelveLabs index.

        Parameters
        ----------
        index_id : str
        query : str
            Natural-language query string.
        search_options : list[str], optional
            Modalities to search over: ``["visual", "conversation", "text_in_video", "logo"]``.
        group_by : str
            ``"clip"`` or ``"video"``.
        threshold : str
            Confidence threshold: ``"low" | "medium" | "high"``.
        page_limit : int
            Maximum results per page.
        filter : dict, optional
            Metadata filter, e.g. ``{"id": ["tlv_..."]}``.

        Returns
        -------
        dict
            Search response with ``data`` list and ``page_info``.
        """
        payload: dict[str, Any] = {
            "index_id": index_id,
            "query": query,
            "search_options": search_options or ["visual", "conversation"],
            "group_by": group_by,
            "threshold": threshold,
            "page_limit": page_limit,
        }
        if filter:
            payload["filter"] = filter

        result = self._post("search", payload)
        log.info(
            "Search completed: index=%s query='%s' hits=%d",
            index_id,
            query[:60],
            len(result.get("data", [])),
        )
        return result

    def search_by_page_token(self, page_token: str) -> dict[str, Any]:
        """Fetch the next page of search results using a page token."""
        return self._get("search", params={"page_token": page_token})

    # ------------------------------------------------------------------
    # Summarization / Generation (Pegasus)
    # ------------------------------------------------------------------

    def summarize_video(
        self,
        video_id: str,
        summary_type: str = "summary",
        prompt: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Generate a text summary of a video using the Pegasus model.

        Parameters
        ----------
        video_id : str
        summary_type : str
            ``"summary" | "chapter" | "highlight"``.
        prompt : str, optional
            Custom instruction prompt for the generation.

        Returns
        -------
        dict
            Generation object with ``summary`` text.
        """
        payload: dict[str, Any] = {"video_id": video_id, "type": summary_type}
        if prompt:
            payload["prompt"] = prompt
        return self._post("summarize", payload)
