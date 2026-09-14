#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

"""HTTP client for remote feedback-audio generation."""

import json
import socket
import time
import pepper_config as settings

try:
    from urllib2 import Request, URLError, HTTPError, urlopen
except ImportError:
    from urllib.request import Request, urlopen
    from urllib.error import URLError, HTTPError


class FeedbackClientError(Exception):
    """Base exception for feedback client failures."""


class FeedbackValidationError(FeedbackClientError):
    """Raised for HTTP 400 validation errors."""


class FeedbackUpstreamError(FeedbackClientError):
    """Raised for HTTP 502 upstream generation/synthesis failures."""


class FeedbackHTTPError(FeedbackClientError):
    """Raised for other non-200 HTTP errors."""


class FeedbackNetworkError(FeedbackClientError):
    """Raised for timeout/connection failures after retries."""


class FeedbackResponseError(FeedbackClientError):
    """Raised for invalid success responses."""


def clamp_round_score(score):
    """Round DTW score and clamp to contract range [0, 100]."""
    try:
        score_value = float(score)
    except Exception:
        raise ValueError("Score must be numeric: %r" % (score,))

    score_int = int(round(score_value))
    if score_int < 0:
        return 0
    if score_int > 100:
        return 100
    return score_int


def score_to_feedback_category(score):
    """Map DTW score to server-facing feedback categories."""
    score_int = clamp_round_score(score)
    if score_int >= 90:
        return "excellent"
    if score_int >= 75:
        return "good"
    if score_int >= 50:
        return "fair"
    return "needs_improvement"


def build_feedback_payload(score, category, exercise_name):
    """Build and validate payload for /generate-feedback-audio."""
    score_int = clamp_round_score(score)
    category = (category or "").strip()
    exercise_name = (exercise_name or "").strip()

    if not category:
        raise ValueError("Category must be a non-empty string.")
    if not exercise_name:
        raise ValueError("Exercise name must be a non-empty string.")

    return {
        "score": score_int,
        "category": category,
        "exercise_name": exercise_name,
    }


def _json_error_detail(body_bytes):
    if not body_bytes:
        return ""
    try:
        if not isinstance(body_bytes, str):
            body_text = body_bytes.decode("utf-8", "replace")
        else:
            body_text = body_bytes
        payload = json.loads(body_text)
        detail = payload.get("detail")
        if detail is None:
            return body_text
        return str(detail)
    except Exception:
        try:
            if isinstance(body_bytes, str):
                return body_bytes
            return body_bytes.decode("utf-8", "replace")
        except Exception:
            return ""


def _read_http_error_body(error):
    try:
        return error.read()
    except Exception:
        return b""


def _response_status(response):
    if hasattr(response, "getcode"):
        return response.getcode()
    return getattr(response, "status", None)


def _response_content_type(response):
    headers = getattr(response, "headers", None)
    if headers is None:
        return ""
    try:
        return headers.get("Content-Type", "")
    except Exception:
        return ""


def _is_http_error(exc):
    return isinstance(exc, HTTPError) or hasattr(exc, "code")


def _is_transient_network_error(exc):
    if isinstance(exc, socket.timeout):
        return True
    if isinstance(exc, URLError):
        return True
    reason = getattr(exc, "reason", None)
    if isinstance(reason, socket.timeout):
        return True
    return False


class FeedbackAudioClient(object):
    """Client for remote feedback audio generation service."""

    def __init__(
        self,
        base_url=settings.DEFAULT_FEEDBACK_SERVER_URL,
        timeout_sec=settings.DEFAULT_TIMEOUT_SEC,
        max_retries=settings.DEFAULT_MAX_RETRIES,
        retry_delay_sec=settings.DEFAULT_RETRY_DELAY_SEC,
        opener=None,
        sleep_fn=None,
    ):
        base_url = (base_url or settings.DEFAULT_FEEDBACK_SERVER_URL).rstrip("/")
        self.base_url = base_url
        self.timeout_sec = float(timeout_sec)
        self.max_retries = int(max_retries)
        self.retry_delay_sec = float(retry_delay_sec)
        self._opener = opener if opener is not None else urlopen
        self._sleep = sleep_fn if sleep_fn is not None else time.sleep

    def healthz(self):
        request = Request(self.base_url + "/healthz")
        response = self._opener(request, timeout=self.timeout_sec)
        status = _response_status(response)
        return status == 200

    def generate_feedback_audio(self, score, category, exercise_name):
        payload = build_feedback_payload(score, category, exercise_name)
        request_data = json.dumps(payload).encode("utf-8")
        request = Request(
            self.base_url + "/generate-feedback-audio",
            data=request_data,
            headers={
                "Content-Type": "application/json",
                "Accept": "audio/wav",
            },
        )

        attempts = self.max_retries + 1
        last_network_error = None

        for attempt in range(attempts):
            try:
                response = self._opener(request, timeout=self.timeout_sec)
                status = _response_status(response)
                body = response.read()
                content_type = _response_content_type(response).lower()

                if status == 200:
                    if "audio/wav" not in content_type:
                        raise FeedbackResponseError(
                            "Expected audio/wav response, got Content-Type: %s" %
                            (_response_content_type(response) or "<missing>")
                        )
                    return body

                detail = _json_error_detail(body)
                self._raise_http_error(status, detail)
            except Exception as exc:
                if _is_http_error(exc):
                    status = int(getattr(exc, "code", 0) or 0)
                    body = _read_http_error_body(exc)
                    detail = _json_error_detail(body)
                    self._raise_http_error(status, detail)

                if _is_transient_network_error(exc):
                    last_network_error = exc
                    if attempt < (attempts - 1):
                        if self.retry_delay_sec > 0:
                            self._sleep(self.retry_delay_sec)
                        continue
                    raise FeedbackNetworkError(
                        "Network error while calling feedback service (%s): %s" %
                        (self.base_url, str(exc))
                    )
                raise

        if last_network_error is not None:
            raise FeedbackNetworkError(
                "Network error while calling feedback service (%s): %s" %
                (self.base_url, str(last_network_error))
            )
        raise FeedbackClientError("Feedback request failed without a response.")

    def _raise_http_error(self, status, detail):
        if status == 400:
            raise FeedbackValidationError(
                "Feedback server validation failed (HTTP 400): %s" % (detail or "No detail provided.")
            )
        if status == 502:
            raise FeedbackUpstreamError(
                "Feedback server upstream failure (HTTP 502): %s" % (detail or "No detail provided.")
            )
        raise FeedbackHTTPError(
            "Feedback server request failed (HTTP %s): %s" %
            (str(status), detail or "No detail provided.")
        )
