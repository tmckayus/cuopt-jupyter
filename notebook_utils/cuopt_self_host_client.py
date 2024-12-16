# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE


import json
import logging
import time
from uuid import UUID

import msgpack
import msgpack_numpy
import requests

msgpack_numpy.patch()


def get_version():
    """
    Return client version.
    """
    return f"24.11 (local)"


log_fmt = "%(asctime)s.%(msecs)03d %(name)s %(levelname)s %(message)s"
date_fmt = "%Y-%m-%d %H:%M:%S"
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format=log_fmt, datefmt=date_fmt)


def set_log_level(level):
    log.setLevel(level)


def load_data(file_path, lp=False):
    with open(file_path, "rb") as f:
        raw_data = f.read()

        # Check the file extension to determine the content type
        ext = file_path.split(".")[-1] if "." in file_path else ""
        if ext == "zlib":
            content_type = "application/zlib"
        elif ext == "msgpack":
            content_type = "application/vnd.msgpack"
        elif ext in ["json", ""]:
            if not ext:
                log.info("No file extension given, assuming JSON")
            content_type = "application/json"
        elif ext == "pickle":
            content_type = "application/octet-stream"
        else:
            raise ValueError(
                f"File extension {ext} is unsupported. "
                "Supported file extensions are "
                ".json, .zlib, .msgpack, or .pickle"
            )
        return raw_data, content_type


def is_uuid(cuopt_problem_data):
    try:
        _ = UUID(cuopt_problem_data, version=4)
        return True
    except Exception:
        return False

class CuOptServiceSelfHostClient:
    """
    This version of the CuOptServiceClient is an interface
    with a self hosted version of the cuOpt core service.
    This client allows users to make calls to a self hosted
    instance of the cuOpt service at a specific ip and port.

    It closely emulates the interface of the managed service client,
    however it does not implement most of the managed service-specific
    features required for interaction with NVIDIA Cloud Functions.

    Parameters
    ----------
    ip (str, optional): The IP address of the cuOpt service.
            Defaults to 0.0.0.0
    port (str, optional): The port of the cuOpt service.
            Defaults to 5000.
    use_https (boolean, optional): Use HTTPS to communicate
            with server in secured way.
    self_signed_cert (str, optional): A complete path to
            self signed certificate. If it's a standard certificate,
            then no need to provide anything.
    polling_interval (int, optional): The duration in seconds between
            consecutive polling attempts. Defaults to 1.
    request_excess_timeout (int, optional): Deprecated.
            Use polling_timeout instead
    only_validate (boolean, optional): Only validates input.
            Defaults to False.
    polling_timeout: (int, optional): The time in seconds that
            the client will poll for a result before exiting and
            returning a request id. The request id may be polled again
            in a call to repoll(). If set to None, the client
            will never timeout and will poll indefinitely.
            Defaults to 600.
    timeout_exception: (boolean, optional):  If True, the client returns
            a TimeoutError exception if polling_timeout seconds passes before a
            solution is returned. The value of the exception contains JSON
            giving the request id so that repoll() may be called to poll again
            for a result.
            If False, the client returns a dictionary containing the
            repoll information with no exception. Defaults to True.
    """

    # Initialize the CuOptServiceSelfHostClient
    def __init__(
        self,
        ip: str = "0.0.0.0",
        port: str = "5000",
        use_https: bool = False,
        self_signed_cert="",
        polling_interval=1,
        request_excess_timeout=None,
        only_validate=False,
        polling_timeout=600,
        timeout_exception=True,
    ):
        self.timeout_exception = timeout_exception
        self.ip = ip
        self.port = port
        self.only_validate = only_validate

        self.protocol = "https" if use_https else "http"
        self.verify = False
        if use_https is True:
            if len(self_signed_cert) > 0:
                self.verify = self_signed_cert
            else:
                self.verify = True

        self.binary_url = (
            f"{self.protocol}://{self.ip}:{self.port}/cuopt/request"  # noqa
        )
        self.polling_interval = polling_interval
        self.timeout = (
            request_excess_timeout
            if request_excess_timeout is not None
            else polling_timeout
        )

    def _get_response(self, response):
        if response.headers["Content-Type"] == "application/json":
            log.debug("reading response as json")
            response = response.json()
        else:
            log.debug("reading response as msgpack")
            response = msgpack.loads(response.content)
        return response

    def _handle_request_exception(self, response):
        r = self._get_response(response)
        msg = r.get("error", r)
        raise ValueError(
            f"cuOpt Error: {response.reason} - {response.status_code}: {msg}"
        )

    def _poll_request(self, response):
        poll_start = time.time()
        while True:
            response = self._get_response(response)
            if (
                "response" in response
                or "result_file" in response
                or "error" in response
            ):
                break
            # For backward compat with images < 24.03
            if "id" in response:
                response["reqId"] = response["id"]
                del response["id"]
            if (
                self.timeout is not None
                and time.time() - poll_start > self.timeout
            ):
                if self.timeout_exception:
                    raise TimeoutError(json.dumps(response))
                else:
                    return response
            time.sleep(self.polling_interval)
            try:
                log.debug(f"GET {self.binary_url}/{response['reqId']}")
                headers = {"Accept": "application/vnd.msgpack"}
                response = requests.get(
                    self.binary_url + f"/{response['reqId']}",
                    verify=self.verify,
                    headers=headers,
                    timeout=30,
                )
                response.raise_for_status()
                log.debug(response.status_code)
            except requests.exceptions.HTTPError as e:
                log.debug(str(e))
                self._handle_request_exception(response)
        return response

    # Send the request to the local cuOpt core service
    def _send_request(self, cuopt_problem_data, filepath, cache, output):
        def serialize(cuopt_problem_data):
            if isinstance(cuopt_problem_data, dict):
                data = msgpack.dumps(cuopt_problem_data)
                content_type = "application/vnd.msgpack"
            elif isinstance(cuopt_problem_data, list):
                if all(isinstance(d, str) for d in cuopt_problem_data):
                    # Make this a list of tuples of content_type and
                    # a byte stream, and serialize the whole thing
                    # with mspagck
                    final_list = []
                    for d in cuopt_problem_data:
                        data, content_type = load_data(d)
                        final_list.append((content_type, data))
                    data = msgpack.dumps(final_list)
                    content_type = "application/vnd.msgpack"
                else:
                    data = msgpack.dumps(cuopt_problem_data)
                    content_type = "application/vnd.msgpack"
            else:
                data, content_type = load_data(cuopt_problem_data)
            return data, content_type

        try:
            log.debug(f"POST {self.binary_url}")
            headers = {}
            params = {"validation_only": self.only_validate}
            params["cache"] = cache
            if is_uuid(cuopt_problem_data):
                data = {}
                params["reqId"] = cuopt_problem_data
                content_type = "application/json"
            elif filepath:
                headers["CUOPT-DATA-FILE"] = cuopt_problem_data
                data = {}
                content_type = "application/json"
            else:
                data, content_type = serialize(cuopt_problem_data)
            headers["CLIENT-VERSION"] = "24.11.local"
            # Immediately return and then poll on the id
            if output:
                headers["CUOPT-RESULT-FILE"] = output
            headers["Content-Type"] = content_type
            headers["Accept"] = "application/vnd.msgpack"
            response = requests.post(
                self.binary_url,
                params=params,
                data=data,
                headers=headers,
                verify=self.verify,
                timeout=30,
            )
            response.raise_for_status()
            log.debug(response.status_code)
        except requests.exceptions.HTTPError as e:
            log.debug(str(e))
            self._handle_request_exception(response)
        if cache:
            return self._get_response(response)
        return self._poll_request(response)

    def _cleanup_response(self, res):
        if "warnings" in res:
            for w in res["warnings"]:
                log.warning(w)
            del res["warnings"]
        if "notes" in res:
            for n in res["notes"]:
                log.info(n)
            del res["notes"]
        return res

    # Get optimized routes for the given cuOpt problem instance
    def get_optimized_routes(
        self, cuopt_problem_json_data, filepath=False, cache=False, output=""
    ):
        """
        Get optimized routing solution for a given problem.

        Parameters
        ----------
        cuopt_problem_json_data : dict or str
            This is either the problem data as a dictionary or the
            path of a file containing the problem data as JSON or
            the reqId of a cached cuopt problem data.
            Please refer to the server doc for the
            structure of this dictionary.
        filepath (boolean, optional): Indicates that cuopt_problem_json_data
            is the relative path of a cuopt data file under the server's
            data directory. The data directory is specified when the server
            is started (see the server documentation for more detail).
            Defaults to False.
        output (string, optional): Optional name of the result file.
            If the server has been configured to write results to files and
            the size of the result is greater than the configured
            limit, the server will write the result to a file with
            this name under the server's result directory (see the
            server documentation for more detail). Defaults to a
            name based on the path if 'filepath' is True,
            or a uuid if 'filepath' is False.
        """
        if filepath and cuopt_problem_json_data.startswith("/"):
            log.warn(
                "Path of the data file on the server was specified, "
                "but an absolute path was given. "
                "Best practice is to specify the relative path of a "
                "data file under the CUOPT_DATA_DIR directory "
                "which was configured when the cuopt server was started."
            )

        res = self._send_request(
            cuopt_problem_json_data, filepath, cache, output
        )
        return self._cleanup_response(res)

    def delete(self, id, running=None, queued=None, cached=None):
        """
        Delete a cached entry by id or abort a job by id.

        Parameters
        ----------
        id : str
            A uuid identifying the cached entry or job to be deleted. The
            wildcard id '*' will match all uuids (filtered by 'running',
            'queued', and 'cached').
        running : bool
            If set to True, the request will be aborted if 'id' is a currently
            running job. Defaults to True if 'id' is a specific uuid and both
            'queued' and 'cached' are unspecified, otherwise False.
        queued : bool
            If set to True, the request will be aborted if 'id' is a currently
            queued job. Defaults to True if 'id' is a specific uuid and both
            'running' and 'cached' are unspecified, otherwise False.
        cached: bool
            If set to True, the request will be aborted if 'id' is a cached
            data entry. Defaults to True if 'id' is a specific uuid and both
            'running' and 'queued' are unspecified, otherwise False.
        """
        try:
            response = requests.delete(
                self.binary_url + f"/{id}",
                params={
                    "running": running,
                    "queued": queued,
                    "cached": cached,
                },
                verify=self.verify,
                timeout=30,
            )
            response.raise_for_status()
            log.debug(response.status_code)
            return response.json()

        except requests.exceptions.HTTPError as e:
            log.debug(str(e))
            self._handle_request_exception(response)

    def repoll(self, data, response_type="obj"):
        """
        Poll for a result when a previous command resulted in
        a timeout. The request id is returned as JSON
        in the result of the original call.

        Parameters
        ----------
        data : str
            A uuid identifying the original request.
            For backward compatibility, data may also be a dictionary
            containing the key 'reqId' where the value is the uuid.
        response_type: str
            For LP problem choose "dict" if response should be returned
            as a dictionary or "obj" for Solution object.
            Defaults to "obj".
            For VRP problem, response_type is ignored and always
            returns a dict.
        """
        if isinstance(data, dict):
            data = data["reqId"]
        headers = {"Accept": "application/vnd.msgpack"}
        response = requests.get(
            self.binary_url + f"/{data}",
            verify=self.verify,
            headers=headers,
            timeout=30,
        )
        if response_type == "dict":
            return self._cleanup_response(self._poll_request(response))
        else:
            return create_lp_response(
                self._cleanup_response(self._poll_request(response))
            )
