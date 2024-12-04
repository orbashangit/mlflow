import os
import shutil
from typing import Any

from mlflow.store.artifact.artifact_repo import (
    ArtifactRepository,
    try_read_trace_data,
    verify_artifact_path,
)
from mlflow.tracing.artifact_utils import TRACE_DATA_FILE_NAME
from mlflow.utils.file_utils import (
    get_file_info,
    list_all,
    local_file_uri_to_path,
    mkdir,
    relative_path_to_artifact_path,
)
from mlflow.utils.uri import validate_path_is_safe

## start of security improvements added on 2024-01-12
def validate_user_supplied_path(path: str, base_directory: str) -> str:
    """
    Validates and sanitizes user-supplied paths to prevent directory traversal attacks.

    Args:
        path: The user-supplied path to validate.
        base_directory: The base directory that all paths should be confined to.

    Returns:
        The sanitized absolute path if valid.

    Raises:
        ValueError: If the path is invalid or attempts to traverse outside the base directory.
    """
    # Validate that the input is a string
    if not isinstance(path, str):
        raise ValueError("The path must be a string.")

    # Manually decode URL-encoded characters (e.g., '%2e' -> '.')
    def manual_unquote(s):
        res = ''
        i = 0
        while i < len(s):
            if s[i] == '%' and i + 2 < len(s):
                hex_value = s[i + 1:i + 3]
                try:
                    res += chr(int(hex_value, 16))
                    i += 3
                except ValueError:
                    res += s[i]
                    i += 1
            else:
                res += s[i]
                i += 1
        return res

    path = manual_unquote(path)

    # Normalize the path to remove traversal sequences
    normalized_path = os.path.normpath(path)

    # Prevent paths that resolve to parent directories
    if normalized_path.startswith("..") or os.path.isabs(normalized_path):
        raise ValueError("Invalid path: Path traversal is not allowed.")

    # Ensure the path does not contain any traversal after normalization
    if any(part == ".." for part in normalized_path.split(os.sep)):
        raise ValueError("Invalid path: Path traversal is not allowed.")

    # Construct the absolute path
    absolute_path = os.path.abspath(os.path.join(base_directory, normalized_path))

    # Ensure the absolute path is within the base directory
    if not absolute_path.startswith(os.path.abspath(base_directory) + os.sep):
        raise ValueError("Invalid path: Access outside the allowed directory is prohibited.")

    return absolute_path
# Fix implemented on 2024-01-12: Added input validation to prevent directory traversal attacks.
## end of security improvements added on 2024-01-12


class LocalArtifactRepository(ArtifactRepository):
    """Stores artifacts as files in a local directory."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._artifact_dir = local_file_uri_to_path(self.artifact_uri)

    @property
    def artifact_dir(self):
        return self._artifact_dir

    def log_artifact(self, local_file, artifact_path=None):
        verify_artifact_path(artifact_path)
        # NOTE: The artifact_path is expected to be in posix format.
        # Posix paths work fine on windows but just in case we normalize it here.
        if artifact_path:
            artifact_path = os.path.normpath(artifact_path)

        artifact_dir = (
            os.path.join(self.artifact_dir, artifact_path) if artifact_path else self.artifact_dir
        )
        if not os.path.exists(artifact_dir):
            mkdir(artifact_dir)
        try:
            shutil.copy2(local_file, os.path.join(artifact_dir, os.path.basename(local_file)))
        except shutil.SameFileError:
            pass

    def _is_directory(self, artifact_path):
        # NOTE: The path is expected to be in posix format.
        # Posix paths work fine on windows but just in case we normalize it here.
        path = os.path.normpath(artifact_path) if artifact_path else ""
        list_dir = os.path.join(self.artifact_dir, path) if path else self.artifact_dir
        return os.path.isdir(list_dir)

    def log_artifacts(self, local_dir, artifact_path=None):
        verify_artifact_path(artifact_path)
        # NOTE: The artifact_path is expected to be in posix format.
        # Posix paths work fine on windows but just in case we normalize it here.
        if artifact_path:
            artifact_path = os.path.normpath(artifact_path)
        artifact_dir = (
            os.path.join(self.artifact_dir, artifact_path) if artifact_path else self.artifact_dir
        )
        if not os.path.exists(artifact_dir):
            mkdir(artifact_dir)
        shutil.copytree(src=local_dir, dst=artifact_dir, dirs_exist_ok=True)

    def download_artifacts(self, artifact_path, dst_path=None):
        """
        Artifacts tracked by ``LocalArtifactRepository`` already exist on the local filesystem.
        If ``dst_path`` is ``None``, the absolute filesystem path of the specified artifact is
        returned. If ``dst_path`` is not ``None``, the local artifact is copied to ``dst_path``.

        Args:
            artifact_path: Relative source path to the desired artifacts.
            dst_path: Absolute path of the local filesystem destination directory to which to
                download the specified artifacts. This directory must already exist. If
                unspecified, the absolute path of the local artifact will be returned.

        Returns:
            Absolute path of the local filesystem location containing the desired artifacts.
        """
        if dst_path:
            return super().download_artifacts(artifact_path, dst_path)
        # NOTE: The artifact_path is expected to be a relative path in posix format.
        # Posix paths work fine on windows but just in case we normalize it here.

        ## start of security improvements added on 2024-01-12
        # Validate the user-supplied artifact_path
        artifact_path = validate_user_supplied_path(artifact_path or "", self.artifact_dir)
        ## end of security improvements added on 2024-01-12

        local_artifact_path = os.path.join(self.artifact_dir, os.path.normpath(artifact_path))
        if not os.path.exists(local_artifact_path):
            raise OSError(f"No such file or directory: '{local_artifact_path}'")
        return os.path.abspath(local_artifact_path)

    def list_artifacts(self, path=None):
        # NOTE: The path is expected to be in posix format.
        # Posix paths work fine on windows but just in case we normalize it here.
        if path:
            path = os.path.normpath(path)
        list_dir = os.path.join(self.artifact_dir, path) if path else self.artifact_dir
        if os.path.isdir(list_dir):
            artifact_files = list_all(list_dir, full_path=True)
            infos = [
                get_file_info(
                    f, relative_path_to_artifact_path(os.path.relpath(f, self.artifact_dir))
                )
                for f in artifact_files
            ]
            return sorted(infos, key=lambda f: f.path)
        else:
            return []

    def _download_file(self, remote_file_path, local_path):
        # NOTE: The remote_file_path is expected to be a relative path in posix format.
        # Posix paths work fine on windows but just in case we normalize it here.

        ## start of security improvements added on 2024-01-12
        # Validate the user-supplied remote_file_path
        remote_file_path = validate_user_supplied_path(remote_file_path, self.artifact_dir)
        ## end of security improvements added on 2024-01-12

        remote_file_path = os.path.join(self.artifact_dir, os.path.normpath(remote_file_path))
        shutil.copy2(remote_file_path, local_path)

    def delete_artifacts(self, artifact_path=None):

        ## start of security improvements added on 2024-01-12
        # Validate the user-supplied artifact_path
        artifact_path = validate_user_supplied_path(artifact_path or "", self._artifact_dir)
        ## end of security improvements added on 2024-01-12

        artifact_path = local_file_uri_to_path(artifact_path)

        if os.path.exists(artifact_path):
            if os.path.isfile(artifact_path):
                os.remove(artifact_path)
            else:
                shutil.rmtree(artifact_path)

    def download_trace_data(self) -> dict[str, Any]:
        """
        Download the trace data.

        Returns:
            The trace data as a dictionary.

        Raises:
            - `MlflowTraceDataNotFound`: The trace data is not found.
            - `MlflowTraceDataCorrupted`: The trace data is corrupted.
        """
        trace_data_path = os.path.join(self.artifact_dir, TRACE_DATA_FILE_NAME)
        return try_read_trace_data(trace_data_path)
