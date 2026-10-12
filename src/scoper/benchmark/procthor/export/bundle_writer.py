from __future__ import annotations
import copy
import fcntl
import io
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Sequence
import numpy as np
from PIL import Image
from scoper.benchmark.procthor.export.ids import audit_id as make_audit_id, build_episode_identity, canonical_json_bytes, derive_seed, frame_id as make_frame_id, model_call_id as make_model_call_id, record_id as make_record_id, sha256_bytes, sha256_json, state_snapshot_id as make_state_snapshot_id
from scoper.benchmark.procthor.export.execution_audit import NativeProfileViolation, validate_execution_audit_records
from scoper.benchmark.procthor.export.leakage import LeakageError, assert_no_operational_leakage, assert_no_public_leakage, audit_public_private_sidecars, audit_public_value
from scoper.benchmark.procthor.export.quality import analyze_encoded_image, is_critical_frame_kind
from scoper.benchmark.procthor.export.schema import BACKEND, COMMON_CONTRACT_SHA256, COMMON_CONTRACT_VERSION, CONTRACT_VERSION, EXECUTION_AUDIT_SCHEMA_VERSION, RFC3339_UTC_RE, SCHEMA_VERSIONS, SEED_NAMESPACES, STATE_SCHEMA_VERSION, TrajectoryExportError, _invalid_terminal_string_field, load_json_strict, load_jsonl_strict, strict_json_loads, validate_common_contract_fixture, validate_asset_cache_manifest, validate_episode_manifest, validate_event_payload, validate_json_value, validate_payload_record, validate_provenance_document, validate_quality_report, validate_run_manifest_minimal, validate_trajectory_record
from scoper.benchmark.procthor.export.types import CaptureConfig, EpisodeContext, EpisodeKeyInput, FrameMetadata, RunContext, SkippedEpisode

class BundleConflictError(TrajectoryExportError):
    pass

class BundleIntegrityError(TrajectoryExportError):
    pass

class EpisodeBundleWriter:
    _JSONL_PATHS = ('trajectory.jsonl', 'public/trajectory.jsonl', 'privileged/trajectory.jsonl', 'public/frames.jsonl', 'public/model_io.jsonl', 'privileged/frame_annotations.jsonl', 'privileged/state_snapshots.jsonl', 'privileged/execution_audit.jsonl', 'privileged/binding_log.jsonl')

    def __init__(self, *, run_context: RunContext, episode_context: EpisodeContext, capture_config: CaptureConfig, run_root: Path, partial_root: Path) -> None:
        self.run_context = run_context
        self.episode_context = episode_context
        self.capture_config = capture_config
        self.run_root = run_root
        self.partial_root = partial_root
        self.final_root = run_root / 'episodes' / episode_context.episode_id
        self.state = 'OPEN'
        self._handles: dict[str, BinaryIO] = {}
        self._sequence_index = 0
        self._frame_ordinal = 0
        self._state_ordinal = 0
        self._model_call_ordinal = 0
        self._audit_ordinal = 0
        self._last_logical_time = 0.0
        self._last_monotonic_time = 0
        self._event_types: list[str] = []
        self._frame_ids: set[str] = set()
        self._model_frame_ids: set[str] = set()
        self._state_ids: set[str] = set()
        self._model_call_ids: set[str] = set()
        self._frame_hashes: set[str] = set()
        self._started_at_utc = _utc_now()
        self._started_monotonic_ns = time.monotonic_ns()

    @classmethod
    def open(cls, *, run_context: RunContext, episode_context: EpisodeContext, capture_config: CaptureConfig, resume: bool) -> 'EpisodeBundleWriter | SkippedEpisode':
        validate_common_contract_fixture()
        _validate_contexts(run_context, episode_context, capture_config)
        trajectory_root = Path(capture_config.root).expanduser().absolute()
        run_root = trajectory_root / run_context.run_id
        _validate_published_run(run_root, run_context)
        episodes_root = run_root / 'episodes'
        episodes_root.mkdir(exist_ok=True)
        final_root = episodes_root / episode_context.episode_id
        partial_root = episodes_root / (episode_context.episode_id + '.partial')
        if final_root.exists():
            manifest_path = final_root / 'manifest.json'
            try:
                manifest, manifest_hash = _validate_complete_bundle(final_root, expected_episode_id=episode_context.episode_id, expected_episode_key=episode_context.episode_key, expected_task_hash=episode_context.task.identity.task_hash, expected_run_id=run_context.run_id)
                _validate_complete_manifest_context(manifest, bundle_root=final_root, run_context=run_context, episode_context=episode_context)
            except BundleConflictError:
                raise
            except Exception as exc:
                raise BundleConflictError('episode_conflict', 'existing final bundle is invalid and will not be overwritten: {}'.format(exc)) from exc
            if not resume:
                raise BundleConflictError('episode_exists', 'complete episode already exists')
            _ensure_run_index_row(run_root, manifest=manifest, manifest_hash=manifest_hash, status='repaired_index', fsync=capture_config.fsync)
            return SkippedEpisode(episode_id=episode_context.episode_id, episode_key=episode_context.episode_key, manifest_path=manifest_path, manifest_sha256=manifest_hash, reason='skipped_complete')
        if partial_root.exists():
            if not resume:
                raise BundleConflictError('incomplete_episode_exists', 'partial episode exists; resume is required')
            _archive_partial(run_root, partial_root, episode_context.episode_id)
        try:
            partial_root.mkdir(mode=493)
        except FileExistsError as exc:
            raise BundleConflictError('incomplete_episode_exists', 'partial episode was created concurrently') from exc
        writer = cls(run_context=run_context, episode_context=episode_context, capture_config=capture_config, run_root=run_root, partial_root=partial_root)
        try:
            writer._initialize_partial()
        except Exception:
            writer._close_handles()
            raise
        return writer

    def __enter__(self) -> 'EpisodeBundleWriter':
        self._require_state('OPEN')
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        if self.state not in ('COMPLETE', 'ABORTED'):
            error = {'error_class': 'infra_error', 'error_code': 'writer_context_unfinalized' if exc is None else 'writer_context_exception', 'message': None if exc is None else type(exc).__name__}
            self.abort(error)
        return False

    def append_event(self, envelope: Mapping[str, Any], *, public_payload: Mapping[str, Any] | None, privileged_payload: Mapping[str, Any] | None) -> str:
        self._require_state('OPEN')
        if not isinstance(envelope, Mapping):
            raise BundleIntegrityError('capture_serialization_error', 'event envelope must be a mapping')
        public = None if public_payload is None else copy.deepcopy(dict(public_payload))
        privileged = None if privileged_payload is None else copy.deepcopy(dict(privileged_payload))
        record_type = envelope.get('record_type')
        if public is None or privileged is None:
            raise BundleIntegrityError('sidecar_payload_required', 'every v1 trajectory record requires public and privileged payloads')
        if public is not None:
            validate_event_payload(record_type, public, privacy='public')
            assert_no_public_leakage(public, root_pointer='/payload', forbidden_absolute_prefixes=(self.run_root, Path.home()))
        if privileged is not None:
            validate_event_payload(record_type, privileged, privacy='privileged')
        record = dict(envelope)
        next_sequence = self._sequence_index + 1
        record.setdefault('schema_version', SCHEMA_VERSIONS['event_index'])
        record.setdefault('record_id', make_record_id(next_sequence))
        record.setdefault('episode_id', self.episode_context.episode_id)
        record.setdefault('sequence_index', next_sequence)
        record.setdefault('step_id', 0)
        for key in ('decision_id', 'action_id', 'low_level_id', 'stream_id', 'stream_obs_id', 'intervention_id', 'model_call_id'):
            record.setdefault(key, None)
        record.setdefault('logical_time_s', self._last_logical_time)
        record.setdefault('monotonic_time_ns', self._last_monotonic_time)
        record.setdefault('wall_time_utc', _utc_now())
        record.setdefault('public_payload', public is not None)
        record.setdefault('privileged_payload', privileged is not None)
        validate_trajectory_record(record)
        if record['sequence_index'] != next_sequence or record['record_id'] != make_record_id(next_sequence):
            raise BundleIntegrityError('record_sequence_invalid', 'writer record id and sequence must be the next ordinal')
        if record['episode_id'] != self.episode_context.episode_id:
            raise BundleIntegrityError('record_episode_mismatch', 'event belongs to another episode')
        if record['public_payload'] is not (public is not None) or record['privileged_payload'] is not (privileged is not None):
            raise BundleIntegrityError('sidecar_presence_mismatch', 'payload booleans must match provided sidecars')
        if record['logical_time_s'] < self._last_logical_time or record['monotonic_time_ns'] < self._last_monotonic_time:
            raise BundleIntegrityError('record_time_regression', 'logical and monotonic times must not regress')
        record_type = record['record_type']
        if next_sequence == 1 and record_type != 'episode_start':
            raise BundleIntegrityError('record_order_invalid', 'first record must be episode_start')
        if self._event_types and self._event_types[-1] == 'episode_end':
            raise BundleIntegrityError('record_order_invalid', 'episode_end must be the final record')
        if record_type == 'episode_start' and next_sequence != 1:
            raise BundleIntegrityError('record_order_invalid', 'episode_start may occur only once')
        public_record = None
        private_record = None
        if public is not None:
            public_record = {'schema_version': SCHEMA_VERSIONS['payload'], 'record_id': record['record_id'], 'record_type': record_type, 'payload': public}
            validate_payload_record(public_record)
        if privileged is not None:
            private_record = {'schema_version': SCHEMA_VERSIONS['payload'], 'record_id': record['record_id'], 'record_type': record_type, 'payload': privileged}
            validate_payload_record(private_record)
        self._append_jsonl('trajectory.jsonl', record)
        if public_record is not None:
            self._append_jsonl('public/trajectory.jsonl', public_record)
        if private_record is not None:
            self._append_jsonl('privileged/trajectory.jsonl', private_record)
        self._sequence_index = next_sequence
        self._last_logical_time = float(record['logical_time_s'])
        self._last_monotonic_time = record['monotonic_time_ns']
        self._event_types.append(record_type)
        if record_type in ('action_start', 'action_end', 'intervention', 'episode_end'):
            self._flush_all()
            self._write_capture_state(status='open')
        return record['record_id']

    def write_frame(self, rgb: Any, metadata: FrameMetadata, privileged_annotation: Mapping[str, Any] | None=None) -> str:
        self._require_state('OPEN')
        if not isinstance(metadata, FrameMetadata):
            raise BundleIntegrityError('frame_metadata_invalid', 'metadata must be FrameMetadata')
        if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2:] != (3,) or (rgb.dtype != np.uint8):
            raise BundleIntegrityError('frame_serialization_error', 'writer only accepts HWC RGB uint8 arrays')
        encoding = self.capture_config.stream_encoding if metadata.frame_kind == 'stream_sample' else self.capture_config.keyframe_encoding
        buffer = io.BytesIO()
        image = Image.fromarray(rgb, mode='RGB')
        if encoding == 'png':
            image.save(buffer, format='PNG', compress_level=6, optimize=False)
            mime_type = 'image/png'
            encoding_parameters = {'compress_level': 6, 'optimize': False}
        elif encoding == 'jpeg':
            image.save(buffer, format='JPEG', quality=self.capture_config.stream_jpeg_quality, subsampling=0, optimize=False, progressive=False)
            mime_type = 'image/jpeg'
            encoding_parameters = {'quality': self.capture_config.stream_jpeg_quality, 'subsampling': 0, 'optimize': False, 'progressive': False}
        else:
            raise BundleIntegrityError('frame_encoding_invalid', 'unsupported frame encoding')
        encoded = buffer.getvalue()
        encoded_quality, decoded = analyze_encoded_image(encoded)
        if decoded['mode'] != 'RGB':
            raise BundleIntegrityError('frame_encoding_invalid', 'encoded frame did not preserve RGB mode')
        return self._write_frame_bytes(encoded, metadata=metadata, mime_type=mime_type, encoding=encoding, encoding_parameters=encoding_parameters, quality=encoded_quality, privileged_annotation=privileged_annotation)

    def write_model_image(self, encoded: bytes, *, mime_type: str, source_frame_id: str, metadata: FrameMetadata) -> str:
        self._require_state('OPEN')
        if source_frame_id not in self._frame_ids:
            raise BundleIntegrityError('dangling_frame_reference', 'model image source_frame_id does not exist')
        if not isinstance(metadata, FrameMetadata) or metadata.frame_kind != 'model_input':
            raise BundleIntegrityError('frame_metadata_invalid', 'model images require frame_kind=model_input')
        if mime_type not in ('image/png', 'image/jpeg'):
            raise BundleIntegrityError('frame_encoding_invalid', 'model image MIME type is unsupported')
        quality, decoded = analyze_encoded_image(encoded)
        expected_format = 'PNG' if mime_type == 'image/png' else 'JPEG'
        if decoded['format'] != expected_format:
            raise BundleIntegrityError('frame_encoding_mismatch', 'encoded image format differs from MIME type')
        capture = dict(metadata.capture)
        if capture.get('source_frame_id') not in (None, source_frame_id):
            raise BundleIntegrityError('frame_source_conflict', 'metadata source_frame_id differs from argument')
        capture['source_frame_id'] = source_frame_id
        adjusted = FrameMetadata(frame_kind=metadata.frame_kind, step_id=metadata.step_id, action_id=metadata.action_id, stream_id=metadata.stream_id, stream_obs_id=metadata.stream_obs_id, logical_time_s=metadata.logical_time_s, monotonic_time_ns=metadata.monotonic_time_ns, wall_time_utc=metadata.wall_time_utc, source=metadata.source, camera=metadata.camera, capture=capture)
        frame = self._write_frame_bytes(encoded, metadata=adjusted, mime_type=mime_type, encoding='png' if mime_type == 'image/png' else 'jpeg', encoding_parameters={'exact_model_input': True}, quality=quality, privileged_annotation=None)
        self._model_frame_ids.add(frame)
        return frame

    def write_state_snapshot(self, snapshot: Mapping[str, Any]) -> str:
        self._require_state('OPEN')
        if not isinstance(snapshot, Mapping):
            raise BundleIntegrityError('capture_serialization_error', 'snapshot must be a mapping')
        self._state_ordinal += 1
        expected = make_state_snapshot_id(self._state_ordinal)
        record = copy.deepcopy(dict(snapshot))
        record.setdefault('schema_version', STATE_SCHEMA_VERSION)
        record.setdefault('snapshot_id', expected)
        record.setdefault('episode_id', self.episode_context.episode_id)
        if record['snapshot_id'] != expected or record['episode_id'] != self.episode_context.episode_id:
            raise BundleIntegrityError('state_identity_invalid', 'snapshot identity or ordinal is invalid')
        validate_json_value(record)
        self._append_jsonl('privileged/state_snapshots.jsonl', record)
        self._state_ids.add(expected)
        return expected

    def write_model_call(self, record: Mapping[str, Any]) -> str:
        self._require_state('OPEN')
        if not isinstance(record, Mapping):
            raise BundleIntegrityError('capture_serialization_error', 'model call must be a mapping')
        self._model_call_ordinal += 1
        expected = make_model_call_id(self._model_call_ordinal)
        value = copy.deepcopy(dict(record))
        value.setdefault('schema_version', SCHEMA_VERSIONS['model_io'])
        value.setdefault('model_call_id', expected)
        value.setdefault('episode_id', self.episode_context.episode_id)
        if value['model_call_id'] != expected or value['episode_id'] != self.episode_context.episode_id:
            raise BundleIntegrityError('model_call_identity_invalid', 'model call identity is invalid')
        assert_no_public_leakage(value, root_pointer='/model_io', forbidden_absolute_prefixes=(self.run_root, Path.home()))
        image_ids = value.get('request', {}).get('image_frame_ids', []) if isinstance(value.get('request'), dict) else []
        if not isinstance(image_ids, list) or any((frame not in self._model_frame_ids for frame in image_ids)):
            raise BundleIntegrityError('model_input_frame_invalid', 'model calls may reference only saved model_input frames')
        self._append_jsonl('public/model_io.jsonl', value)
        self._model_call_ids.add(expected)
        return expected

    def write_execution_audit(self, record: Mapping[str, Any]) -> str:
        self._require_state('OPEN')
        if not isinstance(record, Mapping):
            raise BundleIntegrityError('capture_serialization_error', 'audit must be a mapping')
        self._audit_ordinal += 1
        expected = make_audit_id(self._audit_ordinal)
        value = copy.deepcopy(dict(record))
        value.setdefault('schema_version', EXECUTION_AUDIT_SCHEMA_VERSION)
        value.setdefault('audit_id', expected)
        value.setdefault('episode_id', self.episode_context.episode_id)
        value.setdefault('sequence_index', self._audit_ordinal)
        if value['audit_id'] != expected or value['episode_id'] != self.episode_context.episode_id:
            raise BundleIntegrityError('audit_identity_invalid', 'execution audit identity is invalid')
        validate_json_value(value)
        self._append_jsonl('privileged/execution_audit.jsonl', value)
        return expected

    def write_admission_preflight(self, value: Mapping[str, Any]) -> Path:
        self._require_state('OPEN')
        from scoper.benchmark.procthor.benchmark.admission_schema import validate_reachability_preflight
        document = copy.deepcopy(dict(value))
        validate_reachability_preflight(document)
        if document['cell']['task_hash'] != self.episode_context.task.identity.task_hash:
            raise BundleIntegrityError('admission_preflight_identity_invalid', 'preflight task identity differs from episode')
        path = self.partial_root / 'privileged/admission_preflight.json'
        if path.exists():
            raise BundleConflictError('admission_preflight_conflict', 'admission preflight evidence was already written')
        _write_atomic_json(path, document, fsync=self.capture_config.fsync)
        return path

    def finalize(self, terminal: Mapping[str, Any]) -> Path:
        self._require_state('OPEN')
        self.state = 'FINALIZING'
        try:
            if not self._event_types or self._event_types[0] != 'episode_start':
                raise BundleIntegrityError('record_order_invalid', 'episode_start is missing')
            if self._event_types[-1] != 'episode_end':
                raise BundleIntegrityError('record_order_invalid', 'episode_end is missing')
            if self._event_types.count('episode_start') != 1 or self._event_types.count('episode_end') != 1:
                raise BundleIntegrityError('record_cardinality_invalid', 'episode boundaries must occur exactly once')
            validate_json_value(dict(terminal), json_pointer='/terminal')
            self._flush_all()
            self._close_handles()
            phase_a = self._validate_phase_a(terminal)
            quality_report = phase_a['quality_report']
            _write_atomic_json(self.partial_root / 'quality_report.json', quality_report, fsync=self.capture_config.fsync)
            if not quality_report['structural_valid']:
                raise BundleIntegrityError('bundle_structural_invalid', 'phase-A structural validation failed')
            self._remove_capture_state()
            artifacts = _build_artifact_table(self.partial_root)
            terminal_copy = copy.deepcopy(dict(terminal))
            _validate_terminal(terminal_copy)
            manifest = self._build_manifest(terminal=terminal_copy, artifacts=artifacts, quality_report=quality_report)
            manifest_bytes = canonical_json_bytes(manifest) + b'\n'
            self._validate_phase_b(manifest, manifest_bytes)
            _write_atomic_bytes(self.partial_root / 'manifest.json', manifest_bytes, fsync=self.capture_config.fsync)
            _fsync_directory(self.partial_root, enabled=self.capture_config.fsync)
            os.replace(self.partial_root, self.final_root)
            _fsync_directory(self.final_root.parent, enabled=self.capture_config.fsync)
            _make_episode_read_only(self.final_root)
            manifest_hash = sha256_bytes(manifest_bytes)
            _ensure_run_index_row(self.run_root, manifest=manifest, manifest_hash=manifest_hash, status='finalized', fsync=self.capture_config.fsync)
            self.state = 'COMPLETE'
            return self.final_root / 'manifest.json'
        except Exception as exc:
            self._close_handles()
            if self.partial_root.exists():
                try:
                    (self.partial_root / 'manifest.json').unlink(missing_ok=True)
                    (self.partial_root / 'manifest.json.tmp').unlink(missing_ok=True)
                    self._write_capture_state(status='aborted', error={'error_class': 'infra_error', 'error_code': getattr(exc, 'code', 'bundle_finalize_error'), 'message': type(exc).__name__})
                except Exception:
                    pass
            self.state = 'ABORTED'
            raise

    def abort(self, error: Mapping[str, Any]) -> None:
        if self.state in ('COMPLETE', 'ABORTED'):
            return
        try:
            validate_json_value(dict(error), json_pointer='/error')
        except Exception:
            error = {'error_class': 'infra_error', 'error_code': 'abort_error_not_serializable', 'message': None}
        try:
            self._flush_all()
        except Exception:
            pass
        self._close_handles()
        try:
            self._write_capture_state(status='aborted', error=dict(error))
        finally:
            self.state = 'ABORTED'

    def _initialize_partial(self) -> None:
        for directory in ('public/frames', 'public/model_images', 'privileged/snapshots', 'provenance'):
            (self.partial_root / directory).mkdir(parents=True, exist_ok=True)
        for relative in self._JSONL_PATHS:
            path = self.partial_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            self._handles[relative] = path.open('xb')
        self._copy_static_documents()
        self._write_capture_state(status='open')
        _fsync_directory(self.partial_root, enabled=self.capture_config.fsync)

    def _copy_static_documents(self) -> None:
        identity = self.episode_context.task.identity
        identity_task = copy.deepcopy(dict(identity.identity_task))
        task_id = identity_task['id']
        public_task = {'schema_version': SCHEMA_VERSIONS['task_snapshot'], 'privacy': 'public', 'task_id': task_id, 'instruction': identity_task['instruction'], 'public_task_metadata': {'category': identity_task['category'], 'template_id': identity_task['template']['template_id']}, 'tool_schema_version': self.run_context.protocol.get('tool_schema', {}).get('version')}
        assert_no_public_leakage(public_task, root_pointer='/public_task')
        private_task = {'schema_version': SCHEMA_VERSIONS['task_snapshot'], 'privacy': 'privileged', 'task_id': task_id, 'task_hash': identity.task_hash, 'full_task_spec': identity_task, 'task_path_resolutions': [asdict(item) for item in identity.path_resolutions]}
        if sha256_json(private_task['full_task_spec']) != identity.task_hash:
            raise BundleIntegrityError('task_hash_mismatch', 'identity task hash changed')
        _write_atomic_json(self.partial_root / 'public/task.json', public_task, fsync=self.capture_config.fsync)
        _write_atomic_json(self.partial_root / 'privileged/task.json', private_task, fsync=self.capture_config.fsync)
        documents = {'protocol': self.run_context.protocol, 'policy': self.run_context.policy, 'capture_config': self.run_context.capture_config_document}
        for name, document in documents.items():
            expected_hash = self.run_context.manifest['{}_hash'.format(name)]
            if sha256_json(document) != expected_hash:
                raise BundleIntegrityError('provenance_hash_mismatch', '{} content differs from run manifest'.format(name))
            value = {'schema_version': SCHEMA_VERSIONS['provenance'], 'document_type': name, 'content_hash': expected_hash, 'content': copy.deepcopy(dict(document))}
            validate_provenance_document(value, name)
            document_bytes = canonical_json_bytes(value) + b'\n'
            if sha256_bytes(document_bytes) != self.run_context.manifest['provenance_documents'][name]['sha256']:
                raise BundleIntegrityError('provenance_hash_mismatch', '{} envelope bytes differ from run manifest'.format(name))
            _write_atomic_json(self.partial_root / 'provenance' / (name + '.json'), value, fsync=self.capture_config.fsync)

    def _append_jsonl(self, relative: str, value: Mapping[str, Any]) -> None:
        validate_json_value(dict(value))
        handle = self._handles.get(relative)
        if handle is None or handle.closed:
            raise BundleIntegrityError('writer_handle_closed', 'JSONL handle is not open')
        handle.write(canonical_json_bytes(dict(value)) + b'\n')

    def _flush_all(self) -> None:
        for handle in self._handles.values():
            if handle.closed:
                continue
            handle.flush()
            if self.capture_config.fsync:
                os.fsync(handle.fileno())

    def _close_handles(self) -> None:
        for handle in self._handles.values():
            if not handle.closed:
                try:
                    handle.close()
                except OSError:
                    pass

    def _write_capture_state(self, *, status: str, error: Mapping[str, Any] | None=None) -> None:
        if not self.partial_root.exists():
            return
        value = {'schema_version': 'trajectory-capture-state.v1', 'status': status, 'episode_id': self.episode_context.episode_id, 'episode_key': self.episode_context.episode_key, 'sequence_index': self._sequence_index, 'frame_ordinal': self._frame_ordinal, 'state_ordinal': self._state_ordinal, 'model_call_ordinal': self._model_call_ordinal, 'audit_ordinal': self._audit_ordinal, 'error': None if error is None else copy.deepcopy(dict(error))}
        _write_atomic_json(self.partial_root / 'capture_state.json', value, fsync=self.capture_config.fsync)

    def _remove_capture_state(self) -> None:
        path = self.partial_root / 'capture_state.json'
        path.unlink()
        _fsync_directory(self.partial_root, enabled=self.capture_config.fsync)

    def _write_frame_bytes(self, encoded: bytes, *, metadata: FrameMetadata, mime_type: str, encoding: str, encoding_parameters: Mapping[str, Any], quality: Mapping[str, Any], privileged_annotation: Mapping[str, Any] | None) -> str:
        if not isinstance(encoded, bytes) or not encoded:
            raise BundleIntegrityError('frame_serialization_error', 'encoded frame is empty')
        if RFC3339_UTC_RE.fullmatch(metadata.wall_time_utc) is None:
            raise BundleIntegrityError('frame_metadata_invalid', 'wall_time_utc is invalid')
        if metadata.logical_time_s < 0 or metadata.monotonic_time_ns < 0:
            raise BundleIntegrityError('frame_metadata_invalid', 'frame times must be non-negative')
        validate_json_value(dict(metadata.camera), json_pointer='/camera')
        validate_json_value(dict(metadata.capture), json_pointer='/capture')
        self._frame_ordinal += 1
        frame = make_frame_id(self._frame_ordinal)
        suffix = '.png' if encoding == 'png' else '.jpg'
        directory = 'public/model_images' if metadata.frame_kind == 'model_input' else 'public/frames'
        relative_path = '{}/{}{}'.format(directory, frame, suffix)
        destination = self.partial_root / relative_path
        _write_atomic_bytes(destination, encoded, fsync=self.capture_config.fsync)
        content_hash = sha256_bytes(encoded)
        duplicate = content_hash in self._frame_hashes
        self._frame_hashes.add(content_hash)
        frame_quality = copy.deepcopy(dict(quality))
        if duplicate:
            frame_quality.setdefault('warnings', []).append('exact_duplicate')
            frame_quality['warnings'] = sorted(set(frame_quality['warnings']))
        shape = frame_quality.get('shape')
        if not isinstance(shape, list) or len(shape) != 3:
            width = height = None
        else:
            height, width, _ = shape
        record = {'schema_version': SCHEMA_VERSIONS['frame'], 'frame_id': frame, 'episode_id': self.episode_context.episode_id, 'sequence_index': self._frame_ordinal, 'step_id': metadata.step_id, 'action_id': metadata.action_id, 'stream_id': metadata.stream_id, 'stream_obs_id': metadata.stream_obs_id, 'event_ordinal': dict(metadata.capture).get('event_ordinal', 0), 'frame_kind': metadata.frame_kind, 'source': metadata.source, 'logical_time_s': metadata.logical_time_s, 'monotonic_time_ns': metadata.monotonic_time_ns, 'wall_time_utc': metadata.wall_time_utc, 'path': relative_path, 'sha256': content_hash, 'bytes': len(encoded), 'width': width, 'height': height, 'channels': 3, 'dtype': 'uint8', 'color_space': 'sRGB', 'encoding': encoding, 'encoding_parameters': copy.deepcopy(dict(encoding_parameters)), 'camera': copy.deepcopy(dict(metadata.camera)), 'capture': copy.deepcopy(dict(metadata.capture)), 'public_visibility_summary': None, 'quality': frame_quality}
        assert_no_public_leakage(record, root_pointer='/frame', forbidden_absolute_prefixes=(self.run_root, Path.home()))
        self._append_jsonl('public/frames.jsonl', record)
        self._frame_ids.add(frame)
        if privileged_annotation is not None:
            annotation = copy.deepcopy(dict(privileged_annotation))
            annotation.setdefault('schema_version', 'frame-annotation.v1')
            annotation.setdefault('frame_id', frame)
            annotation.setdefault('episode_id', self.episode_context.episode_id)
            annotation.setdefault('event_ordinal', record['event_ordinal'])
            if annotation['frame_id'] != frame or annotation['episode_id'] != self.episode_context.episode_id:
                raise BundleIntegrityError('frame_annotation_invalid', 'annotation identity mismatch')
            self._append_jsonl('privileged/frame_annotations.jsonl', annotation)
        return frame

    def _validate_phase_a(self, terminal: Mapping[str, Any]) -> dict[str, Any]:
        structural_errors: list[dict[str, Any]] = []
        quality_errors: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        checks = {'records': {'valid': True, 'errors': []}, 'references': {'valid': True, 'errors': []}, 'frames': {'valid': True, 'errors': []}, 'leakage': {'valid': True, 'errors': []}, 'effects': {'valid': True, 'errors': []}, 'split': {'valid': True, 'errors': []}, 'hashes': {'valid': True, 'errors': []}, 'native_profile': {'valid': True, 'errors': []}, 'audit_completeness': {'valid': True, 'errors': []}}
        try:
            index_records = load_jsonl_strict(self.partial_root / 'trajectory.jsonl')
            public_records = load_jsonl_strict(self.partial_root / 'public/trajectory.jsonl')
            privileged_records = load_jsonl_strict(self.partial_root / 'privileged/trajectory.jsonl')
            for record in index_records:
                if not isinstance(record, dict):
                    raise BundleIntegrityError('trajectory_record_schema_invalid', 'trajectory record must be an object')
                validate_trajectory_record(record)
            for record in public_records + privileged_records:
                if not isinstance(record, dict):
                    raise BundleIntegrityError('trajectory_payload_schema_invalid', 'payload record must be an object')
                validate_payload_record(record)
            for record in public_records:
                validate_event_payload(record['record_type'], record['payload'], privacy='public')
            for record in privileged_records:
                validate_event_payload(record['record_type'], record['payload'], privacy='privileged')
            _validate_record_sequence(index_records)
        except Exception as exc:
            _add_check_error(checks, structural_errors, 'records', exc)
            index_records = []
            public_records = []
            privileged_records = []
        try:
            findings = audit_public_private_sidecars(index_records, public_records, privileged_records, forbidden_absolute_prefixes=(self.run_root, Path.home()))
            for relative in ('public/task.json', 'public/frames.jsonl', 'public/model_io.jsonl'):
                path = self.partial_root / relative
                if path.suffix == '.jsonl':
                    values = load_jsonl_strict(path)
                else:
                    values = [load_json_strict(path)]
                for index, value in enumerate(values):
                    findings += audit_public_value(value, root_pointer='/{}/{}'.format(relative, index), forbidden_absolute_prefixes=(self.run_root, Path.home()))
            if findings:
                for finding in findings:
                    exc = LeakageError(finding)
                    _add_check_error(checks, structural_errors, 'leakage', exc)
        except Exception as exc:
            _add_check_error(checks, structural_errors, 'leakage', exc)
        frame_records: list[dict[str, Any]] = []
        try:
            loaded_frames = load_jsonl_strict(self.partial_root / 'public/frames.jsonl')
            previous_frame_sequence = 0
            stream_total = 0
            stream_invalid = 0
            for record in loaded_frames:
                if not isinstance(record, dict):
                    raise BundleIntegrityError('frame_record_invalid', 'frame record must be an object')
                if record.get('schema_version') != SCHEMA_VERSIONS['frame']:
                    raise BundleIntegrityError('frame_record_invalid', 'frame schema version is invalid')
                sequence = record.get('sequence_index')
                if not isinstance(sequence, int) or sequence != previous_frame_sequence + 1:
                    raise BundleIntegrityError('frame_record_invalid', 'frame sequence is not contiguous')
                previous_frame_sequence = sequence
                relative = record.get('path')
                if not isinstance(relative, str) or Path(relative).is_absolute() or '..' in Path(relative).parts:
                    raise BundleIntegrityError('frame_path_invalid', 'frame path is unsafe')
                path = self.partial_root / relative
                encoded = path.read_bytes()
                if len(encoded) != record.get('bytes') or sha256_bytes(encoded) != record.get('sha256'):
                    raise BundleIntegrityError('frame_hash_mismatch', 'frame bytes or SHA256 mismatch')
                decoded_quality, _ = analyze_encoded_image(encoded)
                recorded_quality = record.get('quality')
                if not isinstance(recorded_quality, dict):
                    raise BundleIntegrityError('frame_quality_invalid', 'frame quality is missing')
                for key in ('valid', 'shape', 'dtype', 'dynamic_range', 'luminance_mean', 'luminance_std', 'gradient', 'blur_score', 'errors'):
                    if recorded_quality.get(key) != decoded_quality.get(key):
                        raise BundleIntegrityError('frame_quality_mismatch', 'recorded quality differs from decoded bytes')
                if record.get('frame_kind') == 'stream_sample':
                    stream_total += 1
                    if not decoded_quality['valid']:
                        stream_invalid += 1
                if not decoded_quality['valid'] and is_critical_frame_kind(record.get('frame_kind')):
                    quality_errors.append({'error_code': 'critical_frame_invalid', 'frame_id': record.get('frame_id'), 'details': decoded_quality['errors']})
                for warning in recorded_quality.get('warnings', []):
                    warnings.append({'warning_code': warning, 'frame_id': record.get('frame_id')})
                frame_records.append(record)
            if stream_total and stream_invalid / stream_total > 0.01:
                quality_errors.append({'error_code': 'stream_invalid_ratio_exceeded', 'invalid': stream_invalid, 'total': stream_total})
        except Exception as exc:
            _add_check_error(checks, structural_errors, 'frames', exc)
        try:
            task_document = load_json_strict(self.partial_root / 'privileged/task.json')
            if not isinstance(task_document, dict):
                raise BundleIntegrityError('task_snapshot_invalid', 'private task must be an object')
            if sha256_json(task_document.get('full_task_spec')) != task_document.get('task_hash'):
                raise BundleIntegrityError('task_hash_mismatch', 'task snapshot hash is invalid')
            if task_document.get('task_hash') != self.episode_context.task.identity.task_hash:
                raise BundleIntegrityError('task_hash_mismatch', 'task snapshot differs from context')
            for name in ('protocol', 'policy', 'capture_config'):
                document = load_json_strict(self.partial_root / 'provenance' / (name + '.json'))
                if not isinstance(document, dict):
                    raise BundleIntegrityError('provenance_schema_invalid', 'provenance must be object')
                validate_provenance_document(document, name)
                if document['content_hash'] != self.run_context.manifest['{}_hash'.format(name)]:
                    raise BundleIntegrityError('provenance_hash_mismatch', 'provenance differs from run')
                if sha256_bytes((self.partial_root / 'provenance' / (name + '.json')).read_bytes()) != self.run_context.manifest['provenance_documents'][name]['sha256']:
                    raise BundleIntegrityError('provenance_hash_mismatch', 'provenance envelope bytes differ from run')
        except Exception as exc:
            _add_check_error(checks, structural_errors, 'hashes', exc)
        try:
            payload = self.episode_context.episode_key_payload
            identity_task = self.episode_context.task.identity.identity_task
            if payload['source_split'] != self.episode_context.source_split:
                raise BundleIntegrityError('split_metadata_conflict', 'episode source split mismatch')
            if identity_task['source']['split'] != self.episode_context.source_split:
                raise BundleIntegrityError('split_metadata_conflict', 'task source split mismatch')
            if self.run_context.manifest['task_split'] != self.episode_context.task_split:
                raise BundleIntegrityError('split_metadata_conflict', 'task split mismatch')
        except Exception as exc:
            _add_check_error(checks, structural_errors, 'split', exc)
        try:
            _validate_references(index_records, frame_records, self.partial_root)
        except Exception as exc:
            _add_check_error(checks, structural_errors, 'references', exc)
        try:
            audit_records = load_jsonl_strict(self.partial_root / 'privileged/execution_audit.jsonl')
            validate_execution_audit_records(audit_records, execution_profile=self.episode_context.execution_profile, allowed_agent_fallbacks=tuple(self.run_context.capture_config_document.get('allowed_agent_fallbacks', ())))
        except NativeProfileViolation as exc:
            _add_check_error(checks, quality_errors, 'native_profile', exc)
        except Exception as exc:
            _add_check_error(checks, structural_errors, 'audit_completeness', exc)
        quality_valid = not quality_errors
        structural_valid = not structural_errors
        terminal_error_class = terminal.get('error_class', 'none')
        exclusion_reasons = _derive_exclusion_reasons(episode_context=self.episode_context, run_context=self.run_context, quality_valid=quality_valid, terminal_error_class=terminal_error_class)
        exporter_eligible = structural_valid and quality_valid and (not exclusion_reasons)
        quality_report = {'schema_version': SCHEMA_VERSIONS['quality'], 'episode_id': self.episode_context.episode_id, 'structural_valid': structural_valid, 'quality_valid': quality_valid, 'capture_capabilities': {'profile': self.capture_config.profile, 'stream_export_eligible': self.capture_config.profile in ('training', 'debug'), 'full_world_boundaries': self.capture_config.full_world_boundaries}, 'checks': checks, 'errors': structural_errors + quality_errors, 'warnings': warnings, 'exporter_eligibility': {'step_decision': exporter_eligible, 'stream_decision': exporter_eligible and self.capture_config.profile in ('training', 'debug'), 'verification_recovery': exporter_eligible, 'preference': exporter_eligible, 'memory_update': exporter_eligible, 'memory_reconcile': exporter_eligible, 'memory_route': exporter_eligible}, 'exclusion_reasons': exclusion_reasons}
        validate_quality_report(quality_report)
        return {'quality_report': quality_report, 'frame_records': frame_records}

    def _build_manifest(self, *, terminal: Mapping[str, Any], artifacts: list[dict[str, Any]], quality_report: Mapping[str, Any]) -> dict[str, Any]:
        payload = copy.deepcopy(dict(self.episode_context.episode_key_payload))
        task_identity = self.episode_context.task.identity.identity_task
        resolution = self.episode_context.task.identity.path_resolutions[0]
        error_class = terminal.get('error_class', 'none')
        if error_class not in ('none', 'agent_failure', 'invalid_action', 'budget_exceeded', 'infra_error', 'task_invalid', 'evaluator_invalid'):
            raise BundleIntegrityError('terminal_schema_invalid', 'terminal error_class is invalid')
        exclusions = _derive_exclusion_reasons(episode_context=self.episode_context, run_context=self.run_context, quality_valid=bool(quality_report['quality_valid']), terminal_error_class=error_class)
        train_eligible = not exclusions
        admission_eligible = _candidate_admission_eligible(episode_context=self.episode_context, terminal=terminal, train_eligible=train_eligible)
        task = {'id': task_identity['id'], 'hash': self.episode_context.task.identity.task_hash, 'family': task_identity['category'], 'template_id': task_identity['template']['template_id'], 'source_split': self.episode_context.source_split, 'task_split': self.episode_context.task_split, 'lineage_id': self.run_context.manifest['task_lineage_id'], 'task_bank_hash': self.run_context.manifest['task_bank_hash'], 'task_set_hash': self.run_context.manifest['task_set_hash']}
        dataset_version = '{}@{}'.format(resolution.dataset_name, resolution.dataset_revision) if resolution.locator_kind == 'prior_json' else self.run_context.protocol.get('assets', {}).get('local_house_registry_version')
        artifact_paths = {artifact['path'] for artifact in artifacts}
        model_io_records = load_jsonl_strict(self.partial_root / 'public/model_io.jsonl')
        counts = {'records': _artifact_record_count(artifacts, 'trajectory.jsonl'), 'actions': self._event_types.count('action_start'), 'low_level_actions': self._event_types.count('low_level_action'), 'stream_observations': self._event_types.count('stream_observation'), 'frames': _artifact_record_count(artifacts, 'public/frames.jsonl'), 'interventions': self._event_types.count('intervention'), 'model_calls': _artifact_record_count(artifacts, 'public/model_io.jsonl'), 'prompt_tokens': _sum_model_tokens(model_io_records, 'prompt_tokens'), 'completion_tokens': _sum_model_tokens(model_io_records, 'completion_tokens')}
        ended = _utc_now()
        elapsed = max(0.0, (time.monotonic_ns() - self._started_monotonic_ns) / 1000000000.0)
        seeds = {'base': payload['base_seed']}
        seed_consumption = terminal.get('seed_consumption', {})
        for namespace in SEED_NAMESPACES:
            detail = seed_consumption.get(namespace)
            if detail is None:
                detail = {'consumed': False, 'consumer': None, 'reason': None}
            seeds[namespace] = {'value': self.episode_context.seeds[namespace], 'consumed': detail['consumed'], 'consumer': detail['consumer'], 'reason': detail['reason']}
        manifest = {'schema_version': SCHEMA_VERSIONS['episode_manifest'], 'contract_version': CONTRACT_VERSION, 'bundle_status': 'complete', 'backend': BACKEND, 'run_id': self.run_context.run_id, 'episode_id': self.episode_context.episode_id, 'episode_key': self.episode_context.episode_key, 'episode_key_payload': payload, 'task': task, 'scene': {'scene_id': payload['scene_id'], 'layout_id': payload['layout_id'], 'house_path_hash': sha256_bytes(resolution.normalized_locator.encode('utf-8')), 'house_content_sha256': resolution.content_sha256, 'dataset_version': dataset_version, 'asset_version': self.run_context.protocol.get('assets', {}).get('asset_version')}, 'rollout': {'purpose': self.episode_context.rollout_purpose, 'arm': payload['rollout_arm'], 'execution_profile': self.episode_context.execution_profile, 'repeat_id': payload['repeat_id']}, 'eligibility': {'train_eligible': train_eligible, 'admission_eligible': admission_eligible, 'exclusion_reasons': exclusions}, 'seeds': seeds, 'versions': {'common_contract': {'schema_version': COMMON_CONTRACT_VERSION, 'sha256': COMMON_CONTRACT_SHA256}, 'protocol': self.run_context.manifest['protocol_hash'], 'tool_schema': self.run_context.protocol.get('tool_schema', {}).get('version'), 'policy': self.run_context.manifest['policy_hash'], 'capture': self.run_context.manifest['capture_config_hash'], 'code': self.run_context.manifest.get('code_commit'), 'ai2thor': self.run_context.manifest['runtime_contract'].get('ai2thor_version'), 'procthor': self.run_context.manifest['runtime_contract'].get('procthor_version')}, 'timing': {'started_at_utc': self._started_at_utc, 'ended_at_utc': ended, 'elapsed_seconds': elapsed}, 'counts': counts, 'terminal': {'episode_status': terminal.get('episode_status', 'invalid'), 'done_reason': terminal.get('done_reason'), 'error_class': error_class, 'error_code': terminal.get('error_code'), 'native_task_success': terminal.get('native_task_success'), 'compatibility_task_success': terminal.get('compatibility_task_success')}, 'fallback_summary': copy.deepcopy(terminal.get('fallback_summary', {})), 'artifacts': artifacts, 'bundle_content_sha256': sha256_json(artifacts), 'quality_summary': {'valid': bool(quality_report['quality_valid']), 'errors': copy.deepcopy(list(quality_report['errors'])), 'warnings': copy.deepcopy(list(quality_report['warnings']))}}
        if 'manifest.json' in artifact_paths or 'capture_state.json' in artifact_paths:
            raise BundleIntegrityError('artifact_table_invalid', 'journal/manifest must not self-list')
        validate_episode_manifest(manifest)
        return manifest

    def _validate_phase_b(self, manifest: Mapping[str, Any], manifest_bytes: bytes) -> None:
        validate_episode_manifest(manifest)
        if manifest_bytes != canonical_json_bytes(dict(manifest)) + b'\n':
            raise BundleIntegrityError('manifest_not_canonical', 'candidate manifest bytes differ')
        actual = _build_artifact_table(self.partial_root)
        if actual != manifest['artifacts']:
            raise BundleIntegrityError('artifact_table_mismatch', 'disk artifacts changed during finalization')
        if sha256_json(actual) != manifest['bundle_content_sha256']:
            raise BundleIntegrityError('bundle_content_hash_mismatch', 'artifact table hash mismatch')
        quality = load_json_strict(self.partial_root / 'quality_report.json')
        if not isinstance(quality, dict):
            raise BundleIntegrityError('quality_schema_invalid', 'quality report must be an object')
        validate_quality_report(quality)
        expected_train = not _derive_exclusion_reasons(episode_context=self.episode_context, run_context=self.run_context, quality_valid=quality['quality_valid'], terminal_error_class=manifest['terminal']['error_class'])
        if manifest['eligibility']['train_eligible'] is not expected_train:
            raise BundleIntegrityError('eligibility_mismatch', 'train eligibility was not derived')

    def _require_state(self, required: str) -> None:
        if self.state != required:
            raise BundleIntegrityError('writer_state_invalid', 'writer state is {}, expected {}'.format(self.state, required))

def _validate_contexts(run_context: RunContext, episode_context: EpisodeContext, capture_config: CaptureConfig) -> None:
    if not isinstance(run_context, RunContext):
        raise BundleIntegrityError('run_context_invalid', 'run_context must be RunContext')
    if not isinstance(episode_context, EpisodeContext):
        raise BundleIntegrityError('episode_context_invalid', 'episode_context must be EpisodeContext')
    if not isinstance(capture_config, CaptureConfig):
        raise BundleIntegrityError('capture_config_invalid', 'capture_config must be CaptureConfig')
    validate_run_manifest_minimal(run_context.manifest)
    assert_no_operational_leakage(run_context.manifest, root_pointer='/run_manifest', forbidden_absolute_prefixes=(Path(capture_config.root), Path.home()))
    if run_context.backend != BACKEND or run_context.manifest['backend'] != BACKEND:
        raise BundleIntegrityError('run_context_invalid', 'run backend must be eb_procthor')
    if run_context.run_id != run_context.manifest['run_id'] or episode_context.run_id != run_context.run_id:
        raise BundleIntegrityError('run_context_invalid', 'run ids disagree')
    for name, document in (('protocol', run_context.protocol), ('policy', run_context.policy), ('capture_config', run_context.capture_config_document)):
        validate_json_value(dict(document))
        assert_no_operational_leakage(document, root_pointer='/{}'.format(name), forbidden_absolute_prefixes=(Path(capture_config.root), Path.home()))
        if sha256_json(document) != run_context.manifest['{}_hash'.format(name)]:
            raise BundleIntegrityError('provenance_hash_mismatch', '{} hash differs from run manifest'.format(name))
        envelope = {'schema_version': SCHEMA_VERSIONS['provenance'], 'document_type': name, 'content_hash': run_context.manifest['{}_hash'.format(name)], 'content': dict(document)}
        if sha256_bytes(canonical_json_bytes(envelope) + b'\n') != run_context.manifest['provenance_documents'][name]['sha256']:
            raise BundleIntegrityError('provenance_hash_mismatch', '{} envelope hash differs'.format(name))
    payload = episode_context.episode_key_payload
    try:
        key_input = EpisodeKeyInput(**dict(payload))
    except TypeError as exc:
        raise BundleIntegrityError('episode_context_invalid', 'episode key payload field set is invalid') from exc
    expected_key, expected_id = build_episode_identity(key_input)
    if episode_context.episode_key != expected_key or episode_context.episode_id != expected_id:
        raise BundleIntegrityError('episode_context_invalid', 'episode key/id do not match payload')
    if payload['task_hash'] != episode_context.task.identity.task_hash:
        raise BundleIntegrityError('episode_context_invalid', 'task hash differs from EpisodeKey')
    identity_task = episode_context.task.identity.identity_task
    if payload['task_id'] != identity_task['id'] or payload['scene_id'] != identity_task['source']['house_id'] or payload['source_split'] != identity_task['source']['split'] or (payload['layout_id'] != ''):
        raise BundleIntegrityError('episode_context_invalid', 'backend identity fields disagree')
    if payload['execution_profile'] != episode_context.execution_profile:
        raise BundleIntegrityError('episode_context_invalid', 'execution profile differs')
    if episode_context.source_split != payload['source_split']:
        raise BundleIntegrityError('split_metadata_conflict', 'source split differs')
    if episode_context.task_split != run_context.manifest['task_split']:
        raise BundleIntegrityError('split_metadata_conflict', 'task split differs')
    if episode_context.rollout_purpose != run_context.manifest['rollout_purpose']:
        raise BundleIntegrityError('episode_context_invalid', 'rollout purpose differs')
    if payload['rollout_arm'] != run_context.manifest['rollout_arm']:
        raise BundleIntegrityError('episode_context_invalid', 'rollout arm differs')
    if set(episode_context.seeds) != set(SEED_NAMESPACES):
        raise BundleIntegrityError('episode_context_invalid', 'episode seeds must contain exactly five namespaces')
    for namespace in SEED_NAMESPACES:
        if episode_context.seeds[namespace] != derive_seed(expected_key, namespace):
            raise BundleIntegrityError('episode_context_invalid', '{} seed differs from EpisodeKey'.format(namespace))
    if capture_config.profile not in ('off', 'minimal', 'training', 'debug'):
        raise BundleIntegrityError('capture_config_invalid', 'capture profile is unsupported')
    if capture_config.profile == 'off':
        raise BundleIntegrityError('capture_config_invalid', 'bundle writer cannot use profile=off')
    if capture_config.keyframe_encoding != 'png' or capture_config.stream_encoding != 'jpeg':
        raise BundleIntegrityError('capture_config_invalid', 'v1 frame encodings are png/jpeg')
    if capture_config.stream_jpeg_quality != 90:
        raise BundleIntegrityError('capture_config_invalid', 'v1 JPEG quality must be 90')

def _validate_published_run(run_root: Path, run_context: RunContext) -> None:
    if not run_root.is_dir() or run_root.is_symlink():
        raise BundleConflictError('run_root_invalid', 'published run root does not exist')
    manifest_path = run_root / 'run_manifest.json'
    if manifest_path.is_symlink():
        raise BundleConflictError('run_manifest_conflict', 'run manifest symlink is forbidden')
    manifest_raw = manifest_path.read_bytes()
    manifest = strict_json_loads(manifest_raw.decode('utf-8'), source=str(manifest_path))
    if not isinstance(manifest, dict):
        raise BundleConflictError('run_manifest_conflict', 'run manifest must be an object')
    validate_run_manifest_minimal(manifest)
    if manifest_raw != canonical_json_bytes(manifest) + b'\n':
        raise BundleConflictError('run_manifest_conflict', 'run manifest is not canonical')
    assert_no_operational_leakage(manifest, root_pointer='/run_manifest', forbidden_absolute_prefixes=(run_root.parent, Path.home()))
    if canonical_json_bytes(manifest) != canonical_json_bytes(dict(run_context.manifest)):
        raise BundleConflictError('run_manifest_conflict', 'disk run manifest differs from RunContext')
    for name in ('protocol', 'policy', 'capture_config'):
        path = run_root / 'provenance' / (name + '.json')
        if path.is_symlink():
            raise BundleConflictError('run_manifest_conflict', 'run provenance symlink is forbidden')
        document_raw = path.read_bytes()
        document = strict_json_loads(document_raw.decode('utf-8'), source=str(path))
        if not isinstance(document, dict):
            raise BundleConflictError('run_manifest_conflict', 'run provenance must be object')
        validate_provenance_document(document, name)
        if document_raw != canonical_json_bytes(document) + b'\n':
            raise BundleConflictError('run_manifest_conflict', 'run provenance is not canonical')
        assert_no_operational_leakage(document, root_pointer='/provenance/{}'.format(name), forbidden_absolute_prefixes=(run_root.parent, Path.home()))
        if document['content_hash'] != manifest['{}_hash'.format(name)]:
            raise BundleConflictError('run_manifest_conflict', 'run provenance hash mismatch')
        if sha256_bytes(document_raw) != manifest['provenance_documents'][name]['sha256']:
            raise BundleConflictError('run_manifest_conflict', 'run provenance envelope bytes differ')
    if manifest['common_contract_sha256'] != COMMON_CONTRACT_SHA256:
        raise BundleConflictError('run_manifest_conflict', 'common contract hash mismatch')
    cache_manifest_path = run_root / manifest['asset_cache_manifest_path']
    if cache_manifest_path.is_symlink():
        raise BundleConflictError('run_manifest_conflict', 'asset cache manifest symlink is forbidden')
    cache_raw = cache_manifest_path.read_bytes()
    if sha256_bytes(cache_raw) != manifest['asset_cache_manifest_sha256']:
        raise BundleConflictError('run_manifest_conflict', 'asset cache manifest hash mismatch')
    cache_manifest = strict_json_loads(cache_raw.decode('utf-8'), source=str(cache_manifest_path))
    if not isinstance(cache_manifest, dict):
        raise BundleConflictError('run_manifest_conflict', 'asset cache manifest must be object')
    validate_asset_cache_manifest(cache_manifest)
    if cache_raw != canonical_json_bytes(cache_manifest) + b'\n':
        raise BundleConflictError('run_manifest_conflict', 'asset cache manifest is not canonical')
    if cache_manifest['entries'] != manifest['asset_cache']:
        raise BundleConflictError('run_manifest_conflict', 'asset cache entries differ')
    for entry in cache_manifest['entries']:
        asset_path = run_root / entry['relative_path']
        if asset_path.is_symlink():
            raise BundleConflictError('run_manifest_conflict', 'asset cache symlink is forbidden')
        raw = asset_path.read_bytes()
        if len(raw) != entry['bytes'] or sha256_bytes(raw) != entry['content_sha256']:
            raise BundleConflictError('run_manifest_conflict', 'asset cache bytes were tampered')
        parsed = strict_json_loads(raw.decode('utf-8'), source=str(asset_path))
        if raw != canonical_json_bytes(parsed):
            raise BundleConflictError('run_manifest_conflict', 'asset cache JSON is not canonical')

def _validate_complete_bundle(bundle_root: Path, *, expected_episode_id: str, expected_episode_key: str, expected_task_hash: str, expected_run_id: str) -> tuple[Mapping[str, Any], str]:
    if bundle_root.is_symlink() or not bundle_root.is_dir():
        raise BundleConflictError('episode_conflict', 'final bundle path is not a directory')
    manifest_path = bundle_root / 'manifest.json'
    raw = manifest_path.read_bytes()
    manifest = strict_json_loads(raw.decode('utf-8'), source=str(manifest_path))
    if not isinstance(manifest, dict):
        raise BundleConflictError('episode_conflict', 'manifest must be an object')
    validate_episode_manifest(manifest)
    if raw != canonical_json_bytes(manifest) + b'\n':
        raise BundleConflictError('episode_conflict', 'manifest is not canonical')
    if manifest['episode_id'] != expected_episode_id:
        raise BundleConflictError('episode_id_collision', 'episode id matches a different bundle')
    if manifest['episode_key'] != expected_episode_key:
        raise BundleConflictError('episode_id_collision', 'episode id maps to a different full key')
    if manifest['task']['hash'] != expected_task_hash or manifest['run_id'] != expected_run_id:
        raise BundleConflictError('episode_conflict', 'bundle identity differs from requested episode')
    try:
        rebuilt_key, rebuilt_id = build_episode_identity(EpisodeKeyInput(**dict(manifest['episode_key_payload'])))
    except (TypeError, TrajectoryExportError) as exc:
        raise BundleConflictError('episode_conflict', 'episode_key_payload is invalid') from exc
    if rebuilt_key != manifest['episode_key'] or rebuilt_id != manifest['episode_id']:
        raise BundleConflictError('episode_conflict', 'manifest identity does not match episode_key_payload')
    actual_artifacts = _build_artifact_table(bundle_root)
    if actual_artifacts != manifest['artifacts']:
        raise BundleConflictError('episode_conflict', 'artifact table or bytes were tampered')
    if sha256_json(actual_artifacts) != manifest['bundle_content_sha256']:
        raise BundleConflictError('episode_conflict', 'bundle content hash was tampered')
    quality = load_json_strict(bundle_root / 'quality_report.json')
    if not isinstance(quality, dict):
        raise BundleConflictError('episode_conflict', 'quality report must be an object')
    validate_quality_report(quality)
    if not quality['structural_valid']:
        raise BundleConflictError('episode_conflict', 'complete bundle has structurally invalid quality report')
    if manifest['quality_summary']['valid'] != quality['quality_valid'] or manifest['quality_summary']['errors'] != quality['errors'] or manifest['quality_summary']['warnings'] != quality['warnings']:
        raise BundleConflictError('episode_conflict', 'manifest quality summary differs from quality report')
    return (manifest, sha256_bytes(raw))

def _validate_complete_manifest_context(manifest: Mapping[str, Any], *, bundle_root: Path, run_context: RunContext, episode_context: EpisodeContext) -> None:
    payload = manifest['episode_key_payload']
    identity = episode_context.task.identity
    resolution = identity.path_resolutions[0]
    expected_task = {'id': identity.identity_task['id'], 'hash': identity.task_hash, 'family': identity.identity_task['category'], 'template_id': identity.identity_task['template']['template_id'], 'source_split': episode_context.source_split, 'task_split': episode_context.task_split, 'lineage_id': run_context.manifest['task_lineage_id'], 'task_bank_hash': run_context.manifest['task_bank_hash'], 'task_set_hash': run_context.manifest['task_set_hash']}
    if any((manifest['task'].get(key) != value for key, value in expected_task.items())):
        raise BundleConflictError('episode_conflict', 'manifest task differs from context')
    if manifest['scene']['scene_id'] != payload['scene_id'] or manifest['scene']['layout_id'] != payload['layout_id'] or manifest['scene']['house_content_sha256'] != resolution.content_sha256 or (manifest['scene']['house_path_hash'] != sha256_bytes(resolution.normalized_locator.encode('utf-8'))):
        raise BundleConflictError('episode_conflict', 'manifest scene differs from identity')
    if manifest['rollout']['purpose'] != episode_context.rollout_purpose or manifest['rollout']['arm'] != payload['rollout_arm'] or manifest['rollout']['execution_profile'] != episode_context.execution_profile or (manifest['rollout']['repeat_id'] != payload['repeat_id']):
        raise BundleConflictError('episode_conflict', 'manifest rollout differs from context')
    if manifest['seeds']['base'] != payload['base_seed']:
        raise BundleConflictError('episode_conflict', 'manifest base seed differs')
    for namespace in SEED_NAMESPACES:
        if manifest['seeds'][namespace]['value'] != episode_context.seeds[namespace]:
            raise BundleConflictError('episode_conflict', 'manifest {} seed differs'.format(namespace))
    if manifest['versions']['common_contract']['sha256'] != run_context.manifest['common_contract_sha256'] or manifest['versions']['protocol'] != run_context.manifest['protocol_hash'] or manifest['versions']['policy'] != run_context.manifest['policy_hash'] or (manifest['versions']['capture'] != run_context.manifest['capture_config_hash']):
        raise BundleConflictError('episode_conflict', 'manifest versions differ from run')
    quality = load_json_strict(bundle_root / 'quality_report.json')
    expected_exclusions = _derive_exclusion_reasons(episode_context=episode_context, run_context=run_context, quality_valid=quality['quality_valid'], terminal_error_class=manifest['terminal']['error_class'])
    if manifest['eligibility']['exclusion_reasons'] != expected_exclusions or manifest['eligibility']['train_eligible'] is not (not expected_exclusions) or manifest['eligibility']['admission_eligible'] is not _candidate_admission_eligible(episode_context=episode_context, terminal=manifest['terminal'], train_eligible=not expected_exclusions):
        raise BundleConflictError('episode_conflict', 'manifest eligibility differs from derived value')
    for name in ('protocol', 'policy', 'capture_config'):
        episode_document = (bundle_root / 'provenance' / (name + '.json')).read_bytes()
        run_document = run_context.manifest['provenance_documents'][name]['path']
        if episode_document != (bundle_root.parents[1] / run_document).read_bytes():
            raise BundleConflictError('episode_conflict', 'episode provenance is not byte-identical to run provenance')

def _build_artifact_table(bundle_root: Path) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for path in sorted(bundle_root.rglob('*'), key=lambda item: item.relative_to(bundle_root).as_posix()):
        relative = path.relative_to(bundle_root).as_posix()
        if path.is_symlink():
            raise BundleIntegrityError('artifact_symlink_forbidden', 'artifact symlinks are forbidden')
        if path.is_dir():
            continue
        if not path.is_file():
            raise BundleIntegrityError('artifact_type_invalid', 'non-regular artifact is forbidden')
        if relative in ('manifest.json', 'manifest.json.tmp', 'capture_state.json', 'capture_state.json.tmp'):
            continue
        raw = path.read_bytes()
        record_count = None
        if path.suffix == '.jsonl':
            records = load_jsonl_strict(path)
            record_count = len(records)
            expected = b''.join((canonical_json_bytes(record) + b'\n' for record in records))
            if raw != expected:
                raise BundleIntegrityError('jsonl_not_canonical', 'JSONL artifact is not canonical: {}'.format(relative))
        elif path.suffix == '.json':
            value = strict_json_loads(raw.decode('utf-8'), source=str(path))
            if raw != canonical_json_bytes(value) + b'\n':
                raise BundleIntegrityError('json_not_canonical', 'JSON artifact is not canonical: {}'.format(relative))
        artifacts.append({'path': relative, 'media_type': _media_type(path), 'privacy': _privacy_for_path(relative), 'bytes': len(raw), 'sha256': sha256_bytes(raw), 'record_count': record_count})
    return artifacts

def _validate_record_sequence(records: Sequence[Mapping[str, Any]]) -> None:
    if not records:
        raise BundleIntegrityError('record_cardinality_invalid', 'trajectory is empty')
    if records[0]['record_type'] != 'episode_start' or records[-1]['record_type'] != 'episode_end':
        raise BundleIntegrityError('record_order_invalid', 'trajectory boundaries are invalid')
    if sum((record['record_type'] == 'episode_start' for record in records)) != 1:
        raise BundleIntegrityError('record_cardinality_invalid', 'episode_start cardinality invalid')
    if sum((record['record_type'] == 'episode_end' for record in records)) != 1:
        raise BundleIntegrityError('record_cardinality_invalid', 'episode_end cardinality invalid')
    previous_logical = -1.0
    previous_monotonic = -1
    actions: dict[str, list[str]] = {}
    decisions: set[str] = set()
    for index, record in enumerate(records, start=1):
        if record['sequence_index'] != index or record['record_id'] != make_record_id(index):
            raise BundleIntegrityError('record_sequence_invalid', 'record sequence is not contiguous')
        if record['logical_time_s'] < previous_logical or record['monotonic_time_ns'] < previous_monotonic:
            raise BundleIntegrityError('record_time_regression', 'record time regressed')
        previous_logical = record['logical_time_s']
        previous_monotonic = record['monotonic_time_ns']
        if record['record_type'] == 'decision_input' and record['decision_id']:
            if record['decision_id'] in decisions:
                raise BundleIntegrityError('record_reference_invalid', 'duplicate decision id')
            decisions.add(record['decision_id'])
        action = record['action_id']
        if action:
            actions.setdefault(action, []).append(record['record_type'])
    for action, types in actions.items():
        if types.count('action_start') != 1 or types.count('action_end') != 1:
            raise BundleIntegrityError('record_cardinality_invalid', 'action is not closed')
        if types.index('action_start') > types.index('action_end'):
            raise BundleIntegrityError('record_order_invalid', 'action_end precedes action_start')

def _validate_references(index_records: Sequence[Mapping[str, Any]], frame_records: Sequence[Mapping[str, Any]], bundle_root: Path) -> None:
    frame_ids = {record.get('frame_id') for record in frame_records}
    if None in frame_ids or len(frame_ids) != len(frame_records):
        raise BundleIntegrityError('frame_record_invalid', 'frame ids are missing or duplicated')
    model_calls = load_jsonl_strict(bundle_root / 'public/model_io.jsonl')
    model_frame_ids = {record['frame_id'] for record in frame_records if record.get('frame_kind') == 'model_input'}
    for record in model_calls:
        if not isinstance(record, dict):
            raise BundleIntegrityError('model_io_invalid', 'model I/O must be an object')
        request = record.get('request')
        if not isinstance(request, dict):
            raise BundleIntegrityError('model_io_invalid', 'model request must be an object')
        for frame in request.get('image_frame_ids', []):
            if frame not in model_frame_ids:
                raise BundleIntegrityError('dangling_frame_reference', 'model frame is missing/private')
    state_records = load_jsonl_strict(bundle_root / 'privileged/state_snapshots.jsonl')
    state_ids = {record.get('snapshot_id') for record in state_records if isinstance(record, dict)}
    referenced_model_calls = {record['model_call_id'] for record in index_records if record.get('model_call_id') is not None}
    model_ids = {record.get('model_call_id') for record in model_calls if isinstance(record, dict)}
    if not referenced_model_calls.issubset(model_ids):
        raise BundleIntegrityError('dangling_model_call_reference', 'event references missing model call')
    public_records = load_jsonl_strict(bundle_root / 'public/trajectory.jsonl')
    private_records = load_jsonl_strict(bundle_root / 'privileged/trajectory.jsonl')
    _scan_reference_values(public_records + private_records, frame_ids, state_ids)

def _scan_reference_values(value: Any, frame_ids: set[Any], state_ids: set[Any]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key.endswith('frame_id') and item is not None and (item not in frame_ids):
                raise BundleIntegrityError('dangling_frame_reference', 'payload references missing frame')
            if key.endswith('frame_ids') and isinstance(item, list):
                if any((frame not in frame_ids for frame in item)):
                    raise BundleIntegrityError('dangling_frame_reference', 'payload references missing frame')
            if key.endswith('snapshot_id') and item is not None and (item not in state_ids):
                raise BundleIntegrityError('dangling_state_reference', 'payload references missing snapshot')
            _scan_reference_values(item, frame_ids, state_ids)
    elif isinstance(value, list):
        for item in value:
            _scan_reference_values(item, frame_ids, state_ids)

def _derive_exclusion_reasons(*, episode_context: EpisodeContext, run_context: RunContext, quality_valid: bool, terminal_error_class: str) -> list[str]:
    reasons: list[str] = []
    identity = episode_context.task.identity
    protocol_policy = run_context.protocol.get('data_policy', {})
    lineage_denylist = protocol_policy.get('nontrain_lineage_ids', [])
    task_denylist = protocol_policy.get('nontrain_task_hashes', [])
    if episode_context.source_split != 'train':
        reasons.append('source_split_not_train')
    if episode_context.task_split != 'train':
        reasons.append('task_split_not_train')
    if episode_context.rollout_purpose not in ('admission', 'training', 'dagger'):
        reasons.append('rollout_purpose_not_train')
    if run_context.manifest['task_lineage_id'] in lineage_denylist:
        reasons.append('nontrain_lineage')
    if identity.task_hash in task_denylist:
        reasons.append('nontrain_task_hash')
    if not quality_valid:
        reasons.append('quality_invalid')
    if terminal_error_class in ('infra_error', 'task_invalid', 'evaluator_invalid'):
        reasons.append(terminal_error_class)
    return sorted(set(reasons))

def _candidate_admission_eligible(*, episode_context: EpisodeContext, terminal: Mapping[str, Any], train_eligible: bool) -> bool:
    if episode_context.execution_profile == 'native_ai2thor':
        selected_success = terminal.get('native_task_success')
    else:
        selected_success = terminal.get('compatibility_task_success')
    return bool(train_eligible and episode_context.rollout_purpose == 'admission' and (terminal.get('episode_status') == 'passed') and (terminal.get('error_class') == 'none') and (selected_success is True))

def _validate_terminal(terminal: Mapping[str, Any]) -> None:
    required = {'episode_status', 'done_reason', 'error_class', 'error_code', 'native_task_success', 'compatibility_task_success', 'fallback_summary', 'seed_consumption', 'prompt_tokens', 'completion_tokens'}
    if set(terminal) != required:
        missing = required - set(terminal)
        extra = set(terminal) - required
        raise BundleIntegrityError('terminal_schema_invalid', 'terminal field set differs; missing={}, extra={}'.format(sorted(missing), sorted(extra)))
    if terminal['episode_status'] not in ('passed', 'failed', 'invalid'):
        raise BundleIntegrityError('terminal_schema_invalid', 'episode_status is invalid')
    if terminal['error_class'] not in ('infra_error', 'evaluator_invalid', 'task_invalid', 'budget_exceeded', 'invalid_action', 'agent_failure', 'none'):
        raise BundleIntegrityError('terminal_schema_invalid', 'error_class is invalid')
    invalid_terminal_field = _invalid_terminal_string_field(terminal)
    if invalid_terminal_field is not None:
        raise BundleIntegrityError('terminal_schema_invalid', '{} must be string or null'.format(invalid_terminal_field))
    for key in ('native_task_success', 'compatibility_task_success'):
        if terminal[key] is not None and (not isinstance(terminal[key], bool)):
            raise BundleIntegrityError('terminal_schema_invalid', '{} must be boolean or null'.format(key))
    for key in ('fallback_summary', 'seed_consumption'):
        if not isinstance(terminal[key], Mapping):
            raise BundleIntegrityError('terminal_schema_invalid', '{} must be an object'.format(key))
    seed_consumption = terminal['seed_consumption']
    unknown_namespaces = set(seed_consumption) - set(SEED_NAMESPACES)
    if unknown_namespaces:
        raise BundleIntegrityError('terminal_schema_invalid', 'seed_consumption has unknown namespaces: {}'.format(', '.join(sorted(unknown_namespaces))))
    for namespace, entry in seed_consumption.items():
        if not isinstance(entry, Mapping):
            raise BundleIntegrityError('terminal_schema_invalid', 'seed consumption entry must be an object')
        expected = {'consumed', 'consumer', 'reason'}
        if set(entry) != expected:
            raise BundleIntegrityError('terminal_schema_invalid', 'seed consumption entry field set differs for {}'.format(namespace))
        if type(entry['consumed']) is not bool:
            raise BundleIntegrityError('terminal_schema_invalid', 'seed consumed must be a boolean')
        for key in ('consumer', 'reason'):
            if entry[key] is not None and (not isinstance(entry[key], str)):
                raise BundleIntegrityError('terminal_schema_invalid', 'seed {} must be string or null'.format(key))
    for key in ('prompt_tokens', 'completion_tokens'):
        if terminal[key] is not None and (isinstance(terminal[key], bool) or not isinstance(terminal[key], int) or terminal[key] < 0):
            raise BundleIntegrityError('terminal_schema_invalid', '{} must be non-negative integer or null'.format(key))

def _add_check_error(checks: Mapping[str, Any], all_errors: list[dict[str, Any]], check_name: str, error: Exception) -> None:
    code = getattr(error, 'code', type(error).__name__)
    pointer = getattr(error, 'json_pointer', None)
    row = {'error_code': code, 'json_pointer': pointer, 'message': str(error)}
    checks[check_name]['valid'] = False
    checks[check_name]['errors'].append(row)
    all_errors.append(row)

def _archive_partial(run_root: Path, partial_root: Path, episode_id: str) -> None:
    destination_root = run_root / 'incomplete' / episode_id
    destination_root.mkdir(parents=True, exist_ok=True)
    stem = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    destination = destination_root / stem
    ordinal = 1
    while destination.exists():
        destination = destination_root / '{}-{:03d}'.format(stem, ordinal)
        ordinal += 1
    os.replace(partial_root, destination)
    row = {'original_path': 'episodes/{}.partial'.format(episode_id), 'archived_path': destination.relative_to(run_root).as_posix(), 'reason': 'resume_incomplete_episode', 'archived_at_utc': _utc_now()}
    _append_locked_jsonl(run_root / 'incomplete_index.jsonl', row, fsync=True)
    _fsync_directory(partial_root.parent, enabled=True)

def _ensure_run_index_row(run_root: Path, *, manifest: Mapping[str, Any], manifest_hash: str, status: str, fsync: bool) -> None:
    if status not in ('finalized', 'repaired_index'):
        raise BundleConflictError('episode_index_conflict', 'episode index status is invalid')
    index_path = run_root / 'episodes.jsonl'
    row = {'episode_id': manifest['episode_id'], 'episode_key': manifest['episode_key'], 'task_id': manifest['task']['id'], 'repeat_id': manifest['rollout']['repeat_id'], 'status': status, 'manifest_path': 'episodes/{}/manifest.json'.format(manifest['episode_id']), 'manifest_sha256': manifest_hash, 'train_eligible': manifest['eligibility']['train_eligible'], 'error_class': manifest['terminal']['error_class']}
    lock_path = run_root / '.episodes.lock'
    lock_path.touch(exist_ok=True)
    with lock_path.open('r+b') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            existing = _load_canonical_jsonl(index_path) if index_path.exists() else []
            seen_episode_ids: set[str] = set()
            for item in existing:
                if not isinstance(item, Mapping) or set(item) != set(row):
                    raise BundleConflictError('episode_index_conflict', 'episode index row shape is invalid')
                if item['episode_id'] in seen_episode_ids:
                    raise BundleConflictError('episode_index_conflict', 'episode index contains duplicate episode ids')
                seen_episode_ids.add(item['episode_id'])
            matching = [item for item in existing if item.get('episode_id') == row['episode_id']]
            if matching:
                current = matching[0]
                if current['status'] not in ('finalized', 'repaired_index') or any((current[key] != row[key] for key in row if key != 'status')):
                    raise BundleConflictError('episode_index_conflict', 'episode index differs from complete manifest')
                return
            with index_path.open('ab') as handle:
                handle.write(canonical_json_bytes(row) + b'\n')
                handle.flush()
                if fsync:
                    os.fsync(handle.fileno())
            _fsync_directory(run_root, enabled=fsync)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

def _append_locked_jsonl(path: Path, row: Mapping[str, Any], *, fsync: bool) -> None:
    lock_path = path.with_name('.' + path.name + '.lock')
    lock_path.touch(exist_ok=True)
    with lock_path.open('r+b') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            with path.open('ab') as handle:
                handle.write(canonical_json_bytes(dict(row)) + b'\n')
                handle.flush()
                if fsync:
                    os.fsync(handle.fileno())
            _fsync_directory(path.parent, enabled=fsync)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

def _load_canonical_jsonl(path: Path) -> list[Any]:
    raw = path.read_bytes()
    records = load_jsonl_strict(path)
    expected = b''.join((canonical_json_bytes(record) + b'\n' for record in records))
    if raw != expected:
        raise BundleConflictError('episode_index_conflict', 'run JSONL index is not canonical')
    return records

def _write_atomic_json(path: Path, value: Mapping[str, Any], *, fsync: bool) -> None:
    _write_atomic_bytes(path, canonical_json_bytes(dict(value)) + b'\n', fsync=fsync)

def _write_atomic_bytes(path: Path, value: bytes, *, fsync: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 420)
    except FileExistsError as exc:
        raise BundleConflictError('writer_temporary_conflict', 'temporary file already exists: {}'.format(temporary)) from exc
    try:
        with os.fdopen(descriptor, 'wb') as handle:
            handle.write(value)
            handle.flush()
            if fsync:
                os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent, enabled=fsync)
    except Exception:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise

def _fsync_directory(path: Path, *, enabled: bool) -> None:
    if not enabled:
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

def _make_episode_read_only(root: Path) -> None:
    paths = sorted(root.rglob('*'), key=lambda path: len(path.parts), reverse=True)
    for path in paths:
        if path.is_symlink():
            raise BundleIntegrityError('artifact_symlink_forbidden', 'episode symlinks are forbidden')
        if path.is_file():
            path.chmod(292)
        elif path.is_dir():
            path.chmod(365)
    root.chmod(365)

def _media_type(path: Path) -> str:
    if path.suffix == '.json':
        return 'application/json'
    if path.suffix == '.jsonl':
        return 'application/x-ndjson'
    if path.suffix == '.png':
        return 'image/png'
    if path.suffix in ('.jpg', '.jpeg'):
        return 'image/jpeg'
    if path.suffix == '.gz':
        return 'application/gzip'
    return 'application/octet-stream'

def _privacy_for_path(relative: str) -> str:
    if relative == 'trajectory.jsonl':
        return 'index'
    if relative == 'quality_report.json':
        return 'quality'
    if relative.startswith('public/'):
        return 'public'
    if relative.startswith('privileged/'):
        return 'privileged'
    if relative.startswith('provenance/'):
        return 'provenance'
    raise BundleIntegrityError('artifact_privacy_unknown', 'cannot classify artifact privacy: {}'.format(relative))

def _artifact_record_count(artifacts: Sequence[Mapping[str, Any]], path: str) -> int:
    for artifact in artifacts:
        if artifact['path'] == path:
            return int(artifact['record_count'] or 0)
    return 0

def _sum_model_tokens(records: Sequence[Any], key: str) -> int | None:
    if not records:
        return None
    values = [record.get('usage', {}).get(key) if isinstance(record, Mapping) else None for record in records]
    if any((type(value) is not int or value < 0 for value in values)):
        return None
    return sum(values)

def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
__all__ = ['BundleConflictError', 'BundleIntegrityError', 'EpisodeBundleWriter']
