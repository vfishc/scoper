from __future__ import annotations
import fcntl
import io
import os
import shutil
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Mapping
import numpy as np
from PIL import Image
from scoper.benchmark.behavior.export.ids import SEED_NAMESPACES, canonical_json_bytes, canonical_json_line, frame_id as make_frame_id, json_deep_copy, model_call_id as make_model_call_id, record_id as make_record_id, sha256_bytes, sha256_file, sha256_json, state_snapshot_id as make_state_snapshot_id, strict_json_load, strict_json_loads, validate_json_value, validate_run_id
from scoper.benchmark.behavior.export.leakage import assert_no_public_leakage
from scoper.benchmark.behavior.export.quality import compute_rgb_quality, decode_image_bytes, summarize_frame_quality
from scoper.benchmark.behavior.export.schema import COMMON_CONTRACT_SHA256, CONTRACT_VERSION, build_artifact_table, validate_common_contract_fixture, validate_episode_bundle, validate_episode_bundle_candidate, validate_event_envelope, validate_payload_envelope, validate_provenance_document, validate_raw_episode_files, validate_run_manifest, validate_terminal_input
from scoper.benchmark.behavior.export.types import BundleConflictError, BundleStructuralError, CaptureConfig, EpisodeContext, FrameMetadata, RunContext, SkippedEpisode, TrajectoryConfigError, TrajectoryContractError

class EpisodeBundleWriter:

    def __init__(self, *, run_context: RunContext, episode_context: EpisodeContext, capture_config: CaptureConfig, run_root: Path, partial_root: Path) -> None:
        self.run_context = run_context
        self.episode_context = episode_context
        self.capture_config = capture_config
        self.run_root = run_root
        self.partial_root = partial_root
        self.final_root = run_root / 'episodes' / episode_context.episode_id
        self.state = 'NEW'
        self._handles: dict[str, BinaryIO] = {}
        self._record_count = 0
        self._frame_count = 0
        self._state_count = 0
        self._model_call_count = 0
        self._frame_records: list[Mapping[str, Any]] = []
        self._started_at_utc = _utc_now()
        self._start_monotonic_ns = time.monotonic_ns()
        self._last_logical_time = 0.0
        self._last_monotonic_ns = 0
        self._exposure_profile = 'visual_minimal'
        self._initialize_partial()

    @classmethod
    def open(cls, *, run_context: RunContext, episode_context: EpisodeContext, capture_config: CaptureConfig, resume: bool) -> 'EpisodeBundleWriter | SkippedEpisode':
        _validate_contexts(run_context, episode_context, capture_config)
        validate_common_contract_fixture()
        run_root = _resolve_run_root(capture_config.root, run_context.run_id)
        disk_manifest = strict_json_load(run_root / 'run_manifest.json')
        if disk_manifest != run_context.manifest:
            raise BundleConflictError('run_manifest_conflict', 'RunContext manifest differs from run root')
        try:
            validate_run_manifest(disk_manifest, run_root=run_root)
        except TrajectoryContractError as exc:
            raise BundleConflictError('run_manifest_conflict', str(exc)) from exc
        context_documents = {'protocol': run_context.protocol, 'policy': run_context.policy, 'capture_config': run_context.capture_config_document}
        for name, context_document in context_documents.items():
            try:
                normalized = _normalized_context_document(context_document, name)
            except TrajectoryContractError as exc:
                raise BundleConflictError('provenance_conflict', str(exc)) from exc
            disk_document = strict_json_load(run_root / 'provenance' / f'{name}.json')
            if normalized != disk_document:
                raise BundleConflictError('provenance_conflict', f'RunContext {name} differs from run provenance')
        episodes_root = run_root / 'episodes'
        episodes_root.mkdir(exist_ok=True)
        final_root = episodes_root / episode_context.episode_id
        partial_root = episodes_root / f'{episode_context.episode_id}.partial'
        if final_root.exists():
            if not final_root.is_dir() or final_root.is_symlink():
                raise BundleConflictError('episode_conflict', 'final episode path is not a regular directory')
            try:
                manifest = strict_json_load(final_root / 'manifest.json')
                existing_key = manifest.get('episode_key') if isinstance(manifest, Mapping) else None
                if existing_key is not None and existing_key != episode_context.episode_key:
                    raise BundleConflictError('episode_id_collision', 'episode id exists with a different full EpisodeKey')
                result = validate_episode_bundle(final_root, expected_episode_key=episode_context.episode_key, expected_episode_id=episode_context.episode_id, expected_run_id=run_context.run_id)
            except BundleConflictError:
                raise
            except Exception as exc:
                raise BundleConflictError('episode_conflict', f'existing final episode is invalid: {exc}') from exc
            if not resume:
                raise BundleConflictError('episode_exists', 'complete episode already exists')
            _ensure_episode_index_row(run_root, _episode_index_row(manifest, result.manifest_sha256, status='repaired_index'), repair_only=True)
            return SkippedEpisode(episode_id=episode_context.episode_id, episode_key=episode_context.episode_key, manifest_path=final_root / 'manifest.json', manifest_sha256=result.manifest_sha256, reason='skipped_complete')
        if partial_root.exists():
            if not resume:
                raise BundleConflictError('incomplete_episode_exists', 'partial episode already exists')
            _archive_incomplete_episode(run_root, partial_root, episode_context.episode_id)
        try:
            partial_root.mkdir(mode=493)
        except FileExistsError as exc:
            raise BundleConflictError('incomplete_episode_exists', 'partial episode was created concurrently') from exc
        return cls(run_context=run_context, episode_context=episode_context, capture_config=capture_config, run_root=run_root, partial_root=partial_root)

    def __enter__(self) -> 'EpisodeBundleWriter':
        return self

    def next_record_id(self) -> str:
        self._require_state('OPEN')
        return make_record_id(self._record_count + 1)

    def next_model_call_id(self) -> str:
        self._require_state('OPEN')
        return make_model_call_id(self._model_call_count + 1)

    def __exit__(self, exc_type: Any, exc: BaseException | None, traceback: Any) -> bool:
        if self.state != 'COMPLETE':
            error = {'error_class': 'infra_error', 'error_code': 'context_exit_without_finalize' if exc is None else 'writer_context_exception', 'message': None if exc is None else str(exc)}
            self.abort(error)
        return False

    def append_event(self, envelope: Mapping[str, Any], *, public_payload: Mapping[str, Any] | None, privileged_payload: Mapping[str, Any] | None) -> str:
        self._require_state('OPEN')
        validate_json_value(envelope)
        sequence = self._record_count + 1
        record = _build_event_record(envelope, sequence=sequence, episode_id=self.episode_context.episode_id, start_monotonic_ns=self._start_monotonic_ns, last_logical_time=self._last_logical_time, last_monotonic_ns=self._last_monotonic_ns, has_public=public_payload is not None, has_privileged=privileged_payload is not None)
        validate_event_envelope(record, expected_sequence=sequence)
        public_row = None
        if public_payload is not None:
            if record['record_type'] == 'episode_start':
                self._exposure_profile = public_payload.get('exposure_profile')
                if self._exposure_profile not in {'visual_minimal', 'structured_agent', 'privileged_oracle'}:
                    raise TrajectoryConfigError('exposure_profile_invalid', 'episode exposure profile is invalid')
            assert_no_public_leakage(public_payload, exposure_profile=self._exposure_profile, allow_privileged_oracle=self._exposure_profile == 'privileged_oracle')
            public_row = {'schema_version': 'trajectory-payload.v1', 'record_id': record['record_id'], 'record_type': record['record_type'], 'payload': json_deep_copy(public_payload)}
            validate_payload_envelope(public_row, privacy='public')
        privileged_row = None
        if privileged_payload is not None:
            privileged_row = {'schema_version': 'trajectory-payload.v1', 'record_id': record['record_id'], 'record_type': record['record_type'], 'payload': json_deep_copy(privileged_payload)}
            validate_payload_envelope(privileged_row, privacy='privileged')
        self._append_jsonl('trajectory.jsonl', record)
        if public_row is not None:
            self._append_jsonl('public/trajectory.jsonl', public_row)
        if privileged_row is not None:
            self._append_jsonl('privileged/trajectory.jsonl', privileged_row)
        self._record_count = sequence
        self._last_logical_time = float(record['logical_time_s'])
        self._last_monotonic_ns = int(record['monotonic_time_ns'])
        if record['record_type'] in {'action_end', 'intervention', 'episode_end'}:
            self._flush_all(sync=self.capture_config.fsync)
        self._write_capture_state('open')
        return record['record_id']

    def write_frame(self, rgb: Any, metadata: FrameMetadata, privileged_annotation: Mapping[str, Any] | None=None) -> str:
        self._require_state('OPEN')
        if not isinstance(metadata, FrameMetadata):
            raise TrajectoryConfigError('frame_metadata_invalid', 'metadata must be FrameMetadata')
        array = np.asarray(rgb)
        if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
            raise TrajectoryConfigError('frame_serialization_error', 'writer requires HWC RGB uint8 input')
        encoding = self.capture_config.stream_encoding if metadata.frame_kind == 'stream_sample' else self.capture_config.keyframe_encoding
        encoded, mime_type, parameters = _encode_rgb(array, encoding=encoding, jpeg_quality=self.capture_config.stream_jpeg_quality)
        return self._write_encoded_frame(encoded, rgb=array, mime_type=mime_type, encoding_parameters=parameters, metadata=metadata, privileged_annotation=privileged_annotation)

    def write_model_image(self, encoded: bytes, *, mime_type: str, source_frame_id: str, metadata: FrameMetadata) -> str:
        self._require_state('OPEN')
        if mime_type not in {'image/png', 'image/jpeg'}:
            raise TrajectoryConfigError('encoded_image_invalid', 'model image MIME type must be PNG or JPEG')
        rgb, decoded = decode_image_bytes(encoded)
        expected_format = 'PNG' if mime_type == 'image/png' else 'JPEG'
        if decoded['format'] != expected_format:
            raise TrajectoryConfigError('encoded_image_invalid', 'MIME type does not match encoded image')
        capture = dict(metadata.capture)
        capture['source_frame_id'] = source_frame_id
        model_metadata = replace(metadata, frame_kind='model_input', source='model_preprocessor', capture=capture)
        parameters: Mapping[str, Any] = {'preserved_exact_model_bytes': True}
        return self._write_encoded_frame(encoded, rgb=rgb, mime_type=mime_type, encoding_parameters=parameters, metadata=model_metadata, privileged_annotation=None)

    def write_state_snapshot(self, snapshot: Mapping[str, Any]) -> str:
        self._require_state('OPEN')
        value = json_deep_copy(snapshot)
        self._state_count += 1
        expected = make_state_snapshot_id(self._state_count)
        supplied = value.get('snapshot_id')
        if supplied is not None and supplied != expected:
            self._state_count -= 1
            raise TrajectoryConfigError('state_id_conflict', 'snapshot_id does not match writer ordinal')
        value['snapshot_id'] = expected
        value.setdefault('schema_version', 'eb-behavior-state.v1')
        value.setdefault('episode_id', self.episode_context.episode_id)
        self._append_jsonl('privileged/state_snapshots.jsonl', value)
        return expected

    def write_model_call(self, record: Mapping[str, Any]) -> str:
        self._require_state('OPEN')
        value = json_deep_copy(record)
        self._model_call_count += 1
        expected = make_model_call_id(self._model_call_count)
        supplied = value.get('model_call_id')
        if supplied is not None and supplied != expected:
            self._model_call_count -= 1
            raise TrajectoryConfigError('model_call_id_conflict', 'model_call_id does not match writer ordinal')
        value['model_call_id'] = expected
        value.setdefault('schema_version', 'model-io.v1')
        value.setdefault('episode_id', self.episode_context.episode_id)
        assert_no_public_leakage(value, exposure_profile=self._exposure_profile, allow_privileged_oracle=self._exposure_profile == 'privileged_oracle')
        self._append_jsonl('public/model_io.jsonl', value)
        return expected

    def write_admission_preflight(self, record: Mapping[str, Any]) -> str:
        self._require_state('OPEN')
        value = json_deep_copy(record)
        from scoper.benchmark.behavior.benchmark.admission_schema import validate_reachability_preflight
        validate_reachability_preflight(value)
        destination = self.partial_root / 'privileged' / 'admission_preflight.json'
        _write_json_file(destination, value, fsync=self.capture_config.fsync, exclusive=True)
        return 'privileged/admission_preflight.json'

    def write_render_replay_state(self, record: Mapping[str, Any]) -> None:
        self._require_state('OPEN')
        from scoper.benchmark.behavior.render.particle_segmentation_replay import validate_render_replay_state
        value = json_deep_copy(record)
        validate_render_replay_state(value, episode_id=self.episode_context.episode_id, frame_id=value.get('frame_id'))
        self._append_jsonl('privileged/render_replay_states.jsonl', value)

    def replace_privileged_jsonl(self, filename: str, rows: list[Mapping[str, Any]]) -> None:
        self._require_state('OPEN')
        if filename not in {'frame_annotations.jsonl', 'trajectory.jsonl'}:
            raise TrajectoryConfigError('jsonl_replace_invalid', 'privileged JSONL replacement is not allowed')
        values = [json_deep_copy(row) for row in rows]
        relative = f'privileged/{filename}'
        handle = self._handles.pop(relative, None)
        if handle is not None:
            handle.flush()
            if self.capture_config.fsync:
                os.fsync(handle.fileno())
            handle.close()
        destination = self.partial_root / relative
        temporary = destination.with_name(destination.name + '.tmp')
        payload = b''.join((canonical_json_line(value) for value in values))
        try:
            _write_bytes(temporary, payload, fsync=self.capture_config.fsync, exclusive=True)
            os.replace(temporary, destination)
            if self.capture_config.fsync:
                _fsync_directory(destination.parent)
        finally:
            if temporary.exists():
                temporary.unlink()
        self._write_capture_state('open')

    def finalize(self, terminal: Mapping[str, Any]) -> Path:
        self._require_state('OPEN')
        self.state = 'FINALIZING'
        try:
            validate_json_value(terminal)
            validate_terminal_input(terminal)
            self._close_handles()
            validate_raw_episode_files(self.partial_root, episode_id=self.episode_context.episode_id, task_hash=self.episode_context.task.identity.task_hash)
            frame_summary = summarize_frame_quality(self._frame_records)
            eligibility, exclusion_reasons = self._derive_eligibility(terminal, frame_summary)
            quality_report = self._build_quality_report(frame_summary, exclusion_reasons)
            _write_json_file(self.partial_root / 'quality_report.json', quality_report, fsync=self.capture_config.fsync, exclusive=True)
            capture_state = self.partial_root / 'capture_state.json'
            capture_state.unlink()
            if self.capture_config.fsync:
                _fsync_directory(self.partial_root)
            artifacts = build_artifact_table(self.partial_root)
            terminal_elapsed_seconds = max(0.0, (time.monotonic_ns() - self._start_monotonic_ns) / 1000000000.0)
            manifest = self._build_manifest(terminal=terminal, frame_summary=frame_summary, eligibility=eligibility, exclusion_reasons=exclusion_reasons, artifacts=artifacts, elapsed_seconds=terminal_elapsed_seconds)
            validate_episode_bundle_candidate(self.partial_root, manifest)
            manifest_bytes = canonical_json_line(manifest)
            manifest_temp = self.partial_root / 'manifest.json.tmp'
            _write_bytes(manifest_temp, manifest_bytes, fsync=self.capture_config.fsync, exclusive=True)
            os.replace(manifest_temp, self.partial_root / 'manifest.json')
            if self.capture_config.fsync:
                _fsync_directory(self.partial_root)
            candidate_result = validate_episode_bundle(self.partial_root, expected_episode_key=self.episode_context.episode_key, expected_episode_id=self.episode_context.episode_id, expected_run_id=self.run_context.run_id)
            os.replace(self.partial_root, self.final_root)
            if self.capture_config.fsync:
                _fsync_directory(self.final_root.parent)
            self.state = 'COMPLETE'
            try:
                _ensure_episode_index_row(self.run_root, _episode_index_row(manifest, candidate_result.manifest_sha256, status=terminal['episode_status']), repair_only=False)
            finally:
                _make_episode_read_only(self.final_root)
            return self.final_root / 'manifest.json'
        except Exception:
            if self.state != 'COMPLETE':
                self.abort({'error_class': 'infra_error', 'error_code': 'bundle_finalization_error', 'message': 'episode bundle finalization failed'})
            raise

    def abort(self, error: Mapping[str, Any]) -> None:
        if self.state in {'COMPLETE', 'ABORTED'}:
            return
        self._close_handles()
        self.state = 'ABORTED'
        if self.partial_root.exists():
            payload = {'schema_version': 'capture-state.v1', 'status': 'aborted', 'episode_id': self.episode_context.episode_id, 'episode_key': self.episode_context.episode_key, 'records_written': self._record_count, 'frames_written': self._frame_count, 'error': json_deep_copy(error), 'updated_at_utc': _utc_now()}
            _atomic_json_replace(self.partial_root / 'capture_state.json', payload, fsync=self.capture_config.fsync)

    def _initialize_partial(self) -> None:
        for directory in ('public', 'public/frames', 'privileged', 'privileged/frames', 'privileged/snapshots', 'provenance'):
            (self.partial_root / directory).mkdir(parents=True, exist_ok=True)
        self._write_capture_state('open', exclusive=True)
        for relative in ('trajectory.jsonl', 'public/trajectory.jsonl', 'privileged/trajectory.jsonl'):
            self._open_handle(relative)
        self._write_static_documents()
        self.state = 'OPEN'

    def _write_static_documents(self) -> None:
        identity = self.episode_context.task.identity
        task = identity.identity_task
        protocol_content = _document_content(self.run_context.protocol, 'protocol')
        tool_schema_version = (protocol_content.get('tool_schema') or {}).get('version')
        public_metadata = {}
        for key in ('family', 'category', 'difficulty', 'public_handles'):
            if key in task:
                public_metadata[key] = json_deep_copy(task[key])
        template = task.get('template')
        template_value = task.get('template_id')
        if template_value is None and isinstance(template, Mapping):
            template_value = template.get('template_id') or template.get('id')
        if isinstance(template_value, str) and template_value:
            public_metadata['template'] = template_value
        public_task = {'schema_version': 'task-snapshot.v1', 'privacy': 'public', 'task_id': task['id'], 'instruction': task['instruction'], 'public_task_metadata': public_metadata, 'tool_schema_version': tool_schema_version}
        assert_no_public_leakage(public_task)
        private_task = {'schema_version': 'task-snapshot.v1', 'privacy': 'privileged', 'task_id': task['id'], 'task_hash': identity.task_hash, 'full_task_spec': json_deep_copy(task), 'task_path_resolutions': [asdict(item) for item in identity.path_resolutions]}
        _write_json_file(self.partial_root / 'public/task.json', public_task, fsync=self.capture_config.fsync, exclusive=True)
        _write_json_file(self.partial_root / 'privileged/task.json', private_task, fsync=self.capture_config.fsync, exclusive=True)
        contexts = {'protocol': self.run_context.protocol, 'policy': self.run_context.policy, 'capture_config': self.run_context.capture_config_document}
        for name, context_document in contexts.items():
            run_path = self.run_root / 'provenance' / f'{name}.json'
            run_bytes = run_path.read_bytes()
            run_document = strict_json_loads(run_bytes, source=str(run_path))
            validate_provenance_document(run_document, document_type=name)
            if _normalized_context_document(context_document, name) != run_document:
                raise BundleConflictError('provenance_conflict', f'RunContext {name} differs from run provenance')
            _write_bytes(self.partial_root / 'provenance' / f'{name}.json', run_bytes, fsync=self.capture_config.fsync, exclusive=True)

    def _write_encoded_frame(self, encoded: bytes, *, rgb: np.ndarray, mime_type: str, encoding_parameters: Mapping[str, Any], metadata: FrameMetadata, privileged_annotation: Mapping[str, Any] | None) -> str:
        decoded_rgb, decoded_metadata = decode_image_bytes(encoded)
        expected_format = 'PNG' if mime_type == 'image/png' else 'JPEG'
        if decoded_metadata['format'] != expected_format or decoded_rgb.shape[:2] != rgb.shape[:2]:
            raise TrajectoryConfigError('frame_serialization_error', 'encoded frame does not decode to the source shape')
        self._frame_count += 1
        identifier = make_frame_id(self._frame_count)
        extension = '.png' if mime_type == 'image/png' else '.jpg'
        relative_path = f'public/frames/{identifier}{extension}'
        destination = self.partial_root / relative_path
        temporary = destination.with_name(destination.name + '.tmp')
        try:
            _write_bytes(temporary, encoded, fsync=self.capture_config.fsync, exclusive=True)
            os.replace(temporary, destination)
            if self.capture_config.fsync:
                _fsync_directory(destination.parent)
        except Exception:
            self._frame_count -= 1
            try:
                temporary.unlink()
            except OSError:
                pass
            raise
        height, width = decoded_rgb.shape[:2]
        record = {'schema_version': 'frame-record.v1', 'frame_id': identifier, 'episode_id': self.episode_context.episode_id, 'sequence_index': self._frame_count, 'step_id': metadata.step_id, 'action_id': metadata.action_id, 'stream_id': metadata.stream_id, 'stream_obs_id': metadata.stream_obs_id, 'frame_kind': metadata.frame_kind, 'source': metadata.source, 'logical_time_s': metadata.logical_time_s, 'monotonic_time_ns': metadata.monotonic_time_ns, 'wall_time_utc': metadata.wall_time_utc, 'path': relative_path, 'sha256': sha256_bytes(encoded), 'bytes': len(encoded), 'width': int(width), 'height': int(height), 'channels': 3, 'dtype': 'uint8', 'color_space': 'sRGB', 'encoding': 'png' if mime_type == 'image/png' else 'jpeg', 'encoding_parameters': json_deep_copy(encoding_parameters), 'camera': json_deep_copy(metadata.camera), 'capture': json_deep_copy(metadata.capture), 'public_visibility_summary': {}, 'quality': compute_rgb_quality(decoded_rgb)}
        assert_no_public_leakage(record, exposure_profile=self._exposure_profile, allow_privileged_oracle=self._exposure_profile == 'privileged_oracle')
        self._append_jsonl('public/frames.jsonl', record)
        self._frame_records.append(record)
        if privileged_annotation is not None:
            annotation = json_deep_copy(privileged_annotation)
            annotation.setdefault('schema_version', 'frame-annotation.v1')
            annotation.setdefault('frame_id', identifier)
            annotation.setdefault('episode_id', self.episode_context.episode_id)
            self._append_jsonl('privileged/frame_annotations.jsonl', annotation)
        return identifier

    def _derive_eligibility(self, terminal: Mapping[str, Any], frame_summary: Mapping[str, Any]) -> tuple[Mapping[str, Any], list[str]]:
        reasons: list[str] = []
        context = self.episode_context
        manifest = self.run_context.manifest
        protocol = _document_content(self.run_context.protocol, 'protocol')
        data_policy = protocol.get('data_policy') if isinstance(protocol, Mapping) else {}
        if context.source_split != 'train':
            reasons.append('source_split_nontrain')
        if context.task_split != 'train':
            reasons.append('task_split_nontrain')
        if context.rollout_purpose not in {'admission', 'training', 'dagger'}:
            reasons.append('rollout_purpose_nontrain')
        if manifest['task_lineage_id'] in (data_policy or {}).get('nontrain_lineage_ids', []):
            reasons.append('nontrain_lineage')
        if context.task.identity.task_hash in (data_policy or {}).get('nontrain_task_hashes', []):
            reasons.append('nontrain_task_hash')
        if not frame_summary['quality_valid']:
            reasons.append('quality_invalid')
        error_class = terminal['error_class']
        if error_class in {'infra_error', 'evaluator_invalid', 'task_invalid'}:
            reasons.append(f'terminal_{error_class}')
        episode_start = next((row for row in _read_optional_jsonl_rows(self.partial_root / 'public' / 'trajectory.jsonl') if row.get('record_type') == 'episode_start'), None)
        if episode_start is not None and episode_start['payload'].get('exposure_profile') == 'privileged_oracle':
            reasons.append('privileged_oracle_exposure')
        reasons = sorted(set(reasons))
        train_eligible = not reasons
        return ({'train_eligible': train_eligible, 'admission_eligible': self._derive_admission_eligible(terminal=terminal, train_eligible=train_eligible), 'exclusion_reasons': reasons}, reasons)

    def _derive_admission_eligible(self, *, terminal: Mapping[str, Any], train_eligible: bool) -> bool:
        if self.episode_context.rollout_purpose != 'admission' or not train_eligible:
            return False
        if terminal['episode_status'] != 'passed' or terminal['error_class'] != 'none':
            return False
        preflight_path = self.partial_root / 'privileged' / 'admission_preflight.json'
        if not preflight_path.is_file():
            return False
        preflight = strict_json_load(preflight_path)
        if preflight.get('status') != 'pass' or preflight.get('disposition') != 'admitted':
            return False
        policy = _document_content(self.run_context.policy, 'policy')
        if policy.get('mode') not in {'teacher', 'oracle'}:
            return False
        private_rows = _read_optional_jsonl_rows(self.partial_root / 'privileged' / 'trajectory.jsonl')
        episode_end = next((row for row in private_rows if row.get('record_type') == 'episode_end'), None)
        final_goal = episode_end.get('payload', {}).get('final_goal') if episode_end is not None else None
        predicate_success = final_goal.get('predicate_success') if isinstance(final_goal, Mapping) else None
        return not isinstance(predicate_success, bool) and isinstance(predicate_success, (int, float)) and (float(predicate_success) == 1.0)

    def _build_quality_report(self, frame_summary: Mapping[str, Any], exclusion_reasons: list[str]) -> Mapping[str, Any]:
        profile = self.capture_config.profile
        capabilities = {'profile': profile, 'reset_decision_pre_post_final_rgb': True, 'task_relevant_boundary_state': True, 'actual_model_inputs': True, 'all_stream_rgb': profile in {'training', 'debug'}, 'intervention_evidence': profile in {'training', 'debug'}, 'full_world_reset_final': profile in {'training', 'debug'}, 'debug_full_world_boundaries': profile == 'debug' and self.capture_config.full_world_boundaries}
        errors = list(frame_summary['errors'])
        warnings = list(frame_summary['warnings'])
        return {'schema_version': 'trajectory-quality.v1', 'episode_id': self.episode_context.episode_id, 'structural_valid': True, 'quality_valid': not errors, 'capture_capabilities': capabilities, 'checks': {'records': {'valid': True}, 'references': {'valid': True}, 'frames': {'valid': not errors, 'summary': json_deep_copy(frame_summary)}, 'leakage': {'valid': True}, 'effects': {'valid': True, 'status': 'not_evaluated_e0'}, 'split': {'valid': True}, 'hashes': {'valid': True}}, 'errors': errors, 'warnings': warnings, 'exporter_eligibility': {'raw': not errors, 'stream': bool(frame_summary['stream_export_eligible']) and profile in {'training', 'debug'}}, 'exclusion_reasons': list(exclusion_reasons)}

    def _build_manifest(self, *, terminal: Mapping[str, Any], frame_summary: Mapping[str, Any], eligibility: Mapping[str, Any], exclusion_reasons: list[str], artifacts: list[Mapping[str, Any]], elapsed_seconds: float) -> Mapping[str, Any]:
        context = self.episode_context
        task = context.task.identity.identity_task
        source = task['source']
        protocol = _document_content(self.run_context.protocol, 'protocol')
        protocol_hash = sha256_json(protocol)
        assets = protocol.get('assets') if isinstance(protocol, Mapping) else {}
        source_resolution = next((item for item in context.task.identity.path_resolutions if item.json_pointer == '/source/source_json'), None)
        ended = terminal['ended_at_utc'] if 'ended_at_utc' in terminal else _utc_now()
        records = _read_jsonl_rows(self.partial_root / 'trajectory.jsonl')
        model_calls = _read_optional_jsonl_rows(self.partial_root / 'public/model_io.jsonl')
        prompt_tokens, completion_tokens = _sum_tokens(model_calls)
        seeds = {'base': context.episode_key_payload['base_seed']}
        seed_usage = terminal.get('seed_usage', {})
        for namespace in SEED_NAMESPACES:
            usage = seed_usage.get(namespace)
            seeds[namespace] = {'value': context.seeds[namespace], 'consumed': usage['consumed'] if usage is not None else False, 'consumer': usage['consumer'] if usage is not None else None, 'reason': usage['reason'] if usage is not None else 'not_recorded_e0'}
        run_manifest = self.run_context.manifest
        versions = {'common_contract': {'schema_version': 'trajectory-common-contract.v2', 'sha256': COMMON_CONTRACT_SHA256}, 'protocol': protocol_hash, 'tool_schema': (protocol.get('tool_schema') or {}).get('version'), 'policy': run_manifest['policy_hash'], 'capture': run_manifest['capture_config_hash'], 'code': run_manifest['code_commit'], 'renderer': (run_manifest.get('runtime_contract') or {}).get('renderer_version'), 'simulator': (run_manifest.get('runtime_contract') or {}).get('simulator_version')}
        counts = {'records': len(records), 'actions': sum((row['record_type'] == 'action_start' for row in records)), 'low_level_actions': sum((row['record_type'] == 'low_level_action' for row in records)), 'stream_observations': sum((row['record_type'] == 'stream_observation' for row in records)), 'frames': len(self._frame_records), 'interventions': len({row['intervention_id'] for row in records if row['record_type'] == 'intervention'}), 'model_calls': len(model_calls), 'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens}
        terminal_mapping = {'episode_status': terminal['episode_status'], 'done_reason': terminal['done_reason'], 'error_class': terminal['error_class'], 'error_code': terminal['error_code']}
        quality_summary = {'valid': bool(frame_summary['quality_valid']), 'errors': list(frame_summary['errors']), 'warnings': list(frame_summary['warnings'])}
        return {'schema_version': 'episode-manifest.v1', 'contract_version': CONTRACT_VERSION, 'bundle_status': 'complete', 'backend': 'scoper.benchmark.behavior', 'run_id': self.run_context.run_id, 'episode_id': context.episode_id, 'episode_key': context.episode_key, 'episode_key_payload': json_deep_copy(context.episode_key_payload), 'task': {'id': task['id'], 'hash': context.task.identity.task_hash, 'family': task.get('family') or task.get('category'), 'template_id': source.get('template_id') or (task.get('template') or {}).get('template_id'), 'source_split': context.source_split, 'task_split': context.task_split, 'lineage_id': run_manifest['task_lineage_id'], 'task_bank_hash': run_manifest['task_bank_hash']}, 'scene': {'scene_id': source['scene_model'], 'layout_id': source.get('layout_variant_id') or '', 'source_json_content_sha256': None if source_resolution is None else source_resolution.content_sha256, 'asset_version': {'behavior_assets': (assets or {}).get('behavior_assets_version'), 'scene_profile': (assets or {}).get('scene_profile_version')}}, 'rollout': {'purpose': context.rollout_purpose, 'arm': context.episode_key_payload['rollout_arm'], 'execution_profile': context.execution_profile, 'repeat_id': context.episode_key_payload['repeat_id']}, 'eligibility': json_deep_copy(eligibility), 'seeds': seeds, 'versions': versions, 'timing': {'started_at_utc': self._started_at_utc, 'ended_at_utc': ended, 'elapsed_seconds': elapsed_seconds}, 'counts': counts, 'terminal': terminal_mapping, 'artifacts': artifacts, 'bundle_content_sha256': sha256_json(artifacts), 'quality_summary': quality_summary}

    def _append_jsonl(self, relative: str, value: Mapping[str, Any]) -> None:
        handle = self._open_handle(relative)
        handle.write(canonical_json_line(value))

    def _open_handle(self, relative: str) -> BinaryIO:
        if relative not in self._handles:
            path = self.partial_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            self._handles[relative] = path.open('ab')
        return self._handles[relative]

    def _flush_all(self, *, sync: bool) -> None:
        for handle in self._handles.values():
            handle.flush()
            if sync:
                os.fsync(handle.fileno())

    def _close_handles(self) -> None:
        for handle in list(self._handles.values()):
            try:
                handle.flush()
                if self.capture_config.fsync:
                    os.fsync(handle.fileno())
            finally:
                handle.close()
        self._handles.clear()

    def _write_capture_state(self, status: str, *, exclusive: bool=False) -> None:
        payload = {'schema_version': 'capture-state.v1', 'status': status, 'episode_id': self.episode_context.episode_id, 'episode_key': self.episode_context.episode_key, 'records_written': self._record_count, 'frames_written': self._frame_count, 'updated_at_utc': _utc_now()}
        path = self.partial_root / 'capture_state.json'
        if exclusive:
            _write_json_file(path, payload, fsync=self.capture_config.fsync, exclusive=True)
        else:
            _atomic_json_replace(path, payload, fsync=self.capture_config.fsync)

    def _require_state(self, expected: str) -> None:
        if self.state != expected:
            raise BundleStructuralError('writer_state_invalid', f'writer state is {self.state}, expected {expected}')

def _validate_contexts(run: RunContext, episode: EpisodeContext, capture: CaptureConfig) -> None:
    if not isinstance(run, RunContext) or not isinstance(episode, EpisodeContext) or (not isinstance(capture, CaptureConfig)):
        raise TrajectoryConfigError('writer_context_invalid', 'writer requires the frozen E0 context dataclasses')
    validate_run_id(run.run_id)
    if run.backend != 'scoper.benchmark.behavior' or episode.run_id != run.run_id:
        raise TrajectoryConfigError('writer_context_invalid', 'run/backend context mismatch')
    if capture.profile not in {'off', 'minimal', 'training', 'debug'} or capture.profile == 'off':
        raise TrajectoryConfigError('capture_config_invalid', 'writer requires minimal, training, or debug profile')
    if capture.keyframe_encoding != 'png' or capture.stream_encoding != 'jpeg' or capture.stream_jpeg_quality != 90:
        raise TrajectoryConfigError('capture_config_invalid', 'v1 frame encoding settings are fixed')
    payload = episode.episode_key_payload
    if payload.get('task_hash') != episode.task.identity.task_hash:
        raise TrajectoryConfigError('writer_context_invalid', 'task hash differs from EpisodeKey payload')
    from scoper.benchmark.behavior.export.ids import EpisodeKeyInput, build_episode_identity
    try:
        key_input = EpisodeKeyInput(**payload)
    except TypeError as exc:
        raise TrajectoryConfigError('writer_context_invalid', f'episode key payload fields are invalid: {exc}') from exc
    expected_key, expected_id = build_episode_identity(key_input)
    if (expected_key, expected_id) != (episode.episode_key, episode.episode_id):
        raise TrajectoryConfigError('writer_context_invalid', 'episode key/id do not match payload')
    if set(episode.seeds) != set(SEED_NAMESPACES):
        raise TrajectoryConfigError('writer_context_invalid', 'episode seeds must contain exactly five namespaces')
    from scoper.benchmark.behavior.export.ids import derive_all_seeds
    if any((isinstance(value, bool) or not isinstance(value, int) for value in episode.seeds.values())):
        raise TrajectoryConfigError('writer_context_invalid', 'episode seed values must be integers, not booleans')
    if dict(episode.seeds) != derive_all_seeds(episode.episode_key):
        raise TrajectoryConfigError('writer_context_invalid', 'episode seeds do not match EpisodeKey')
    if episode.source_split != payload['source_split'] or episode.execution_profile != payload['execution_profile']:
        raise TrajectoryConfigError('writer_context_invalid', 'episode context identity fields conflict')

def _resolve_run_root(root: Path, run_id: str) -> Path:
    candidate = Path(root)
    run_root = candidate if candidate.name == run_id and (candidate / 'run_manifest.json').exists() else candidate / run_id
    if run_root.name.endswith('.initializing') or not run_root.is_dir():
        raise BundleConflictError('run_root_invalid', 'episode writer requires a published final run root')
    return run_root

def _build_event_record(envelope: Mapping[str, Any], *, sequence: int, episode_id: str, start_monotonic_ns: int, last_logical_time: float, last_monotonic_ns: int, has_public: bool, has_privileged: bool) -> Mapping[str, Any]:
    now_relative = max(0, time.monotonic_ns() - start_monotonic_ns)
    monotonic = envelope.get('monotonic_time_ns', max(now_relative, last_monotonic_ns))
    logical = envelope.get('logical_time_s', max(last_logical_time, monotonic / 1000000000.0))
    record = {'schema_version': envelope.get('schema_version', 'trajectory-record.v1'), 'record_id': envelope.get('record_id', make_record_id(sequence)), 'record_type': envelope.get('record_type'), 'episode_id': envelope.get('episode_id', episode_id), 'sequence_index': envelope.get('sequence_index', sequence), 'step_id': envelope.get('step_id', 0), 'decision_id': envelope.get('decision_id'), 'action_id': envelope.get('action_id'), 'low_level_id': envelope.get('low_level_id'), 'stream_id': envelope.get('stream_id'), 'stream_obs_id': envelope.get('stream_obs_id'), 'intervention_id': envelope.get('intervention_id'), 'model_call_id': envelope.get('model_call_id'), 'logical_time_s': logical, 'monotonic_time_ns': monotonic, 'wall_time_utc': envelope.get('wall_time_utc', _utc_now()), 'public_payload': has_public, 'privileged_payload': has_privileged}
    unknown = set(envelope) - set(record)
    if unknown:
        raise TrajectoryConfigError('event_schema_invalid', f'unknown event envelope fields: {sorted(unknown)}')
    return record

def _encode_rgb(array: np.ndarray, *, encoding: str, jpeg_quality: int) -> tuple[bytes, str, Mapping[str, Any]]:
    image = Image.fromarray(array, mode='RGB')
    buffer = io.BytesIO()
    if encoding == 'png':
        image.save(buffer, format='PNG', compress_level=6, optimize=False)
        return (buffer.getvalue(), 'image/png', {'compress_level': 6, 'optimize': False})
    if encoding == 'jpeg':
        image.save(buffer, format='JPEG', quality=jpeg_quality, subsampling=0, optimize=False, progressive=False)
        return (buffer.getvalue(), 'image/jpeg', {'quality': jpeg_quality, 'subsampling': 0, 'optimize': False, 'progressive': False})
    raise TrajectoryConfigError('frame_encoding_invalid', f'unsupported frame encoding {encoding!r}')

def _document_content(value: Mapping[str, Any], expected_type: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TrajectoryConfigError('provenance_invalid', f'{expected_type} context must be a mapping')
    if value.get('schema_version') == 'provenance-document.v1':
        validate_provenance_document(value, document_type=expected_type)
        content = value['content']
    else:
        content = value
    if not isinstance(content, Mapping):
        raise TrajectoryConfigError('provenance_invalid', f'{expected_type} content must be a mapping')
    return content

def _normalized_context_document(value: Mapping[str, Any], expected_type: str) -> Mapping[str, Any]:
    if value.get('schema_version') == 'provenance-document.v1':
        validate_provenance_document(value, document_type=expected_type)
        return value
    content = _document_content(value, expected_type)
    return {'schema_version': 'provenance-document.v1', 'document_type': expected_type, 'content_hash': sha256_json(content), 'content': content}

def _write_json_file(path: Path, value: Mapping[str, Any], *, fsync: bool, exclusive: bool) -> None:
    _write_bytes(path, canonical_json_line(value), fsync=fsync, exclusive=exclusive)

def _write_bytes(path: Path, data: bytes, *, fsync: bool, exclusive: bool) -> None:
    flags = os.O_WRONLY | os.O_CREAT | (os.O_EXCL if exclusive else os.O_TRUNC)
    descriptor = os.open(path, flags, 420)
    with os.fdopen(descriptor, 'wb') as handle:
        handle.write(data)
        handle.flush()
        if fsync:
            os.fsync(handle.fileno())

def _atomic_json_replace(path: Path, value: Mapping[str, Any], *, fsync: bool) -> None:
    temp = path.with_name(path.name + '.tmp')
    if temp.exists():
        temp.unlink()
    _write_json_file(temp, value, fsync=fsync, exclusive=True)
    os.replace(temp, path)
    if fsync:
        _fsync_directory(path.parent)

def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

def _make_episode_read_only(root: Path) -> None:
    for path in sorted(root.rglob('*'), key=lambda item: len(item.parts), reverse=True):
        if path.is_file():
            path.chmod(292)
        elif path.is_dir():
            path.chmod(365)
    root.chmod(365)

def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')

def _archive_incomplete_episode(run_root: Path, partial_root: Path, episode_id: str) -> None:
    base = run_root / 'incomplete' / episode_id
    base.mkdir(parents=True, exist_ok=True)
    attempt = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    target = base / attempt
    ordinal = 1
    while target.exists():
        target = base / f'{attempt}-{ordinal:03d}'
        ordinal += 1
    original_relative = partial_root.relative_to(run_root).as_posix()
    os.replace(partial_root, target)
    row = {'source_path': original_relative, 'archived_path': target.relative_to(run_root).as_posix(), 'reason': 'resume_restart_from_reset', 'archived_at_utc': _utc_now()}
    _append_locked_jsonl(run_root / 'incomplete_index.jsonl', row)

def _episode_index_row(manifest: Mapping[str, Any], manifest_sha256: str, *, status: str) -> Mapping[str, Any]:
    return {'episode_id': manifest['episode_id'], 'episode_key': manifest['episode_key'], 'task_id': manifest['task']['id'], 'repeat_id': manifest['rollout']['repeat_id'], 'status': status, 'manifest_path': f"episodes/{manifest['episode_id']}/manifest.json", 'manifest_sha256': manifest_sha256, 'train_eligible': manifest['eligibility']['train_eligible'], 'error_class': manifest['terminal']['error_class']}

def _ensure_episode_index_row(run_root: Path, row: Mapping[str, Any], *, repair_only: bool) -> None:
    path = run_root / 'episodes.jsonl'
    path.touch(exist_ok=True)
    with path.open('a+b') as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.seek(0)
            existing = []
            for number, raw in enumerate(handle, 1):
                value = strict_json_loads(raw, source=f'{path}:{number}')
                if not isinstance(value, Mapping):
                    raise BundleConflictError('episodes_index_invalid', 'episodes index row is not a mapping')
                existing.append(value)
            matching = [item for item in existing if item.get('episode_id') == row['episode_id']]
            if len(matching) > 1:
                raise BundleConflictError('episodes_index_invalid', 'episode id occurs more than once in run index')
            if matching:
                current = matching[0]
                stable_fields = set(row) - {'status'}
                if any((current.get(field) != row.get(field) for field in stable_fields)):
                    raise BundleConflictError('episodes_index_conflict', 'existing episode index row conflicts')
                return
            repaired = dict(row)
            if repair_only:
                repaired['status'] = 'repaired_index'
            handle.seek(0, os.SEEK_END)
            handle.write(canonical_json_line(repaired))
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

def _append_locked_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    with path.open('a+b') as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.seek(0, os.SEEK_END)
            handle.write(canonical_json_line(row))
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

def _read_jsonl_rows(path: Path) -> list[Mapping[str, Any]]:
    rows = _read_optional_jsonl_rows(path)
    if not path.is_file():
        raise BundleStructuralError('artifact_missing', f'required JSONL missing: {path}')
    return rows

def _read_optional_jsonl_rows(path: Path) -> list[Mapping[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open('rb') as handle:
        for number, raw in enumerate(handle, 1):
            value = strict_json_loads(raw, source=f'{path}:{number}')
            if not isinstance(value, Mapping):
                raise BundleStructuralError('jsonl_schema_invalid', f'line {number} in {path} is not a mapping')
            rows.append(value)
    return rows

def _sum_tokens(model_calls: list[Mapping[str, Any]]) -> tuple[int | None, int | None]:
    prompt_values = []
    completion_values = []
    for call in model_calls:
        usage = call.get('usage')
        if isinstance(usage, Mapping):
            if isinstance(usage.get('prompt_tokens'), int):
                prompt_values.append(usage['prompt_tokens'])
            if isinstance(usage.get('completion_tokens'), int):
                completion_values.append(usage['completion_tokens'])
    return (sum(prompt_values) if prompt_values else None, sum(completion_values) if completion_values else None)
__all__ = ['EpisodeBundleWriter']
