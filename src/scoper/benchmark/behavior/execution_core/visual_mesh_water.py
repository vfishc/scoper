from __future__ import annotations
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any
import numpy as np

@dataclass(frozen=True)
class VisualMeshWaterConfig:
    color: tuple[float, float, float] = (0.0, 0.82, 1.0)
    body_opacity: float = 0.48
    surface_opacity: float = 0.88
    roughness: float = 0.03
    radius_scale: float = 0.34
    height_scale: float = 0.7
    bottom_margin_scale: float = 0.18
    top_margin_scale: float = 0.14
    min_radius: float = 0.012
    min_height: float = 0.006
    surface_height: float = 0.004
    surface_lift: float = 0.001
    surface_radius_scale: float = 1.02

def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, 'detach'):
        value = value.detach()
    if hasattr(value, 'cpu'):
        value = value.cpu()
    if hasattr(value, 'numpy'):
        value = value.numpy()
    return np.array(value)

def as_float_triplet(value: Any) -> tuple[float, float, float]:
    arr = to_numpy(value).reshape(-1)
    return (float(arr[0]), float(arr[1]), float(arr[2]))

def quat_xyzw_inverse_rotate(vec: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = quat_xyzw
    q = np.array([w, x, y, z], dtype=float)
    q = q / max(np.linalg.norm(q), 1e-08)
    w, x, y, z = q
    rot = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], dtype=float)
    return rot.T @ vec

def object_aabb_local_center(obj: Any) -> tuple[float, float, float]:
    try:
        aabb_center = np.array(as_float_triplet(getattr(obj, 'aabb_center')), dtype=float)
        root_position, root_orientation = obj.get_position_orientation(frame='scene')
        root_position_np = np.array(as_float_triplet(root_position), dtype=float)
        root_orientation_np = np.array(to_numpy(root_orientation).reshape(-1)[:4], dtype=float)
        local = quat_xyzw_inverse_rotate(aabb_center - root_position_np, root_orientation_np)
        return (float(local[0]), float(local[1]), float(local[2]))
    except Exception:
        return (0.0, 0.0, 0.0)

def visual_mesh_water_geometry(container: Any, config: VisualMeshWaterConfig) -> dict[str, Any]:
    extent = as_float_triplet(getattr(container, 'aabb_extent', (0.122967, 0.141234, 0.142378)))
    local_center = object_aabb_local_center(container)
    min_xy_extent = max(min(extent[0], extent[1]), 1e-06)
    z_extent = max(extent[2], 1e-06)
    radius = max(min_xy_extent * config.radius_scale, config.min_radius)
    bottom_margin = max(z_extent * config.bottom_margin_scale, config.min_height)
    top_margin = max(z_extent * config.top_margin_scale, config.min_height)
    available_height = max(z_extent - bottom_margin - top_margin, config.min_height)
    height = min(max(z_extent * config.height_scale, config.min_height), available_height)
    bottom_z = local_center[2] - z_extent * 0.5 + bottom_margin
    center_z = bottom_z + height * 0.5
    surface_z = bottom_z + height + config.surface_lift
    return {'aabb_extent': list(extent), 'aabb_local_center': list(local_center), 'radius': float(radius), 'height': float(height), 'body_translation': [float(local_center[0]), float(local_center[1]), float(center_z)], 'surface_translation': [float(local_center[0]), float(local_center[1]), float(surface_z)], 'surface_height': float(config.surface_height), 'bottom_margin': float(bottom_margin), 'top_margin': float(top_margin)}

def make_preview_material(stage: Any, material_path: str, color: list[float], opacity: float, roughness: float) -> dict[str, Any]:
    from pxr import Gf, Sdf, UsdShade
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, f'{material_path}/Shader')
    shader.CreateIdAttr('UsdPreviewSurface')

    def set_input(name: str, value_type: Any, value: Any) -> None:
        inp = shader.GetInput(name)
        if not inp:
            inp = shader.CreateInput(name, value_type)
        inp.Set(value)
    set_input('diffuseColor', Sdf.ValueTypeNames.Color3f, Gf.Vec3f(*color))
    set_input('opacity', Sdf.ValueTypeNames.Float, float(opacity))
    set_input('roughness', Sdf.ValueTypeNames.Float, float(roughness))
    set_input('metallic', Sdf.ValueTypeNames.Float, 0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    return {'material_path': material_path, 'shader_path': f'{material_path}/Shader', 'color': list(color), 'opacity': float(opacity), 'roughness': float(roughness), 'method': 'UsdPreviewSurface'}

def create_cylinder(stage: Any, prim_path: str, radius: float, height: float, translation: list[float], material_path: str, color: list[float], opacity: float) -> dict[str, Any]:
    from pxr import Gf, UsdGeom, UsdShade
    cylinder = UsdGeom.Cylinder.Define(stage, prim_path)
    prim = cylinder.GetPrim()
    cylinder.GetRadiusAttr().Set(float(radius))
    cylinder.GetHeightAttr().Set(float(height))
    cylinder.GetExtentAttr().Set([Gf.Vec3f(-float(radius), -float(radius), -float(height) * 0.5), Gf.Vec3f(float(radius), float(radius), float(height) * 0.5)])
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(Gf.Vec3d(*translation))
    gprim = UsdGeom.Gprim(prim)
    gprim.CreateDisplayColorAttr().Set([Gf.Vec3f(*color)])
    gprim.CreateDisplayOpacityAttr().Set([float(opacity)])
    material_prim = stage.GetPrimAtPath(material_path)
    if material_prim.IsValid():
        UsdShade.MaterialBindingAPI(prim).Bind(UsdShade.Material(material_prim), UsdShade.Tokens.strongerThanDescendants)
    UsdGeom.Imageable(prim).MakeVisible()
    return {'success': True, 'prim_path': prim_path, 'radius': float(radius), 'height': float(height), 'translation': [float(v) for v in translation], 'material_path': material_path, 'display_color': list(color), 'display_opacity': float(opacity), 'method': 'UsdGeom.Cylinder'}

def create_visual_mesh_water(stage: Any, container: Any, config: VisualMeshWaterConfig, *, body_path: str, surface_path: str, body_material_path: str, surface_material_path: str, edit_context: Any | None=None) -> dict[str, Any]:
    color = [float(v) for v in config.color]
    geometry = visual_mesh_water_geometry(container, config)
    context = edit_context if edit_context is not None else nullcontext()
    with context:
        body_material = make_preview_material(stage, body_material_path, color=color, opacity=config.body_opacity, roughness=config.roughness)
        surface_material = make_preview_material(stage, surface_material_path, color=color, opacity=config.surface_opacity, roughness=config.roughness)
        body = create_cylinder(stage, body_path, radius=geometry['radius'], height=geometry['height'], translation=geometry['body_translation'], material_path=body_material['material_path'], color=color, opacity=config.body_opacity)
        surface = create_cylinder(stage, surface_path, radius=geometry['radius'] * config.surface_radius_scale, height=geometry['surface_height'], translation=geometry['surface_translation'], material_path=surface_material['material_path'], color=color, opacity=config.surface_opacity)
    return {'success': True, 'geometry': geometry, 'body': body, 'surface': surface, 'body_material': body_material, 'surface_material': surface_material}

def set_visual_mesh_water_visibility(stage: Any, prim_paths: tuple[str, str], visible: bool, edit_context: Any | None=None) -> None:
    from pxr import UsdGeom
    context = edit_context if edit_context is not None else nullcontext()
    with context:
        for prim_path in prim_paths:
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                continue
            imageable = UsdGeom.Imageable(prim)
            if visible:
                imageable.MakeVisible()
            else:
                imageable.MakeInvisible()

class ContainerWaterVisual:

    def __init__(self, og_module: Any, container: Any, substance: str='water', config: VisualMeshWaterConfig | None=None) -> None:
        self._og = og_module
        self._container = container
        self._substance = substance
        self._config = config or VisualMeshWaterConfig()
        self._filled = False
        self._created = False

    @property
    def is_filled(self) -> bool:
        return self._filled

    @property
    def container(self) -> Any:
        return self._container

    @property
    def substance(self) -> str:
        return self._substance

    @property
    def body_prim_path(self) -> str:
        token = self._safe_token(self._substance)
        return f'{self._parent_prim_path()}/VisualWaterBody_{token}'

    @property
    def surface_prim_path(self) -> str:
        token = self._safe_token(self._substance)
        return f'{self._parent_prim_path()}/VisualWaterSurface_{token}'

    @property
    def prim_paths(self) -> tuple[str, str]:
        return (self.body_prim_path, self.surface_prim_path)

    def fill(self) -> dict[str, Any]:
        if self._filled and self._created:
            return {'success': True, 'already_filled': True}
        stage = self._og.sim.stage
        token = self._safe_token(self._substance)
        if self._created:
            set_visual_mesh_water_visibility(stage, self.prim_paths, True, edit_context=self._edit_context())
        else:
            water = create_visual_mesh_water(stage, self._container, self._config, body_path=self.body_prim_path, surface_path=self.surface_prim_path, body_material_path=f'/World/Looks/visual_mesh_{token}_water_body', surface_material_path=f'/World/Looks/visual_mesh_{token}_water_surface', edit_context=self._edit_context())
            try:
                self._og.sim.step()
            except Exception:
                pass
            self._created = True
        self._filled = True
        return {'success': True, 'object': getattr(self._container, 'name', ''), 'substance': self._substance, 'body_prim_path': self.body_prim_path, 'surface_prim_path': self.surface_prim_path, 'visible': True}

    def empty(self) -> dict[str, Any]:
        if not self._filled:
            return {'success': True, 'already_empty': True}
        stage = self._og.sim.stage
        set_visual_mesh_water_visibility(stage, self.prim_paths, False, edit_context=self._edit_context())
        self._filled = False
        return {'success': True, 'object': getattr(self._container, 'name', ''), 'substance': self._substance, 'body_prim_path': self.body_prim_path, 'surface_prim_path': self.surface_prim_path, 'visible': False}

    def transfer_to(self, target: 'ContainerWaterVisual') -> dict[str, Any]:
        source_info = self.empty()
        target_info = target.fill()
        return {'success': source_info['success'] and target_info['success'], 'source': source_info, 'target': target_info}

    def _parent_prim_path(self) -> str:
        root_link = getattr(self._container, 'root_link', None)
        if root_link is not None:
            path = getattr(root_link, 'prim_path', None)
            if path:
                return str(path)
        return str(getattr(self._container, 'prim_path', '/World/unknown'))

    def _edit_context(self) -> Any:
        try:
            return self._og.sim.editing_usd()
        except Exception:
            return nullcontext()

    @staticmethod
    def _safe_token(value: str) -> str:
        token = ''.join((ch if ch.isalnum() else '_' for ch in str(value)))
        return token.strip('_') or 'liquid'
