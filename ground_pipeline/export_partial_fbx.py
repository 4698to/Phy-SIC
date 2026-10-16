import base64
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

response_path = Path(sys.argv[-2])
output_path = Path(sys.argv[-1])
blend_path = output_path.with_suffix(".blend")
with response_path.open("r", encoding="utf-8-sig") as f:
    data = json.load(f)

normal = Vector(data["plane_normal"]).normalized()
up = Vector((0.0, 0.0, -1.0))
v = normal.cross(up)
s = v.length
c = normal.dot(up)
if s < 1e-9:
    rotation = Matrix.Identity(3) if c > 0 else Matrix.Diagonal((-1.0, -1.0, -1.0))
else:
    vx = Matrix(((0.0, -v.z, v.y), (v.z, 0.0, -v.x), (-v.y, v.x, 0.0)))
    rotation = Matrix.Identity(3) + vx + (vx @ vx) * ((1.0 - c) / (s * s))
plane_d = float(data["plane_d"])

def to_world(point):
    q = rotation @ Vector(point)
    q.z -= plane_d
    return tuple(q)

def new_vertices_object(name, points):
    mesh = bpy.data.meshes.new(name + "Mesh")
    mesh.from_pydata(points, [], [])
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    scene.collection.objects.link(obj)
    return obj

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
for collection in list(bpy.data.collections):
    if collection.name != "Collection":
        bpy.data.collections.remove(collection)
scene = bpy.context.scene
scene.unit_settings.system = "METRIC"

# Keep the exact API point arrays in the original camera coordinate system.
plane_raw = new_vertices_object("API_plane_points_camera_xyz", data["plane_points"])
inliers_raw = new_vertices_object("API_inlier_points_camera_xyz", data["inlier_points"])

# Ground plane surface transformed to the ground-aligned local world frame.
steps = int(round(math.sqrt(len(data["plane_points"]))))
if steps * steps != len(data["plane_points"]):
    raise ValueError("plane_points must form a square grid")
vertices = [to_world(p) for p in data["plane_points"]]
faces = []
for row in range(steps - 1):
    for col in range(steps - 1):
        a = row * steps + col
        faces.append((a, a + 1, a + steps + 1, a + steps))
mesh = bpy.data.meshes.new("GroundPlaneMesh")
mesh.from_pydata(vertices, [], faces)
mesh.update()
ground = bpy.data.objects.new("Ground_Plane_World_Local", mesh)
scene.collection.objects.link(ground)

# Display mesh for the returned ground inlier sample points, preserving exact raw vertices above.
base_verts = [Vector((0.0, 0.0, 0.0))]
for i in range(1, 3):
    z = 1.0 - i * 2.0 / 3.0
    radius = math.sqrt(1.0 - z * z)
    for j in range(5):
        angle = 2.0 * math.pi * j / 5.0
        base_verts.append(Vector((radius * math.cos(angle), radius * math.sin(angle), z)))
base_verts.append(Vector((0.0, 0.0, -1.0)))
base_faces = []
for j in range(5):
    jn = (j + 1) % 5
    base_faces.extend(((0, 1 + j, 1 + jn), (11, 6 + jn, 6 + j), (1 + j, 6 + j, 6 + jn, 1 + jn)))
point_radius = 0.012
cloud_verts = []
cloud_faces = []
for point in data["inlier_points"]:
    center = Vector(to_world(point))
    offset = len(cloud_verts)
    cloud_verts.extend(tuple(center + p * point_radius) for p in base_verts)
    cloud_faces.extend(tuple(offset + i for i in face) for face in base_faces)
cloud_mesh = bpy.data.meshes.new("GroundInlierMarkersMesh")
cloud_mesh.from_pydata(cloud_verts, [], cloud_faces)
cloud_mesh.update()
cloud = bpy.data.objects.new("Ground_Inlier_Samples_Display", cloud_mesh)
scene.collection.objects.link(cloud)

# Approximate camera pose and intrinsics in the ground-aligned local frame.
camera_data = bpy.data.cameras.new("EstimatedCamera")
camera_data.lens = float(data["focal_px"]) * 36.0 / float(data["image_hw"][1])
camera_data.sensor_width = 36.0
camera = bpy.data.objects.new("Camera_Estimated_Local", camera_data)
scene.collection.objects.link(camera)
camera.location = (0.0, 0.0, -plane_d)
right = (rotation @ Vector((1.0, 0.0, 0.0))).normalized()
cam_up = (rotation @ Vector((0.0, -1.0, 0.0))).normalized()
forward = (rotation @ Vector((0.0, 0.0, 1.0))).normalized()
camera.rotation_mode = "QUATERNION"
camera.rotation_quaternion = Matrix((right, cam_up, -forward)).transposed().to_quaternion()
scene.camera = camera

# Preserve API scalar/array fields as FBX custom properties, excluding point arrays represented above.
metadata = bpy.data.objects.new("API_Response_Metadata", None)
scene.collection.objects.link(metadata)
metadata["coordinate_note"] = "Raw API point objects are in camera xyz; ground mesh is ground-aligned local world xyz."
for key, value in data.items():
    if key in {"plane_points", "inlier_points", "visualization"}:
        continue
    metadata[key] = json.dumps(value, separators=(",", ":")) if isinstance(value, (list, dict)) else value
metadata["visualization_keys"] = json.dumps(list(data.get("visualization", {}).keys()))

# Embed the three API visualization images as textured image cards in the FBX.
image_cards = []
image_tmp_paths = []
image_items = list(data.get("visualization", {}).items())
card_x = 2.0
card_gap = 0.12
card_h = 0.8
for index, (key, uri) in enumerate(image_items):
    prefix = "data:image/png;base64,"
    if not isinstance(uri, str) or not uri.startswith(prefix):
        metadata["visualization_" + key] = str(uri)
        continue
    raw = base64.b64decode(uri[len(prefix):])
    fd, temp_name = tempfile.mkstemp(suffix=".png", prefix="ground_api_")
    os.close(fd)
    Path(temp_name).write_bytes(raw)
    image_tmp_paths.append(temp_name)
    image = bpy.data.images.load(temp_name, check_existing=False)
    image.name = "API_visualization_" + key
    image.pack()
    width, height = image.size
    card_w = card_h * width / height
    x0 = card_x
    x1 = x0 + card_w
    y0 = index * (card_h + card_gap)
    y1 = y0 + card_h
    card_mesh = bpy.data.meshes.new("VisualizationCard_" + key + "Mesh")
    card_mesh.from_pydata([(x0, y0, 0.0), (x1, y0, 0.0), (x1, y1, 0.0), (x0, y1, 0.0)], [], [(0, 1, 2, 3)])
    card_mesh.update()
    uv = card_mesh.uv_layers.new(name="UVMap")
    for loop_index, coord in zip(card_mesh.polygons[0].loop_indices, ((0, 0), (1, 0), (1, 1), (0, 1))):
        uv.data[loop_index].uv = coord
    card = bpy.data.objects.new("API_Visualization_" + key, card_mesh)
    scene.collection.objects.link(card)
    material = bpy.data.materials.new("API_Visualization_Material_" + key)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    image_node = nodes.new("ShaderNodeTexImage")
    image_node.image = image
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    output_node = nodes.new("ShaderNodeOutputMaterial")
    material.node_tree.links.new(image_node.outputs["Color"], bsdf.inputs["Base Color"])
    material.node_tree.links.new(bsdf.outputs["BSDF"], output_node.inputs["Surface"])
    card.data.materials.append(material)
    image_cards.append(card)
    card_x += card_w + card_gap

# Export selected asset objects and custom properties; packed texture files are embedded.
for obj in bpy.context.selected_objects:
    obj.select_set(False)
export_objects = [plane_raw, inliers_raw, ground, cloud, camera, metadata, *image_cards]
for obj in export_objects:
    obj.select_set(True)
bpy.context.view_layer.objects.active = ground
output_path.parent.mkdir(parents=True, exist_ok=True)
bpy.ops.export_scene.fbx(
    filepath=str(output_path),
    use_selection=True,
    apply_unit_scale=True,
    global_scale=1.0,
    axis_forward="-Z",
    axis_up="Y",
    object_types={"MESH", "CAMERA", "EMPTY"},
    use_custom_props=True,
    path_mode="COPY",
    embed_textures=True,
    bake_anim=False,
)
for temp_name in image_tmp_paths:
    try:
        os.remove(temp_name)
    except OSError:
        pass
print("FBX_EXPORTED", output_path, output_path.stat().st_size)
bpy.ops.wm.save_as_mainfile(filepath=str(blend_path), compress=True)
print("BLEND_SAVED", blend_path, blend_path.stat().st_size)
print("OBJECTS", [(obj.name, obj.type, len(obj.data.vertices) if obj.type == "MESH" else None) for obj in export_objects])
print("EMBEDDED_VISUALIZATIONS", [obj.name for obj in image_cards])
