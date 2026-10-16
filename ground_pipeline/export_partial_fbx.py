import base64
import json
import math
import os
import sys
import tempfile
from array import array
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

response_path = Path(sys.argv[-2])
output_path = Path(sys.argv[-1])
blend_path = output_path if output_path.suffix.lower() == ".blend" else output_path.with_suffix(".blend")
with response_path.open("r", encoding="utf-8-sig") as f:
    data = json.load(f)
if data.get("person_points") and len(data["person_points"]) > 20000:
    stride = math.ceil(len(data["person_points"]) / 20000)
    data["person_points"] = data["person_points"][::stride][:20000]

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
    coordinates = array("f", (component for point in points for component in point))
    mesh = bpy.data.meshes.new(name + "Mesh")
    mesh.vertices.add(len(coordinates) // 3)
    mesh.vertices.foreach_set("co", coordinates)
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
person_raw = None
if data.get("person_points"):
    person_raw = new_vertices_object("API_person_points_camera_xyz", data["person_points"])

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

# Display returned ground inliers as a lightweight vertex cloud in the ground-aligned frame.
cloud = new_vertices_object("Ground_Inlier_Samples_Display", [to_world(p) for p in data["inlier_points"]])

# Display returned person samples as a lightweight vertex cloud in the ground-aligned frame.
person_display = None
if person_raw is not None:
    full_person_points = data["person_points"]
    display_limit = 20000
    stride = max(1, math.ceil(len(full_person_points) / display_limit))
    display_points = full_person_points[::stride][:display_limit]
    person_display = new_vertices_object("Person_PointCloud_Display_Sampled", [to_world(p) for p in display_points])
    person_display["display_count"] = len(display_points)
    person_display["source_count"] = len(full_person_points)
    person_display["sampling_stride"] = stride

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

# Visualize the API's person bbox and centroids; the API does not return person point samples.
person_objects = []
bbox_values = data.get("person_bbox_3d_m")
if bbox_values is not None:
    lo = bbox_values[:3]
    hi = bbox_values[3:]
    camera_corners = [
        (x, y, z)
        for x in (lo[0], hi[0])
        for y in (lo[1], hi[1])
        for z in (lo[2], hi[2])
    ]
    corner_order = [0, 1, 3, 2, 4, 5, 7, 6]
    edge_pairs = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    bbox_mesh = bpy.data.meshes.new("PersonBBoxMesh")
    bbox_mesh.from_pydata([to_world(p) for p in camera_corners], [(corner_order[a], corner_order[b]) for a, b in edge_pairs], [])
    bbox_mesh.update()
    bbox_obj = bpy.data.objects.new("Person_3D_BBox_Estimated", bbox_mesh)
    scene.collection.objects.link(bbox_obj)
    bbox_obj["source"] = "API person_bbox_3d_m; estimated AABB in camera coordinates"
    person_objects.append(bbox_obj)

    for field, name, radius in (("person_centroid_m", "Person_Centroid_Estimated", 0.045), ("contact_centroid_m", "Person_GroundContact_Centroid_Estimated", 0.06)):
        value = data.get(field)
        if value is None:
            continue
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=radius, location=to_world(value))
        marker = bpy.context.object
        marker.name = name
        marker["source_field"] = field
        marker["camera_xyz_m"] = json.dumps(value, separators=(",", ":"))
        person_objects.append(marker)

# Preserve API scalar/array fields as FBX custom properties, excluding point arrays represented above.
metadata = bpy.data.objects.new("API_Response_Metadata", None)
scene.collection.objects.link(metadata)
metadata["coordinate_note"] = "Raw API point objects are in camera xyz; ground mesh is ground-aligned local world xyz."
for key, value in data.items():
    if key in {"plane_points", "inlier_points", "person_points", "masks", "depth_b64", "visualization"}:
        continue
    metadata[key] = json.dumps(value, separators=(",", ":")) if isinstance(value, (list, dict)) else value
metadata["visualization_keys"] = json.dumps(list(data.get("visualization", {}).keys()))
metadata["mask_keys"] = json.dumps(list(data.get("masks", {}).keys()))
if data.get("depth_b64"):
    depth_path = output_path.with_name(output_path.stem + "_depth.npy")
    depth_path.write_bytes(base64.b64decode(data["depth_b64"]))
    metadata["depth_npy_filename"] = depth_path.name
    depth_text = bpy.data.texts.new("API_depth_b64_npy")
    depth_text.write(data["depth_b64"])

# Embed API visualization and segmentation masks as textured image cards.
image_cards = []
image_tmp_paths = []
image_items = list(data.get("visualization", {}).items())
image_items.extend(("mask_" + key, value) for key, value in data.get("masks", {}).items() if value)
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
export_objects = [plane_raw, inliers_raw, *([person_raw] if person_raw else []), ground, cloud, *([person_display] if person_display else []), camera, metadata, *person_objects, *image_cards]
for obj in export_objects:
    obj.select_set(True)
bpy.context.view_layer.objects.active = ground
blend_path.parent.mkdir(parents=True, exist_ok=True)
for temp_name in image_tmp_paths:
    try:
        os.remove(temp_name)
    except OSError:
        pass
bpy.ops.wm.save_as_mainfile(filepath=str(blend_path), compress=True)
print("BLEND_SAVED", blend_path, blend_path.stat().st_size)
print("OBJECTS", [(obj.name, obj.type, len(obj.data.vertices) if obj.type == "MESH" else None) for obj in export_objects])
print("EMBEDDED_IMAGES", [obj.name for obj in image_cards])
