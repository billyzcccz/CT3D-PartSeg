"""CT3D Part-Level 3D Segmentation Backend (Blender + YOLOv11-seg + Ray-Cast Boolean Separation).

Run inside Blender's Python or an environment where `bpy` is installed:
    blender --background --python backend/segment_pipeline.py
"""

from math import radians
import math
import os
import shutil
import tempfile

import bpy
import cv2
import numpy as np
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from mathutils import Vector
from ultralytics import YOLO
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app, origins="*")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS = os.path.abspath(os.path.join(BASE_DIR, "..", "weights", "best.pt"))
WEIGHTS_PATH = os.environ.get("YOLO_WEIGHTS_PATH", DEFAULT_WEIGHTS)
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {"glb"}

_YOLO_MODEL = None


def get_yolo_model():
    """Load and cache the YOLOv11-seg model."""
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        _YOLO_MODEL = YOLO(WEIGHTS_PATH)
    return _YOLO_MODEL


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def reset_scene():
    """Reset Blender scene between segmentation requests to prevent stale meshes."""
    bpy.ops.wm.read_factory_settings(use_empty=True)


def setup_camera(view_type):
    """Create an orthographic camera positioned along the specified canonical view."""
    bpy.ops.object.camera_add()
    camera = bpy.context.view_layer.objects.active
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = 2

    view_poses = {
        "FRONT": ((0, -10, 0), (radians(90), 0, 0)),
        "BACK": ((0, 10, 0), (radians(90), 0, radians(180))),
        "LEFT": ((-10, 0, 0), (radians(90), 0, radians(270))),
        "RIGHT": ((10, 0, 0), (radians(90), 0, radians(90))),
        "TOP": ((0, 0, 10), (0, 0, 0)),
        "BOTTOM": ((0, 0, -10), (radians(180), 0, 0)),
    }
    loc, rot = view_poses[view_type]
    camera.location = loc
    camera.rotation_euler = rot
    bpy.context.scene.camera = camera
    return camera


def render_view(view_type, output_dir):
    """Render the active orthographic camera view to a PNG file."""
    output_filepath = os.path.join(output_dir, f"{view_type}.png")
    bpy.context.scene.render.filepath = output_filepath
    bpy.context.scene.render.resolution_x = 1920
    bpy.context.scene.render.resolution_y = 1080
    bpy.ops.render.render(write_still=True)


def delete_default_cube():
    """Remove Blender default Cube meshes if present."""
    for obj in list(bpy.context.scene.objects):
        if obj.type == "MESH" and obj.name.startswith("Cube"):
            bpy.data.objects.remove(obj, do_unlink=True)


def render_multiview(input_glb, output_dir):
    """Import GLB mesh and render 6 orthogonal projection views."""
    reset_scene()
    bpy.ops.import_scene.gltf(filepath=input_glb)
    delete_default_cube()

    for view_type in ["FRONT", "BACK", "LEFT", "RIGHT", "TOP", "BOTTOM"]:
        setup_camera(view_type)
        render_view(view_type, output_dir)


def yolo_wheel_detection(image_path):
    """Run YOLOv11-seg on the rendered view and return foreground wheel pixel coordinates."""
    model = get_yolo_model()
    class_names = model.names
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    results = model.predict(source=image_path, conf=0.25)
    coords = []

    for result in results:
        if result.masks is None:
            continue
        masks = result.masks.data.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        for mask, cls_id in zip(masks, classes):
            if class_names[int(cls_id)] == "wheel":
                binary_mask = (mask > 0).astype(np.uint8) * 255
                h, w = image.shape[:2]
                mask_resized = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                found = cv2.findNonZero(mask_resized)
                if found is not None and len(found) > 0:
                    coords = found

    return coords


def find_circle_centers(coords):
    """Split detected wheel points along X median and fit left/right ellipses."""
    x_coords = [coord[0][0] for coord in coords]
    y_coords = [coord[0][1] for coord in coords]
    points = np.array([x_coords, y_coords]).T
    sorted_points = points[np.argsort(points[:, 0])]
    median_x = np.median(sorted_points[:, 0])

    left_points = sorted_points[sorted_points[:, 0] < median_x]
    right_points = sorted_points[sorted_points[:, 0] >= median_x]

    left_circle = cv2.fitEllipse(left_points)
    right_circle = cv2.fitEllipse(right_points)
    return left_circle[0], right_circle[0]


def cast_ray_and_create_cylinder(camera, model, circle_center):
    """Cast a ray from the right-view camera through the fitted wheel center and boolean-intersect."""
    ray_origin = camera.location
    x, y = circle_center
    ray_direction = Vector((-14000, x - 750, y - 1350))
    depsgraph = bpy.context.view_layer.depsgraph

    hit, location, _normal, _index, _obj, _matrix = bpy.context.scene.ray_cast(
        depsgraph, ray_origin, ray_direction
    )
    if not hit:
        return None

    bpy.ops.mesh.primitive_cylinder_add(radius=0.3, depth=0.28, location=location)
    cylinder = bpy.context.view_layer.objects.active
    cylinder.rotation_euler = (math.radians(90), 0, -math.radians(90))

    bool_modifier = model.modifiers.new(name="Boolean", type="BOOLEAN")
    bool_modifier.operation = "INTERSECT"
    bool_modifier.use_self = True
    bool_modifier.object = cylinder

    bpy.context.view_layer.objects.active = model
    if model.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.modifier_apply(modifier=bool_modifier.name)

    bpy.ops.object.select_all(action="DESELECT")
    model.select_set(True)
    bpy.context.view_layer.objects.active = model
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.separate(type="SELECTED")
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.data.objects.remove(cylinder)
    return bpy.context.object


def separate_wheel_mesh(coords, output_blend_path, export_glb_path=None):
    """Project wheel center onto active mesh and save separated Blender/GLB outputs."""
    _left_center, right_center = find_circle_centers(coords)

    camera = bpy.data.objects["Camera"]
    camera.location = (10, 0, 0)
    camera.rotation_euler = (radians(90), 0, radians(90))
    camera.data.ortho_scale = 2

    model = bpy.context.view_layer.objects.active
    if model is None or model.type != "MESH":
        mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
        if not mesh_objects:
            raise ValueError("No MESH object found in scene.")
        model = mesh_objects[0]

    cast_ray_and_create_cylinder(camera, model, right_center)
    bpy.ops.wm.save_as_mainfile(filepath=output_blend_path)
    if export_glb_path:
        bpy.ops.export_scene.gltf(filepath=export_glb_path, export_format="GLB")


def run_pipeline(input_glb_path, work_dir, export_glb=False):
    """Run full 6-view rendering -> YOLOv11-seg -> ellipse fit -> 3D boolean separation."""
    render_dir = os.path.join(work_dir, "views")
    os.makedirs(render_dir, exist_ok=True)
    render_multiview(input_glb_path, render_dir)

    right_image_path = os.path.join(render_dir, "RIGHT.png")
    coords = yolo_wheel_detection(right_image_path)
    if coords is None or len(coords) < 5:
        raise ValueError("Insufficient wheel mask pixels detected on RIGHT projection.")

    blend_path = os.path.join(work_dir, "segmented_model.blend")
    glb_path = os.path.join(work_dir, "segmented_model.glb") if export_glb else None
    separate_wheel_mesh(coords, blend_path, export_glb_path=glb_path)
    return glb_path if export_glb else blend_path


@app.route("/segment", methods=["POST"])
def segment():
    """Accept a .glb file upload and return the segmented .blend (or ?format=glb) file."""
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type; expected .glb"}), 400

    export_glb = request.args.get("format", "blend").lower() == "glb"
    temp_dir = tempfile.mkdtemp(dir=UPLOAD_FOLDER)
    try:
        safe_name = secure_filename(file.filename) or "input.glb"
        input_glb_path = os.path.join(temp_dir, safe_name)
        file.save(input_glb_path)

        output_path = run_pipeline(input_glb_path, temp_dir, export_glb=export_glb)
        download_name = "segmented_model.glb" if export_glb else "segmented_model.blend"
        return send_file(
            output_path,
            as_attachment=True,
            download_name=download_name,
            mimetype="application/octet-stream",
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    port = int(os.environ.get("SEGMENT_PORT", "3546"))
    app.run(host="0.0.0.0", port=port, use_reloader=False, threaded=False)
