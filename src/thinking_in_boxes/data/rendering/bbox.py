# 3D bounding-box overlay and the bbox-face masks derived from it.

import blenderproc as bproc
import bpy
import cv2
import numpy as np
from mathutils import Matrix, Vector

COLOR_FRONT = (255, 80, 80)
COLOR_BACK = (80, 255, 80)
COLOR_TOP = (80, 80, 255)
COLOR_BOTTOM = (255, 255, 80)
COLOR_LEFT = (255, 80, 255)
COLOR_RIGHT = (80, 255, 255)


def _classify_face_color(fd, of, ou, orr):
    best_c, best_d = COLOR_FRONT, -2.0
    for ax, c in [
        (of, COLOR_FRONT),
        (-of, COLOR_BACK),
        (ou, COLOR_TOP),
        (-ou, COLOR_BOTTOM),
        (orr, COLOR_RIGHT),
        (-orr, COLOR_LEFT),
    ]:
        d = fd.dot(ax)
        if d > best_d:
            best_d, best_c = d, c
    return best_c


def _collect_bbox_faces(obj, cam_pose):
    bpy.context.scene.camera.matrix_world = Matrix(cam_pose)
    bo = obj.blender_obj
    pw = [bo.matrix_world @ Vector(v) for v in bo.bound_box]
    p2 = bproc.camera.project_points(np.array(pw)).astype(int)
    fi = [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 4, 5, 1),
        (1, 5, 6, 2),
        (2, 6, 7, 3),
        (3, 7, 4, 0),
    ]
    oc = sum(pw, Vector()) / 8
    r3 = bo.matrix_world.to_3x3()
    of = (r3 @ Vector((0, -1, 0))).normalized()
    ou = (r3 @ Vector((0, 0, 1))).normalized()
    orr = (r3 @ Vector((1, 0, 0))).normalized()
    cp = Vector(cam_pose[:3, 3])
    faces = []
    for f in fi:
        vw = [pw[i] for i in f]
        fc = sum(vw, Vector()) / 4
        fn = (vw[1] - vw[0]).cross(vw[3] - vw[0])
        if fn.length < 1e-8:
            continue
        fn = fn.normalized()
        if fn.dot(fc - oc) < 0:
            fn = -fn
        if (cp - fc).dot(fn) <= 0:
            continue
        fd = (fc - oc).normalized()
        faces.append(
            (
                (fc - cp).length,
                np.array([p2[i] for i in f]),
                _classify_face_color(fd, of, ou, orr),
            )
        )
    return faces


def draw_bboxes_and_masks(img, objects, cam_pose, fill_alpha=0.2):
    h, w = img.shape[:2]
    cp = Vector(cam_pose[:3, 3])

    per_object = []
    for oi, obj in enumerate(objects):
        faces = _collect_bbox_faces(obj, cam_pose)
        if not faces:
            continue
        bo = obj.blender_obj
        pw = [bo.matrix_world @ Vector(v) for v in bo.bound_box]
        od = (sum(pw, Vector()) / 8 - cp).length
        per_object.append((od, oi, faces))
    per_object.sort(key=lambda x: x[0])

    result = img.copy()
    occupied = np.zeros((h, w), dtype=np.uint8)
    obj_mask_buf = np.zeros((h, w), dtype=np.uint8)

    masks = [np.zeros((h, w), dtype=np.uint8) for _ in range(len(objects))]

    for _, oi, faces in per_object:
        obj_mask_buf[:] = 0
        for _, p, _ in faces:
            cv2.fillConvexPoly(obj_mask_buf, p, 255)

        visible = cv2.bitwise_and(obj_mask_buf, cv2.bitwise_not(occupied))
        masks[oi] = visible.copy()

        fs = sorted(faces, key=lambda x: -x[0])
        ol = result.copy()
        for _, p, _ in fs:
            cv2.fillConvexPoly(ol, p, (128, 128, 128))
        ov = ol.copy()
        for _, p, c in fs:
            cv2.fillConvexPoly(ov, p, c)
        ol = cv2.addWeighted(ov, fill_alpha, ol, 1.0 - fill_alpha, 0)
        for _, p, _ in fs:
            cv2.polylines(ol, [p], True, (0, 0, 0), 1)

        v3 = cv2.merge([visible, visible, visible])
        np.copyto(result, ol, where=(v3 > 0))

        cv2.bitwise_or(occupied, obj_mask_buf, dst=occupied)

    return result, masks
