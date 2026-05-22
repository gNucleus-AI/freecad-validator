#!/usr/bin/env python3

import logging
import os
import sys

import numpy as np
import pyvista as pv

# `FreeCAD` must precede `Mesh` — Mesh's C extension binds against FreeCAD
# symbols and segfaults on macOS otherwise. Route through the project
# loader so FreeCAD is discovered at common install paths without the
# caller having to set PYTHONPATH / FREECAD_LIB.
from freecad_validator._freecad_loader import import_freecad

FreeCAD = import_freecad()
import Mesh  # noqa: E402  — must come after FreeCAD import


# ----------------------------- FreeCAD → Mesh -----------------------------
def read_bbox_diagonal_from_freecad(freecad_path: str) -> float:
    """Return the diagonal length of the union of all solid-body bboxes, or 0.0 if none."""
    try:
        doc = FreeCAD.openDocument(freecad_path)
        FreeCAD.setActiveDocument(doc.Name)
        extents = None
        for obj in doc.Objects:
            if not (obj.TypeId.startswith("PartDesign::Body") or obj.TypeId.startswith("Part::Feature")):
                continue
            if not (hasattr(obj, "Shape") and obj.Shape and getattr(obj.Shape, "Faces", None)):
                continue
            bb = obj.Shape.BoundBox
            if extents is None:
                extents = [bb.XMin, bb.YMin, bb.ZMin, bb.XMax, bb.YMax, bb.ZMax]
            else:
                extents[0] = min(extents[0], bb.XMin)
                extents[3] = max(extents[3], bb.XMax)
                extents[1] = min(extents[1], bb.YMin)
                extents[4] = max(extents[4], bb.YMax)
                extents[2] = min(extents[2], bb.ZMin)
                extents[5] = max(extents[5], bb.ZMax)
        FreeCAD.closeDocument(doc.Name)
        if extents is None:
            return 0.0
        dx, dy, dz = extents[3]-extents[0], extents[4]-extents[1], extents[5]-extents[2]
        return float(np.sqrt(dx*dx + dy*dy + dz*dz))
    except Exception as e:
        logging.warning(f"Failed to read bbox from {freecad_path}: {e}")
        return 0.0


def read_tessellation_from_freecad(freecad_path: str, tolerance: float = 0.1):
    """
    Read tessellation data directly from a FreeCAD file.

    Args:
        freecad_path: Path to the FreeCAD file (.FCStd)
        tolerance: Linear deflection (mm) for the primary tessellation pass.

    Returns:
        tuple: (vertices, faces) as numpy arrays, or (None, None) if failed
    """
    try:
        # Open the FreeCAD document
        doc = FreeCAD.openDocument(freecad_path)
        FreeCAD.setActiveDocument(doc.Name)

        # Collect bodies that have solid geometry (can be tessellated)
        bodies = []
        for obj in doc.Objects:
            if obj.TypeId.startswith("PartDesign::Body") or obj.TypeId.startswith("Part::Feature"):
                if hasattr(obj, "Shape") and obj.Shape:
                    # Check if the shape has solid geometry (faces)
                    # Skip shapes that only have edges/wires (like spines, sketches, etc.)
                    if hasattr(obj.Shape, 'Faces') and len(obj.Shape.Faces) > 0:
                        bodies.append(obj)
                        logging.info(f"Added solid body: {obj.Name} with {len(obj.Shape.Faces)} faces")
                    else:
                        logging.info(f"Skipping non-solid feature: {obj.Name} (no faces, only edges/wires)")

        if not bodies:
            logging.error("No solid bodies with faces found in FreeCAD file (only edges/wires/spines)")
            FreeCAD.closeDocument(doc.Name)
            return None, None

        # Combine all shapes into one
        combined_shape = None
        for body in bodies:
            if combined_shape is None:
                combined_shape = body.Shape.copy()
            else:
                combined_shape = combined_shape.fuse(body.Shape)

        # Get tessellation data directly from the combined shape
        logging.info(f"Attempting tessellation with tolerance {tolerance}")
        tessellation = combined_shape.tessellate(tolerance)

        if tessellation and len(tessellation) >= 2:
            vertices, faces = tessellation
            logging.info(f"Raw tessellation: {len(vertices)} vertices, {len(faces)} faces")

            # Convert to numpy arrays
            vertices_array = np.array([[v.x, v.y, v.z] for v in vertices], dtype=float)
            faces_array = np.array(faces, dtype=np.int64)

            FreeCAD.closeDocument(doc.Name)
            logging.info(f"Loaded FreeCAD tessellation: {len(vertices_array)} vertices, {len(faces_array)} faces")
            return vertices_array, faces_array

        logging.warning(f"Tessellation with tolerance {tolerance} failed, trying different tolerances...")

        # Try different tolerances
        for tolerance in [0.01, 0.05, 0.2, 0.5]:
            logging.info(f"Trying tessellation with tolerance {tolerance}")
            tessellation = combined_shape.tessellate(tolerance)
            if tessellation and len(tessellation) >= 2:
                vertices, faces = tessellation
                logging.info(f"Raw tessellation (tolerance {tolerance}): {len(vertices)} vertices, {len(faces)} faces")

                if len(vertices) > 0 and len(faces) > 0:
                    vertices_array = np.array([[v.x, v.y, v.z] for v in vertices], dtype=float)
                    faces_array = np.array(faces, dtype=np.int64)

                    FreeCAD.closeDocument(doc.Name)
                    logging.info(f"Loaded FreeCAD tessellation: {len(vertices_array)} vertices, {len(faces_array)} faces")
                    return vertices_array, faces_array

        # If all tessellation attempts failed, try using Mesh module
        logging.warning("All tessellation attempts failed, trying Mesh module...")
        mesh_obj = Mesh.Mesh()
        mesh_obj.addMesh(Mesh.Mesh(combined_shape))

        if len(mesh_obj.Points) > 0 and len(mesh_obj.Facets) > 0:
            # Convert mesh to our format
            vertices_array = np.array([[p.x, p.y, p.z] for p in mesh_obj.Points], dtype=float)
            faces_array = np.array([[f[0], f[1], f[2]] for f in mesh_obj.Facets], dtype=np.int64)

            FreeCAD.closeDocument(doc.Name)
            logging.info(f"Loaded FreeCAD mesh: {len(vertices_array)} vertices, {len(faces_array)} faces")
            return vertices_array, faces_array

        logging.error("All tessellation methods failed")
        FreeCAD.closeDocument(doc.Name)
        return None, None

    except Exception as e:
        logging.error(f"Failed to read FreeCAD file {freecad_path}: {e}")
        return None, None

def read_edge_data_from_freecad(freecad_path: str, tolerance: float = 0.05):
    """
    Read edge data from a FreeCAD file.

    Args:
        freecad_path: Path to the FreeCAD file (.FCStd)
        tolerance: Deflection tolerance for discretize()

    Returns:
        list: Edge data as a list of ``{'points': np.ndarray}`` dicts,
        one per polyline.
    """
    try:
        # Open the FreeCAD document
        doc = FreeCAD.openDocument(freecad_path)
        FreeCAD.setActiveDocument(doc.Name)

        # Collect bodies that have solid geometry (can be tessellated)
        bodies = []
        for obj in doc.Objects:
            if obj.TypeId.startswith("PartDesign::Body") or obj.TypeId.startswith("Part::Feature"):
                if hasattr(obj, "Shape") and obj.Shape:
                    # Check if the shape has solid geometry (faces)
                    # Skip shapes that only have edges/wires (like spines, sketches, etc.)
                    if hasattr(obj.Shape, 'Faces') and len(obj.Shape.Faces) > 0:
                        bodies.append(obj)
                        logging.info(f"Added solid body: {obj.Name} with {len(obj.Shape.Faces)} faces")
                    else:
                        logging.info(f"Skipping non-solid feature: {obj.Name} (no faces, only edges/wires)")

        if not bodies:
            logging.error("No valid bodies found in FreeCAD file")
            FreeCAD.closeDocument(doc.Name)
            return None

        edges_data = []

        for body in bodies:
            for edge in body.Shape.Edges:
                # Quick check for invalid edges
                if not edge.isValid():
                    continue

                # Discretize the edge into points
                discretized_pts = edge.discretize(QuasiDeflection=tolerance)
                if len(discretized_pts) < 2:
                    continue

                # Convert to numpy array
                points = np.array([[pt.x, pt.y, pt.z] for pt in discretized_pts], dtype=float)
                edges_data.append({'points': points})

        FreeCAD.closeDocument(doc.Name)

        logging.info(f"Loaded edge data: {len(edges_data)} polylines")
        return edges_data

    except Exception as e:
        logging.error(f"Failed to read edge data from FreeCAD file {freecad_path}: {e}")
        return None


# ----------------------------- Rendering -----------------------------

def render_edges_only(edge_data, out_png: str, size: int = 512, edge_color: str = "black"):
    """
    Render only edges when no mesh data is available.
    """
    try:
        # Set up the plotter
        plotter = pv.Plotter(off_screen=True, window_size=[size, size])
        plotter.background_color = '#f0f0f0'

        # Add lighting
        plotter.add_light(pv.Light(position=(1, 1, 1), focal_point=(0, 0, 0), intensity=0.8))
        plotter.add_light(pv.Light(position=(-1, -1, 1), focal_point=(0, 0, 0), intensity=0.4))

        # Add all edges
        all_points = []
        for e in edge_data:
            points = e.get('points')
            if points is not None and points.shape[0] > 1:
                all_points.extend(points.tolist())

        if all_points:
            all_points = np.array(all_points)

            # Calculate bounds for camera setup
            bounds = [all_points[:, 0].min(), all_points[:, 0].max(),
                     all_points[:, 1].min(), all_points[:, 1].max(),
                     all_points[:, 2].min(), all_points[:, 2].max()]
            center = [(bounds[0] + bounds[1])/2, (bounds[2] + bounds[3])/2, (bounds[4] + bounds[5])/2]
            size_val = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])

            # Add each edge as a line
            for e in edge_data:
                points = e.get('points')
                if points is not None and points.shape[0] > 1:
                    # Create line segments between consecutive points
                    line_segments = []
                    for i in range(len(points) - 1):
                        line_segments.extend([points[i], points[i + 1]])

                    if line_segments:
                        line_segments = np.array(line_segments)
                        plotter.add_lines(line_segments, color=edge_color, width=2)

            # Set camera
            plotter.camera_position = 'iso'
            plotter.camera.focal_point = center
            plotter.camera.parallel_projection = True
            plotter.camera.parallel_scale = size_val * 0.8

        # Take screenshot
        plotter.screenshot(out_png, transparent_background=False)
        plotter.close()
        logging.info(f"Rendered edges-only PNG: {out_png}")
        return True

    except Exception as e:
        logging.error(f"Edge-only rendering failed: {e}", exc_info=True)
        return False

def render_tessellation_with_pyvista(
    vertices: np.ndarray,
    faces: np.ndarray,
    out_png: str,
    size: int = 512,
    edge_data=None,
    face_alpha: float = 1.0,
    mesh_color: str = "#87CEEB",
    edge_color: str = "black"
):
    """
    Renders a 3D mesh to a PNG using PyVista for a high-quality, smooth result.
    """
    try:
        # Debug logging for array shapes
        logging.info(f"Input arrays - vertices shape: {vertices.shape}, faces shape: {faces.shape}")
        logging.info(f"Vertices dtype: {vertices.dtype}, faces dtype: {faces.dtype}")

        # Validate input arrays
        if len(vertices) == 0:
            logging.warning("No vertices provided - will render edges only")
            # If we have edge data but no mesh, we can still render edges
            if edge_data and len(edge_data) > 0:
                logging.info("Rendering edges only since no mesh data available")
                return render_edges_only(edge_data, out_png, size, edge_color)
            else:
                logging.error("No vertices and no edge data provided")
                return False

        if len(faces) == 0:
            logging.warning("No faces provided - will render edges only")
            # If we have edge data but no faces, we can still render edges
            if edge_data and len(edge_data) > 0:
                logging.info("Rendering edges only since no face data available")
                return render_edges_only(edge_data, out_png, size, edge_color)
            else:
                logging.error("No faces and no edge data provided")
                return False

        # Ensure faces is 2D array
        if faces.ndim == 1:
            logging.warning(f"Faces array is 1D with shape {faces.shape}, reshaping...")
            # If faces is 1D, assume it's a flat array of vertex indices
            # We need to reshape it to (num_faces, 3) for triangles
            if len(faces) % 3 != 0:
                logging.error(f"Cannot reshape faces array of length {len(faces)} into triangles")
                return False
            faces = faces.reshape(-1, 3)
            logging.info(f"Reshaped faces to: {faces.shape}")

        # 1. PyVista requires faces to be in a specific format:
        # [num_vertices_in_face, v1_idx, v2_idx, v3_idx, ...]
        # For triangles, this is [3, v1, v2, v3] for each face.
        num_faces = faces.shape[0]

        # Ensure faces has the right number of columns (3 for triangles)
        if faces.shape[1] != 3:
            logging.error(f"Expected faces to have 3 columns (triangles), but got {faces.shape[1]}")
            return False

        # Create an array of '3's, one for each face
        num_vertices_per_face = np.full((num_faces, 1), 3)

        # Debug the shapes before concatenation
        logging.info(f"Before concatenation - num_vertices_per_face shape: {num_vertices_per_face.shape}, faces shape: {faces.shape}")

        # Prepend the '3's to the faces array
        pv_faces = np.hstack((num_vertices_per_face, faces)).flatten()

        # 2. Create the PyVista mesh object
        mesh = pv.PolyData(vertices, pv_faces)

        # 3. Set up the plotter (the rendering scene)
        # off_screen=True is crucial for running without a pop-up window
        plotter = pv.Plotter(off_screen=True, window_size=[size, size])

        # Set CAD-like background (light gray)
        plotter.background_color = '#f0f0f0'

        # Add multiple light sources for technical CAD lighting
        plotter.add_light(pv.Light(position=(1, 1, 1), focal_point=(0, 0, 0), intensity=0.8))
        plotter.add_light(pv.Light(position=(-1, -1, 1), focal_point=(0, 0, 0), intensity=0.4))
        plotter.add_light(pv.Light(position=(0, 0, -1), focal_point=(0, 0, 0), intensity=0.3))

        # 4. Add the mesh to the plotter with CAD-like appearance
        plotter.add_mesh(
            mesh,
            color=mesh_color,
            smooth_shading=False,  # Faceted look for CAD models
            opacity=face_alpha,
            show_edges=False,  # Hide tessellation edges for cleaner look
            edge_color=edge_color,
            line_width=1,  # Thin edges
            specular=0.1, # Less shiny for technical look
            specular_power=5,
            metallic=0.8,  # Metallic finish
            roughness=0.3  # Slightly rough surface
        )

        # 5. (Optional) Add the separate B-Rep edges with cleaner rendering
        if edge_data:
            for e in edge_data:
                points = e.get('points')
                if points is not None and points.shape[0] > 1:
                    # Create line segments between consecutive points
                    line_segments = []
                    for i in range(len(points) - 1):
                        line_segments.extend([points[i], points[i + 1]])

                    if line_segments:
                        line_segments = np.array(line_segments)
                        plotter.add_lines(line_segments, color=edge_color, width=2)

        # 6. Set camera to orthographic view for CAD-like perspective
        # Calculate bounds for proper orthographic setup
        bounds = mesh.bounds
        center = mesh.center
        size = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])

        # Set orthographic camera
        plotter.camera_position = 'iso'  # Isometric view
        plotter.camera.focal_point = center
        plotter.camera.parallel_projection = True  # Enable orthographic projection
        plotter.camera.parallel_scale = size * 0.8  # Adjust scale for proper framing

        # 7. Take the screenshot
        plotter.screenshot(out_png, transparent_background=False)
        plotter.close()
        logging.info(f"Rendered PNG with PyVista: {out_png}")
        return True

    except Exception as e:
        logging.error(f"PyVista rendering failed: {e}", exc_info=True)
        return False


# ----------------------------- Orchestrator -----------------------------
def render_freecad_file(
    freecad_path: str,
    output_png_path: str,
    *,
    finer_mesh: bool = False,
):
    """
    Render a FreeCAD file to PNG by extracting tessellation and edge data.

    Args:
        freecad_path: Path to the FreeCAD file (.FCStd)
        output_png_path: Output PNG file path
        finer_mesh: When False (default), tessellate with fixed tolerances
            (mesh=0.1mm, edge=0.05mm) — memory-bounded for small/medium
            parts. When True, scale tolerances to the part bounding-box
            diagonal for higher-fidelity output (can 10×+ vertex/face counts
            on small parts; only enable where memory headroom is available).

    Returns:
        bool: True if successful, False otherwise
    """
    out_dir = os.path.dirname(output_png_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if finer_mesh:
        # Adaptive: scale linear deflection to bbox diagonal so small parts
        # don't render as polygons and large parts don't get over-tessellated.
        bbox_diag = read_bbox_diagonal_from_freecad(freecad_path)
        if bbox_diag > 0:
            mesh_tol = max(min(bbox_diag * 0.001, 0.1), 1e-4)
            edge_tol = max(min(bbox_diag * 0.0005, 0.05), 5e-5)
        else:
            mesh_tol, edge_tol = 0.1, 0.05
        logging.info(f"finer_mesh=True; bbox_diag={bbox_diag:.3f}mm, mesh_tol={mesh_tol:.5f}, edge_tol={edge_tol:.5f}")
    else:
        mesh_tol, edge_tol = 0.1, 0.05

    # Read edge data from FreeCAD file
    edges = read_edge_data_from_freecad(freecad_path, tolerance=edge_tol)
    if edges is None:
        logging.error("Failed to read edge data from FreeCAD file")
        return False

    # Read tessellation data from FreeCAD file
    vertices, faces = read_tessellation_from_freecad(freecad_path, tolerance=mesh_tol)
    if vertices is None or faces is None:
        logging.error("Failed to read FreeCAD tessellation")
        return False

    logging.info(f"Tessellation data: {len(vertices)} vertices, {len(faces)} faces")
    logging.info(f"Vertex shape: {vertices.shape if vertices is not None else 'None'}")
    logging.info(f"Face shape: {faces.shape if faces is not None else 'None'}")
    if faces is not None and len(faces) > 0:
        logging.info(f"Face sample: {faces[0] if len(faces) > 0 else 'Empty'}")

    logging.info(f"Rendering with {len(vertices)} vertices, {len(faces)} faces, {len(edges)} edges")
    return render_tessellation_with_pyvista(
        vertices, faces, output_png_path,
        size=512, edge_data=edges,
        face_alpha=1,
        mesh_color="#c0c0c0", edge_color="#404040"  # Metallic gray for CAD look
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        stream=sys.stderr,
    )
    if len(sys.argv) != 3:
        sys.exit(2)
    sys.exit(0 if render_freecad_file(sys.argv[1], sys.argv[2]) else 1)
