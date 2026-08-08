# agx_wrap/world.py
import math
import numpy as np
import agx
import agxSDK
import agxCollide
import agxOSG
import agxRender
import agxModel

from agxPythonModules.utils.environment import simulation, root, application
from agxPythonModules.utils.callbacks import StepEventCallback as Sec
from agxPythonModules.utils.numpy_utils import wrap_vector_as_numpy_array


def _color(node, rgba):
    agxOSG.setDiffuseColor(node, agx.Vec4f(*rgba))
    agxOSG.setAmbientColor(node, agx.Vec4f(rgba[0]*0.3, rgba[1]*0.3, rgba[2]*0.3, rgba[3]))
    agxOSG.setSpecularColor(node, agx.Vec4f(0.3, 0.3, 0.3, 1))
    agxOSG.setShininess(node, 60)


# Ocean creation utility
def create_ocean(height=1.5, res_xy=(80, 80), size_xy=(500, 500), cell_h=20.0):
    """
    Builds animated heightfield-water + hydrodynamics registration (WindAndWaterController).
    Returns (hf, water_geom, wwc, wave_updater).
    """
    # Heightfield
    rx, ry = res_xy # Resolution
    sx, sy = size_xy # Size
    hf = agxCollide.HeightField(rx, ry, sx, sy, cell_h) # cell size in Z
    water_geom = agxCollide.Geometry(hf) # Water geometry
    water_geom.setMaterial(agx.Material("waterMaterial")) # Water material
    node = agxOSG.createVisual(water_geom, root()) # Visual node
    _color(node, (0.0, 0.74902, 1.0, 0.5)) # Water color (RGBA)

    simulation().add(water_geom) # Add water geometry to simulation

    # Hydrodynamics controller (water registration)
    wwc = agxModel.WindAndWaterController() # Hydrodynamics controller
    wwc.addWater(water_geom) # Register water geometry
    simulation().add(wwc) # Add controller to simulation

    # Pre-allocate heights vector and wrap as numpy
    heights = agx.RealVector(hf.getResolutionX() * hf.getResolutionY()) # Pre-allocated heights vector
    # Initialize heights to zero
    for _ in range(hf.getResolutionX() * hf.getResolutionY()):
        heights.append(0.0)

    # Wrap heights as numpy array for easy manipulation
    np_heights = wrap_vector_as_numpy_array(heights, np.float64).reshape(
        (hf.getResolutionX(), hf.getResolutionY())
    )

    # Build indices once
    jj = np.stack((np.arange(hf.getResolutionY()),) * hf.getResolutionX()) # Y indices
    ii = np.stack((np.arange(hf.getResolutionX()),) * hf.getResolutionY(), axis=1) # X indices

    # Wave parameters
    amp = float(height)

    # Wave function
    def wave_function(t: float):
        # Two-component traveling wave (tunable):
        np_heights[:] = amp * (
            0.40 * np.sin(1.00 * jj + 0.60 * t) +
            0.10 * np.sin(1.20 * ii + 0.60 * jj + 1.45 * t)
        ) 
        hf.setHeights(heights) # Update heightfield heights

    # Hook wave as PRE_COLLIDE so contacts see updated surface
    Sec.preCollideCallback(lambda t: wave_function(t)) # Pre-collide callback for wave update
    wave_function(0.0) # Initial wave update

    # Camera (optional nice default)
    cam = application().getCameraData() # Get camera data
    cam.eye = agx.Vec3(31.1, -87.59, 44.283) # Camera eye position
    cam.center = agx.Vec3(9.57, 2.147, 6.95) # Camera center position
    cam.up = agx.Vec3(-0.039, 0.375, 0.925) # Camera up vector
    cam.nearClippingPlane = 0.1 # Near clipping plane
    cam.farClippingPlane = 5000 # Far clipping plane
    application().applyCameraData(cam) # Apply camera data

    return hf, water_geom, wwc # Return heightfield, water geometry, and wind-and-water controller


# Body colorization utility
def colorize_body(rb: agx.RigidBody, rgba=(1.0, 1.0, 0.8, 1.0)):
    node = agxOSG.createVisual(rb, root()) # Create visual node for rigid body
    _color(node, rgba) # Apply color to the visual node


def _add_visual_geometry(geom, rgba):
    geom.setEnableCollisions(False)
    simulation().add(geom)
    node = agxOSG.createVisual(geom, root())
    _color(node, rgba)
    return node


def create_waypoint_marker(x, y, z=6.0, rgba=(0.2, 0.8, 0.2, 1.0)):
    pole = agxCollide.Geometry(agxCollide.Cylinder(0.6, 8.0))
    pole.setPosition(agx.Vec3(x, y, z))
    _add_visual_geometry(pole, rgba)
    top = agxCollide.Geometry(agxCollide.Sphere(1.5))
    top.setPosition(agx.Vec3(x, y, z + 5.0))
    _add_visual_geometry(top, rgba)
    return pole, top


def create_force_arrow(rgba=(1.0, 0.3, 0.0, 1.0)):
    body = agx.RigidBody()
    body.setMotionControl(agx.RigidBody.KINEMATICS)

    shaft = agxCollide.Geometry(agxCollide.Cylinder(0.25, 5.0))
    shaft.setEnableCollisions(False)
    body.add(shaft)

    tip = agxCollide.Geometry(agxCollide.Cylinder(0.7, 1.5))
    tip.setEnableCollisions(False)
    tip.setLocalPosition(agx.Vec3(0, 3.25, 0))
    body.add(tip)

    body.setPosition(agx.Vec3(0, 0, -100))
    simulation().add(body)
    node = agxOSG.createVisual(body, root())
    _color(node, rgba)
    return body, node


def create_trail_dot(x, y, z=4.0, rgba=(0.3, 0.5, 1.0, 0.6)):
    dot = agxCollide.Geometry(agxCollide.Sphere(0.5))
    dot.setPosition(agx.Vec3(x, y, z))
    dot.setEnableCollisions(False)
    simulation().add(dot)
    node = agxOSG.createVisual(dot, root())
    _color(node, rgba)
    agxOSG.setAlpha(node, rgba[3])
    return dot

